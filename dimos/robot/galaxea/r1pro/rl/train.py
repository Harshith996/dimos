# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Bounded CPU RSL-RL PPO smoke, checkpoint reload and deterministic replay."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
import torch

from dimos.robot.galaxea.r1pro.rl.env import TrackingConfig, TrackingEnv
from dimos.robot.galaxea.r1pro.rl.model import (
    ARM_JOINT_NAMES,
    MANIFEST_PATH,
    WRIST_SITE_NAMES,
    build_xml,
    collision_pairs,
    load_model,
)
from dimos.robot.galaxea.r1pro.rl.targets import (
    FkTrajectorySource,
    TargetFrame,
    generate_joint_trajectory,
)


class TrackingVecEnv(VecEnv):  # type: ignore[misc]  # RSL-RL 3.1.2 has no py.typed marker.
    """Sequential independent MuJoCo lanes with RSL-RL episode reset semantics."""

    def __init__(
        self,
        config: TrackingConfig,
        num_envs: int = 2,
        source_root: Path | None = None,
        target_sources: list[Callable[[float], TargetFrame | None]] | None = None,
    ) -> None:
        if not 1 <= num_envs <= 4:
            raise ValueError("Preparation smoke supports 1 to 4 CPU environments.")
        if target_sources is not None and len(target_sources) != num_envs:
            raise ValueError("Each environment needs its own target source.")
        self.envs = [
            TrackingEnv(load_model(source_root), ARM_JOINT_NAMES, WRIST_SITE_NAMES, config)
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self.num_actions = 14
        self.max_episode_length = config.episode_steps
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = asdict(config)
        self._target_sources = target_sources
        self._hold_targets: list[TargetFrame] = []
        self.last_infos: list[dict[str, Any]] = []
        for env in self.envs:
            env.reset()
            self._hold_targets.append(env.current_pose_target())

    def get_observations(self) -> TensorDict:
        for i, env in enumerate(self.envs):
            target = (
                replace(self._hold_targets[i], stamp_s=env.time_s)
                if self._target_sources is None
                else self._target_sources[i](env.time_s)
            )
            env.set_target(target)
        obs = np.stack([env.observation() for env in self.envs])
        return TensorDict({"policy": torch.tensor(obs, dtype=torch.float32)}, [self.num_envs])

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError("Batched actions have the wrong shape.")
        rewards, dones, timeouts = [], [], []
        self.last_infos = []
        for i, env in enumerate(self.envs):
            _, reward, terminated, truncated, info = env.step(actions[i].detach().cpu().numpy())
            rewards.append(reward)
            dones.append(terminated or truncated)
            timeouts.append(truncated)
            self.last_infos.append(info)
            if terminated or truncated:
                info["terminal_observation"] = env.observation().tolist()
                env.reset()
            self.episode_length_buf[i] = env.episode_length
        terms = [info["reward_terms"] for info in self.last_infos if "reward_terms" in info]
        logs = {"safety/terminations": float(sum(dones))}
        if terms:
            logs.update(
                {
                    "reward/" + name: float(np.mean([row[name] for row in terms]))
                    for name in terms[0]
                }
            )
        return (
            self.get_observations(),
            torch.tensor(rewards),
            torch.tensor(dones),
            {"time_outs": torch.tensor(timeouts), "log": logs},
        )


def procedural_source(source_root: Path | None, seed: int) -> FkTrajectorySource:
    """Build a small sampled FK curriculum, without a dynamics feasibility claim."""
    model = load_model(source_root)
    scratch = TrackingEnv(model, ARM_JOINT_NAMES, WRIST_SITE_NAMES)
    home = np.zeros(14)

    def is_valid(qpos: NDArray[np.float64]) -> bool:
        scratch.reset(qpos=qpos)
        return not collision_pairs(model, scratch.data)

    def fk(qpos: NDArray[np.float64]) -> TargetFrame:
        scratch.reset(qpos=qpos)
        return scratch.current_pose_target()

    trajectory = generate_joint_trajectory(
        model.jnt_range[:, 0],
        model.jnt_range[:, 1],
        home,
        seed=seed,
        split="train",
        scenario="excursion",
        is_valid=is_valid,
        duration_s=1.0,
        amplitude_rad=0.02,
    )
    return FkTrajectorySource(trajectory, fk)


def smoke_runner_config() -> dict[str, Any]:
    return {
        "num_steps_per_env": 8,
        "save_interval": 1,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [32, 32],
            "critic_hidden_dims": [32, 32],
            "activation": "elu",
            "init_noise_std": 0.1,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 0.0003,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.0,
        },
    }


def replay(runner: OnPolicyRunner, env: TrackingVecEnv, steps: int) -> list[dict[str, Any]]:
    """Replay a loaded checkpoint on current hold targets, without policy updates."""
    policy = runner.get_inference_policy(device="cpu")
    rows = []
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(env.get_observations())
            _, rewards, dones, _ = env.step(actions)
            if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                raise ValueError("Replay produced nonfinite actions or rewards.")
            rows.append(
                {
                    "actions": actions.tolist(),
                    "rewards": rewards.tolist(),
                    "dones": dones.tolist(),
                    "infos": env.last_infos,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument("--target-mode", choices=["hold", "procedural"], default="hold")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 2:
        parser.error("The model/control integration gate limits this CLI to 1 or 2 smoke updates.")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = (
        TrackingConfig(**json.loads(args.config.read_text())) if args.config else TrackingConfig()
    )
    if config.episode_steps > 100:
        parser.error("Preparation episodes are capped at 100 steps.")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = (
        [procedural_source(args.source_root, args.seed + i) for i in range(args.num_envs)]
        if args.target_mode == "procedural"
        else None
    )
    source_callbacks: list[Callable[[float], TargetFrame | None]] | None = (
        [source.sample for source in sources] if sources is not None else None
    )
    env = TrackingVecEnv(config, args.num_envs, args.source_root, source_callbacks)
    runner_cfg = smoke_runner_config()
    runner = OnPolicyRunner(env, copy.deepcopy(runner_cfg), str(args.output), device="cpu")
    if args.replay_checkpoint:
        runner.load(str(args.replay_checkpoint), load_optimizer=False)
        checkpoint = args.replay_checkpoint
    else:
        before = [p.detach().clone() for p in runner.alg.policy.parameters()]
        runner.learn(args.iterations)
        if not any(
            not torch.equal(a, b)
            for a, b in zip(before, runner.alg.policy.parameters(), strict=True)
        ):
            raise RuntimeError("PPO smoke did not update any policy parameter.")
        checkpoint = args.output / f"model_{args.iterations - 1}.pt"
    for p in runner.alg.policy.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("Checkpoint parameters are nonfinite.")
    fresh_env = TrackingVecEnv(config, args.num_envs, args.source_root, source_callbacks)
    fresh_runner = OnPolicyRunner(fresh_env, smoke_runner_config(), device="cpu")
    fresh_runner.load(str(checkpoint), load_optimizer=False)
    rows = replay(fresh_runner, fresh_env, steps=8)
    (args.output / "replay.json").write_text(json.dumps(rows, indent=2))
    manifest = {
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "status": "plumbing_smoke_only",
        "task_mastery_validated": False,
        "device": "cpu",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "updates": 0 if args.replay_checkpoint else args.iterations,
        "samples": 0 if args.replay_checkpoint else args.iterations * 8 * args.num_envs,
        "target_distribution": args.target_mode + "_not_teleoperation",
        "deployment_matched": False,
        "model_assets": json.loads(MANIFEST_PATH.read_text()),
        "simulation_adapter": {
            "kp": 80,
            "kv": 8,
            "joint_damping": 2,
            "physics_dt_s": 0.002,
            "deployment_matched": False,
        },
        "reward": {
            "position_weight": 4.0,
            "orientation_weight": 2.0,
            "command_rate_weight": 0.0,
            "joint_velocity_weight": 0.0,
            "elbow_chest": "deferred",
            "tuning_validated": False,
        },
        "model_input_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "model_xml_sha256": hashlib.sha256(build_xml(args.source_root).encode()).hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "mujoco", "rsl-rl-lib")
        },
        "environment": asdict(config),
        "runner": runner_cfg,
        "integration_gate": "Model dynamics, collision validity and approved hardware control contract remain required before larger training.",
    }
    target_manifest = (
        [
            {
                "times_s": source.trajectory.times_s.tolist(),
                "knots_rad": source.trajectory.knots_rad.tolist(),
                "seed": source.trajectory.seed,
                "split": source.trajectory.split,
                "scenario": source.trajectory.scenario,
            }
            for source in sources
        ]
        if sources
        else {"mode": "hold"}
    )
    target_json = json.dumps(target_manifest, sort_keys=True)
    (args.output / "targets.json").write_text(target_json)
    manifest["targets_sha256"] = hashlib.sha256(target_json.encode()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "device",
                    "updates",
                    "samples",
                    "checkpoint",
                    "checkpoint_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
