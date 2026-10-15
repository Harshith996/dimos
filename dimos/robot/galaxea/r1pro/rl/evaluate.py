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


"""Run a bounded held-out simulation evaluation independently of reward."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
import torch

from dimos.robot.galaxea.r1pro.rl.env import TrackingConfig, TrackingEnv
from dimos.robot.galaxea.r1pro.rl.evaluation import EpisodeTrace, evaluate_episode
from dimos.robot.galaxea.r1pro.rl.model import (
    ARM_JOINT_NAMES,
    MANIFEST_PATH,
    WRIST_SITE_NAMES,
    build_xml,
    collision_pairs,
    joint_addresses,
    load_model,
)
from dimos.robot.galaxea.r1pro.rl.targets import (
    HELDOUT_MANIFEST,
    EvaluationCase,
    FkTrajectorySource,
    TargetFrame,
    TargetSource,
    generate_joint_trajectory,
)
from dimos.robot.galaxea.r1pro.rl.train import TrackingVecEnv, smoke_runner_config

Array = NDArray[np.float64]


def make_fk_source(model: mujoco.MjModel, case: EvaluationCase) -> FkTrajectorySource:
    """Use independent FK state so target generation cannot mutate the simulation."""
    data = mujoco.MjData(model)
    qids, _ = joint_addresses(model)
    sites = [model.site(name).id for name in WRIST_SITE_NAMES]
    limits = np.array([model.jnt_range[model.joint(name).id] for name in ARM_JOINT_NAMES])

    def valid(q: Array) -> bool:
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        return not collision_pairs(model, data)

    def fk(q: Array) -> TargetFrame:
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        quat = np.empty((2, 4))
        for index, site in enumerate(sites):
            mujoco.mju_mat2Quat(quat[index], data.site_xmat[site])  # type: ignore[attr-defined]  # Compiled MuJoCo API lacks this stub.
        return TargetFrame(data.site_xpos[sites].copy(), quat, 0.0)

    path = generate_joint_trajectory(
        limits[:, 0],
        limits[:, 1],
        np.zeros(14),
        seed=case.seed,
        split="heldout",
        scenario=case.scenario,
        duration_s=case.duration_s,
        is_valid=valid,
    )
    return FkTrajectorySource(path, fk, dropout_s=case.dropout_s, stale_s=case.stale_s)


def run_case(
    env: TrackingEnv,
    source: TargetSource,
    action_fn: Callable[[Array], Array],
    *,
    requested_duration_s: float,
) -> tuple[EpisodeTrace | None, dict[str, Any]]:
    """Do not autoreset or discard failed frames; record exact requested duration."""
    env.reset()
    rows: list[dict[str, Any]] = []
    last = env.current_pose_target()
    ended = False
    reason = "duration_reached"

    def record(target: TargetFrame) -> None:
        positions, quat = env.wrist_pose()
        qpos = env.joint_positions()
        limits = np.array(
            [env.model.jnt_range[env.model.joint(name).id] for name in ARM_JOINT_NAMES]
        )
        rows.append(
            {
                "time_s": env.time_s,
                "target_positions_m": target.positions_m.tolist(),
                "actual_positions_m": positions.tolist(),
                "target_quaternions_wxyz": target.quaternions_wxyz.tolist(),
                "actual_quaternions_wxyz": quat.tolist(),
                "stamp_s": target.stamp_s,
                "valid": target.valid,
                "position_valid": target.position_valid.tolist(),
                "orientation_valid": target.orientation_valid.tolist(),
                "collision": bool(collision_pairs(env.model, env.data)),
                "limit_violation": bool(np.any(qpos < limits[:, 0]) or np.any(qpos > limits[:, 1])),
                "limited_joint_command_rad": env.command_positions_rad.tolist(),
            }
        )

    for _ in range(int(np.ceil(requested_duration_s / env.config.control_dt_s))):
        frame = source.sample(env.time_s)
        target = replace(last, stamp_s=env.time_s, valid=False) if frame is None else frame
        last = target
        record(target)
        env.set_target(target)
        action = np.asarray(action_fn(env.observation()), dtype=float)
        rows[-1]["raw_action"] = action.tolist()
        _, _, terminated, truncated, info = env.step(action, target)
        rows[-1]["step_info"] = info
        if terminated or truncated:
            ended = terminated or env.time_s < requested_duration_s - 1e-9
            reason = str(info.get("termination_reason", "episode_truncated"))
            break
    if env.time_s > rows[-1]["time_s"] + 1e-9:
        frame = source.sample(env.time_s)
        record(replace(last, valid=False) if frame is None else frame)
    if len(rows) < 4:
        return None, {
            "completed": 0,
            "requested_duration_s": requested_duration_s,
            "observed_duration_s": env.time_s,
            "failure": "too_few_samples",
            "termination_reason": reason,
            "trace": rows,
        }
    trace = EpisodeTrace(
        np.array([r["time_s"] for r in rows]),
        np.array([r["target_positions_m"] for r in rows]),
        np.array([r["actual_positions_m"] for r in rows]),
        np.array([r["target_quaternions_wxyz"] for r in rows]),
        np.array([r["actual_quaternions_wxyz"] for r in rows]),
        np.array([r["stamp_s"] for r in rows]),
        np.array([r["valid"] for r in rows], bool),
        np.array([r["collision"] for r in rows], bool),
        np.array([r["limit_violation"] for r in rows], bool),
        np.array([r["position_valid"] for r in rows], bool),
        np.array([r["orientation_valid"] for r in rows], bool),
    )
    metrics = evaluate_episode(
        trace,
        target_timeout_s=env.config.max_target_age_s,
        requested_duration_s=requested_duration_s,
        terminated_early=ended,
    )
    return trace, {"metrics": metrics, "termination_reason": reason, "trace": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="Existing local RSL-RL smoke checkpoint; no policy updates"
    )
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--case", type=int, default=0, choices=range(len(HELDOUT_MANIFEST)))
    args = parser.parse_args()
    if not 0.08 <= args.duration <= 2 or not np.isclose(
        args.duration / 0.02, round(args.duration / 0.02)
    ):
        parser.error(
            "Preparation evaluator supports multiples of 0.02 seconds from 0.08 to 2 seconds; larger evaluation needs the integration gate."
        )
    case = replace(HELDOUT_MANIFEST[args.case], duration_s=args.duration)
    # Fault timing scales with the requested short smoke duration.
    if case.dropout_s is not None:
        case = replace(case, dropout_s=(args.duration * 0.4, args.duration * 0.7))
    if case.stale_s is not None:
        case = replace(case, stale_s=(args.duration * 0.4, args.duration * 0.9))
    model = load_model(args.source_root)
    config = TrackingConfig(episode_steps=int(np.ceil(args.duration / 0.02)))
    env = TrackingEnv(model, ARM_JOINT_NAMES, WRIST_SITE_NAMES, config)
    source = make_fk_source(model, case)

    def zero_action(obs: Array) -> Array:
        return np.zeros(14)

    action_fn: Callable[[Array], Array] = zero_action
    if args.checkpoint is not None:
        torch.set_num_threads(2)
        policy_env = TrackingVecEnv(config, num_envs=1, source_root=args.source_root)
        runner = OnPolicyRunner(policy_env, smoke_runner_config(), device="cpu")
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device="cpu")

        def checkpoint_action(obs: Array) -> Array:
            with torch.inference_mode():
                batch = TensorDict({"policy": torch.tensor(obs, dtype=torch.float32)[None]}, [1])
                return np.asarray(policy(batch)[0].detach().cpu().numpy(), dtype=np.float64)

        action_fn = checkpoint_action
    _, result = run_case(env, source, action_fn, requested_duration_s=args.duration)
    manifest = {
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "case": asdict(case),
        "controller": "RSL_RL_checkpoint" if args.checkpoint else "debug_zero_rate_hold",
        "config": asdict(config),
        "model_sha256": hashlib.sha256(build_xml(args.source_root).encode()).hexdigest(),
        "asset_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "dynamics_validated": False,
        "unsupported_metrics": [
            "overshoot",
            "settling_time",
            "production_IK_comparison",
            "hardware_latency",
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metrics": result.get("metrics"),
                "termination_reason": result["termination_reason"],
            }
        )
    )


if __name__ == "__main__":
    main()
