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
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-untyped]
from tensordict import TensorDict  # type: ignore[import-untyped]
import torch

from dimos.robot.galaxea.r1pro.rl.controllers import DifferentialIKController
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
    source_root as resolve_asset_root,
)
from dimos.robot.galaxea.r1pro.rl.recorded_targets import RecordedTargetSource
from dimos.robot.galaxea.r1pro.rl.targets import (
    HELDOUT_MANIFEST,
    EvaluationCase,
    FkTrajectorySource,
    FrozenTargetSource,
    JointTrajectory,
    TargetFrame,
    TargetSource,
    generate_joint_trajectory,
)
from dimos.robot.galaxea.r1pro.rl.train import TrackingVecEnv, smoke_runner_config

Array = NDArray[np.float64]
FROZEN_MANIFEST_PATH = MANIFEST_PATH.with_name("r1pro_rl_evaluation.json")
GENERATOR_CONFIG: dict[str, Any] = {
    "max_speed_rad_s": 0.2,
    "margin_rad": 0.02,
    "amplitude_rad": 0.08,
    "validation_samples": 401,
}

SERIALIZATION_CONFIG = {
    "wrist_component_decimal_places": 10,
    "wrist_component_absolute_error_bound": 5.01e-11,
    "rounding": "numpy_round",
    "omitted_fields": "TargetFrame defaults only",
    "joint_knots_quantized": False,
}


def make_fk_source(
    model: mujoco.MjModel, case: EvaluationCase, trajectory: JointTrajectory | None = None
) -> FkTrajectorySource:
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

    path = (
        trajectory
        if trajectory is not None
        else generate_joint_trajectory(
            limits[:, 0],
            limits[:, 1],
            np.zeros(14),
            seed=case.seed,
            split="heldout",
            scenario=case.scenario,
            duration_s=case.duration_s,
            is_valid=valid,
            **GENERATOR_CONFIG,
        )
    )
    return FkTrajectorySource(path, fk, dropout_s=case.dropout_s, stale_s=case.stale_s)


def content_sha256(value: Any) -> str:
    """Hash canonical JSON; reject nonfinite numbers rather than serializing NaN."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def model_identity(asset_root: Path | None) -> dict[str, Any]:
    converter = Path(__file__).with_name("model.py")
    generator = Path(__file__).with_name("targets.py")
    assets = json.loads(MANIFEST_PATH.read_text())
    normalized_xml = build_xml(asset_root).replace(
        str(resolve_asset_root(asset_root)), "<ASSET_ROOT>"
    )
    return {
        "repository": assets["repository"],
        "commit": assets["commit"],
        "package": assets["package"],
        "asset_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "model_xml_sha256": hashlib.sha256(normalized_xml.encode()).hexdigest(),
        "model_converter_sha256": hashlib.sha256(converter.read_bytes()).hexdigest(),
        "target_source_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "mujoco_version": importlib.metadata.version("mujoco"),
        "numpy_version": np.__version__,
        "joint_names": list(ARM_JOINT_NAMES),
        "wrist_sites": list(WRIST_SITE_NAMES),
        "reference_frame": "world",
        "tcp_calibrated": False,
    }


def sampled_kinematic_evidence(model: mujoco.MjModel, path: JointTrajectory) -> dict[str, Any]:
    """Measure reported contacts and limits, never claim a global clearance bound."""
    data = mujoco.MjData(model)
    qids, _ = joint_addresses(model)
    limits = np.array([model.jnt_range[model.joint(name).id] for name in ARM_JOINT_NAMES])
    times = np.linspace(0, path.times_s[-1], GENERATOR_CONFIG["validation_samples"])
    records = []
    min_margin = float("inf")
    min_contact: float | None = None
    forbidden_samples = 0
    for time in times:
        q = path.sample(float(time))
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        distances = [float(data.contact[index].dist) for index in range(data.ncon)]
        if distances:
            value = min(distances)
            min_contact = value if min_contact is None else min(min_contact, value)
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        min_margin = min(min_margin, margin)
        pairs = collision_pairs(model, data, penetration_tolerance_m=1e-5)
        forbidden_samples += bool(pairs)
        records.append(
            {
                "time_s": float(time),
                "qpos_rad": q.tolist(),
                "reported_contact_distances_m": distances,
                "forbidden_pairs": pairs,
                "joint_limit_margin_rad": margin,
            }
        )
    if forbidden_samples or min_margin < GENERATOR_CONFIG["margin_rad"] - 1e-12:
        raise ValueError("Frozen path fails sampled collision or joint-limit constraints")
    return {
        "samples": len(times),
        "interval_s": [0.0, float(path.times_s[-1])],
        "sample_records_sha256": content_sha256(records),
        "forbidden_collision_samples": forbidden_samples,
        "minimum_joint_limit_margin_rad": min_margin,
        "minimum_reported_contact_distance_m": min_contact,
        "maximum_reported_penetration_m": max(0.0, -min_contact)
        if min_contact is not None
        else 0.0,
        "penetration_tolerance_m": 1e-5,
        "collision_semantics": "Only MuJoCo-reported enabled contacts with distance < -1e-5 m are violations. Model exclusions apply. No positive global clearance is established.",
        "dynamics_validated": False,
        "continuous_collision_checked": False,
    }


def serialized_samples(
    source: FkTrajectorySource, duration_s: float, control_dt_s: float
) -> list[dict[str, Any]]:
    result = []
    for time in np.linspace(0, duration_s, round(duration_s / control_dt_s) + 1):
        frame = source.sample(float(time))
        payload = None
        if frame is not None:
            payload = {
                "positions_m": np.round(frame.positions_m, 10).tolist(),
                "quaternions_wxyz": np.round(frame.quaternions_wxyz, 10).tolist(),
                "stamp_s": frame.stamp_s,
            }
            if not frame.valid:
                payload["valid"] = False
            if not np.all(frame.position_valid):
                payload["position_valid"] = frame.position_valid.tolist()
            if not np.all(frame.orientation_valid):
                payload["orientation_valid"] = frame.orientation_valid.tolist()
            if frame.reference_frame != "world":
                payload["reference_frame"] = frame.reference_frame
            if frame.provenance != "procedural_fk":
                payload["provenance"] = frame.provenance
            if frame.calibration_id is not None:
                payload["calibration_id"] = frame.calibration_id
            if frame.elbow_positions_m is not None:
                payload["elbow_positions_m"] = frame.elbow_positions_m.tolist()
            if frame.chest_quaternion_wxyz is not None:
                payload["chest_quaternion_wxyz"] = frame.chest_quaternion_wxyz.tolist()
        result.append({"query_time_s": float(time), "frame": payload})
    return result


def freeze_manifest(
    model: mujoco.MjModel, config: TrackingConfig, asset_root: Path | None
) -> dict[str, Any]:
    """Realize the small suite once. Evaluation subsequently reuses these samples."""
    duration = config.episode_steps * config.control_dt_s
    cases = []
    for template in HELDOUT_MANIFEST:
        case = replace(
            template,
            duration_s=duration,
            dropout_s=(round(duration * 0.4, 12), round(duration * 0.7, 12))
            if template.dropout_s is not None
            else None,
            stale_s=(round(duration * 0.4, 12), round(duration * 0.9, 12))
            if template.stale_s is not None
            else None,
        )
        source = make_fk_source(model, case)
        path = source.trajectory
        samples = serialized_samples(source, duration, config.control_dt_s)
        cases.append(
            {
                "case": asdict(case),
                "joint_times_s": path.times_s.tolist(),
                "joint_knots_rad": path.knots_rad.tolist(),
                "samples": samples,
                "samples_sha256": content_sha256(samples),
                "sampled_kinematic_evidence": sampled_kinematic_evidence(model, path),
            }
        )
    manifest = {
        "schema_version": 1,
        "identity": model_identity(asset_root),
        "control_config": asdict(config),
        "generator_config": GENERATOR_CONFIG,
        "target_serialization": SERIALIZATION_CONFIG,
        "interpolation": "quintic_zero_endpoint_velocity_acceleration",
        "cases": cases,
        "purpose": "small_simulation_kinematic_regression_not_human_motion",
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    return manifest


def load_frozen_manifest(
    path: Path, model: mujoco.MjModel, config: TrackingConfig, asset_root: Path | None
) -> dict[str, Any]:
    """Verify stored hashes and recompute FK/contact evidence against current assets."""
    manifest: dict[str, Any] = json.loads(path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != content_sha256(payload):
        raise ValueError("Frozen manifest digest mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("identity") != model_identity(
        asset_root
    ):
        raise ValueError("Frozen manifest model, source or runtime identity changed")
    if (
        manifest.get("control_config") != asdict(config)
        or manifest.get("generator_config") != GENERATOR_CONFIG
        or manifest.get("target_serialization") != SERIALIZATION_CONFIG
    ):
        raise ValueError("Frozen manifest control or generator configuration changed")
    if len(manifest["cases"]) != 4:
        raise ValueError("Frozen evaluation requires its four reviewed cases")
    for entry in manifest["cases"]:
        case = EvaluationCase(**entry["case"])
        path_data = JointTrajectory(
            np.array(entry["joint_times_s"]),
            np.array(entry["joint_knots_rad"]),
            "heldout",
            case.scenario,
            case.seed,
        )
        if (
            path_data.times_s.shape != (5,)
            or path_data.knots_rad.shape != (5, 14)
            or not np.isfinite(path_data.knots_rad).all()
            or np.any(np.diff(path_data.times_s) <= 0)
            or path_data.times_s[0] != 0
            or path_data.times_s[-1] != case.duration_s
        ):
            raise ValueError("Frozen joint trajectory has invalid knots or times")
        regenerated = make_fk_source(model, case).trajectory
        if not np.array_equal(regenerated.times_s, path_data.times_s) or not np.array_equal(
            regenerated.knots_rad, path_data.knots_rad
        ):
            raise ValueError("Frozen realized knots differ from the pinned deterministic generator")
        FrozenTargetSource(entry["samples"])
        samples = serialized_samples(
            make_fk_source(model, case, path_data), case.duration_s, config.control_dt_s
        )
        if entry["samples_sha256"] != content_sha256(entry["samples"]) or entry[
            "samples_sha256"
        ] != content_sha256(samples):
            raise ValueError("Frozen target samples differ from their digest or realized FK source")
        evidence = sampled_kinematic_evidence(model, path_data)
        if evidence != entry["sampled_kinematic_evidence"]:
            raise ValueError("Frozen sampled kinematic evidence changed")
    return manifest


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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--checkpoint", type=Path, help="Existing local RSL-RL checkpoint, inference only"
    )
    selection.add_argument("--controller", choices=("hold", "differential-ik"), default="hold")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Must equal frozen duration, or 0.08..2 seconds for a recorded stream",
    )
    parser.add_argument("--manifest", type=Path, default=FROZEN_MANIFEST_PATH)
    parser.add_argument(
        "--target-stream",
        type=Path,
        help="User-owned already-retargeted JSON targets, separate from the frozen suite",
    )
    parser.add_argument(
        "--freeze-manifest", type=Path, help="Write a new explicit frozen manifest, then exit"
    )
    parser.add_argument("--case", type=int, default=0, choices=range(len(HELDOUT_MANIFEST)))
    args = parser.parse_args()
    model = load_model(args.source_root)
    config = TrackingConfig(episode_steps=20)
    if args.freeze_manifest is not None:
        if args.target_stream is not None or args.checkpoint is not None:
            parser.error(
                "Freeze a procedural manifest separately from recorded streams or checkpoint evaluation"
            )
        frozen = freeze_manifest(model, config, args.source_root)
        with args.freeze_manifest.open("x") as stream:
            stream.write(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "frozen_manifest": str(args.freeze_manifest),
                    "manifest_sha256": frozen["manifest_sha256"],
                }
            )
        )
        return
    if args.output is None:
        parser.error("--output is required for evaluation")

    source: TargetSource
    source_identity: dict[str, Any]
    if args.target_stream is not None:
        duration = 0.4 if args.duration is None else args.duration
        if not 0.08 <= duration <= 2 or not np.isclose(
            duration / config.control_dt_s, round(duration / config.control_dt_s)
        ):
            parser.error(
                "Recorded-stream preparation supports multiples of 0.02 seconds from 0.08 to 2 seconds"
            )
        config = replace(config, episode_steps=round(duration / config.control_dt_s))
        recording = RecordedTargetSource(args.target_stream)
        source = recording
        case_metadata: dict[str, Any] = {"kind": "recorded_target_stream", "duration_s": duration}
        source_identity = {
            "recorded_stream_sha256": recording.sha256,
            "recorded_metadata": recording.metadata,
        }
    else:
        frozen = load_frozen_manifest(args.manifest, model, config, args.source_root)
        frozen_case = frozen["cases"][args.case]
        case = EvaluationCase(**frozen_case["case"])
        if args.duration is not None and not np.isclose(
            args.duration, case.duration_s, rtol=0, atol=1e-12
        ):
            parser.error(
                "Duration differs from frozen case; create a separately reviewed manifest to change targets"
            )
        duration = case.duration_s
        case_metadata = asdict(case)
        source = FrozenTargetSource(frozen_case["samples"])
        source_identity = {
            "frozen_manifest_sha256": frozen["manifest_sha256"],
            "frozen_target_samples_sha256": frozen_case["samples_sha256"],
        }
    env = TrackingEnv(model, ARM_JOINT_NAMES, WRIST_SITE_NAMES, config)

    def zero_action(obs: Array) -> Array:
        return np.zeros(14)

    action_fn: Callable[[Array], Array] = zero_action
    controller_details: dict[str, Any] = {"name": "debug_zero_rate_hold"}
    if args.controller == "differential-ik":
        action_fn = DifferentialIKController(env, gain=2.0, damping=0.05)
        utility = MANIFEST_PATH.parents[1] / "dimos/manipulation/planning/utils/kinematics_utils.py"
        controller_details = {
            "name": "DimOS_math_MuJoCo_differential_IK",
            "gain": 2.0,
            "damping": 0.05,
            "reused_math_sha256": hashlib.sha256(utility.read_bytes()).hexdigest(),
            "production_Pink_controller": False,
        }
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
        controller_details = {"name": "RSL_RL_checkpoint", "updates_performed": 0}
    _, result = run_case(env, source, action_fn, requested_duration_s=duration)
    manifest = {
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "case": case_metadata,
        **source_identity,
        "controller": controller_details["name"],
        "controller_details": controller_details,
        "config": asdict(config),
        "model_sha256": hashlib.sha256(build_xml(args.source_root).encode()).hexdigest(),
        "asset_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "config_sha256": content_sha256(asdict(config)),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "dynamics_validated": False,
        "unsupported_metrics": [
            "overshoot",
            "settling_time",
            "production_Pink_comparison",
            "hardware_latency",
        ],
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
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
