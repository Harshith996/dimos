# Copyright 2026 Dimensional Inc.
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

from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import pinocchio as pin
import pytest

from dimos.hardware.whole_body.damiao.adapter import apply_inertial_overrides
from dimos.hardware.whole_body.damiao.config import LinkInertial, load_inertial_overrides
from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
from dimos.hardware.whole_body.spec import MotorCommand, MotorState
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.dual_openyam.calibration.gravity_fit import (
    COM_BOX_M,
    REPORT_DIR_NAME,
    Measurement,
    PoseMarker,
    default_holdout,
    extract_measurements,
    find_previous_json,
    fit_gravity,
    friction_flags,
    generate_gravity_report,
    load_gravity_recording,
    merge_inertial_json,
    pose_windows,
)
from dimos.robot.manipulators.dual_openyam.calibration.gravity_poses import (
    MEASURED_JOINTS,
    ArmModel,
    Pose,
    PoseSet,
    generate_poses,
    home_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.step_test import JointStepTestModule
from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_HOME_JOINTS
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_JOINTS,
)

OLD_RECORDING = (
    Path(__file__).resolve().parents[5]
    / "recordings"
    / "20261015-235349-091c-dual-openyam-step-test"
)
# A camera and bracket on the gripper, a heavier forearm: what the rig likely has.
PERTURBATION = {
    "left_gripper": (0.25, np.array([-0.02, 0.0, -0.01])),  # d mass (kg), d(m*c) (kg m)
    "left_link3": (0.15, np.array([0.0, 0.0, -0.02])),
}


@pytest.fixture(scope="module")
def arm() -> ArmModel:
    return ArmModel()


@pytest.fixture(scope="module")
def pose_set(arm: ArmModel) -> PoseSet:
    return generate_poses(arm)


@pytest.fixture(scope="module")
def right_arm() -> ArmModel:
    return ArmModel(side="right")


@pytest.fixture(scope="module")
def right_pose_set(right_arm: ArmModel) -> PoseSet:
    return generate_poses(right_arm)


def perturbed_model(arm: ArmModel, perturbation: dict[str, tuple[float, np.ndarray]]) -> pin.Model:
    model = arm.model.copy()
    for link, (d_mass, d_moment) in perturbation.items():
        joint_id = model.getFrameId(link)
        joint_id = int(model.frames[joint_id].parentJoint)
        inertia = model.inertias[joint_id]
        mass = inertia.mass + d_mass
        com = (inertia.mass * inertia.lever + d_moment) / mass
        model.inertias[joint_id] = pin.Inertia(mass, com, inertia.inertia)
    return model


def synthetic_measurements(
    arm: ArmModel,
    truth: pin.Model,
    poses: list[Pose],
    *,
    noise_nm: float,
    friction_nm: float,
    seed: int = 0,
) -> list[Measurement]:
    """tau_pd = g_true(q) - g_model(q) + noise, as the +/- dither average would give."""
    rng = np.random.default_rng(seed)
    data = truth.createData()
    out = []
    for pose in poses:
        q = arm.full_q(pose.joints_rad)
        error = np.asarray(pin.computeGeneralizedGravity(truth, data, q)) - arm.gravity(q)
        for joint, row in zip(arm.measured_joints, arm.measured_rows, strict=True):
            out.append(
                Measurement(
                    pose=pose.name,
                    joint=joint,
                    kp=80.0,
                    kd=5.0,
                    dither_deg=1.0,
                    q_rad=tuple(float(v) for v in q),
                    tau_pd_nm=float(error[row] + rng.normal(0.0, noise_nm)),
                    friction_nm=float(friction_nm * (1.0 + 0.1 * rng.normal())),
                )
            )
    return out


def corrected_gravity(arm: ArmModel, links: dict[str, LinkInertial], q: np.ndarray) -> np.ndarray:
    model = arm.model.copy()
    apply_inertial_overrides(model, links)
    return np.asarray(pin.computeGeneralizedGravity(model, model.createData(), q))


# Fit on synthetic data


def test_fit_recovers_an_injected_perturbation_on_held_out_poses(
    arm: ArmModel, pose_set: PoseSet
) -> None:
    truth = perturbed_model(arm, PERTURBATION)
    poses = [home_pose(), *pose_set.identification]
    measurements = synthetic_measurements(arm, truth, poses, noise_nm=0.02, friction_nm=0.4)
    result = fit_gravity(arm, measurements)

    assert result.holdout_poses == ["pose_04", "pose_09", "pose_14"]
    assert len(result.train_poses) == 13
    assert result.bounds_ok
    before = result.rms_by_joint(holdout=True, after=False)
    after = result.rms_by_joint(holdout=True, after=True)
    assert set(before) == set(MEASURED_JOINTS)
    assert max(before["left_joint2"], before["left_joint3"]) > 0.5
    for joint in MEASURED_JOINTS:
        assert after[joint] < 0.06, (joint, after[joint])
        assert after[joint] < 0.5 * before[joint] or before[joint] < 0.1
    # The corrected model predicts the true gravity torque at poses it never saw.
    rows = list(arm.measured_rows)
    truth_data = truth.createData()
    for pose in pose_set.identification:
        if pose.name not in result.holdout_poses:
            continue
        q = arm.full_q(pose.joints_rad)
        expected = np.asarray(pin.computeGeneralizedGravity(truth, truth_data, q))[rows]
        got = corrected_gravity(arm, result.links, q)[rows]
        assert np.abs(got - expected).max() < 0.08
    assert result.rank >= 6 and result.condition_number < 50.0
    determined = [p for p in result.parameters if p.determined]
    assert determined and all(p.std_err > 0.0 for p in determined)
    assert all(p.observable <= 1.0 + 1e-9 for p in result.parameters)


def test_right_arm_fit_recovers_a_right_side_perturbation(
    right_arm: ArmModel, right_pose_set: PoseSet
) -> None:
    truth = perturbed_model(
        right_arm,
        {
            "right_gripper": (0.25, np.array([-0.02, 0.0, -0.01])),
            "right_link3": (0.15, np.array([0.0, 0.0, -0.02])),
        },
    )
    poses = [home_pose("right"), *right_pose_set.identification]
    measurements = synthetic_measurements(right_arm, truth, poses, noise_nm=0.02, friction_nm=0.4)
    assert {m.joint for m in measurements} == {f"right_joint{i}" for i in (2, 3, 4, 5)}
    result = fit_gravity(right_arm, measurements)
    assert result.side == "right" and result.bounds_ok
    assert set(result.links) == {
        "right_link2",
        "right_link3",
        "right_link4",
        "right_link5",
        "right_gripper",
    }
    before = result.rms_by_joint(holdout=True, after=False)
    after = result.rms_by_joint(holdout=True, after=True)
    assert set(after) == {f"right_joint{i}" for i in (2, 3, 4, 5)}
    assert max(before.values()) > 0.5
    assert max(after.values()) < 0.06
    # The left arm, held at home, is untouched by the correction.
    q = right_arm.full_q(right_pose_set.identification[0].joints_rad)
    assert np.allclose(corrected_gravity(right_arm, result.links, q)[:6], right_arm.gravity(q)[:6])
    # Left-arm measurements do not fit a right-arm model.
    left_measurement = Measurement("x", "left_joint2", 80.0, 5.0, 1.0, tuple(q), 0.0, 0.3)
    with pytest.raises(ValueError, match="right measured joints"):
        fit_gravity(right_arm, [left_measurement])


def test_merge_joins_both_arms_into_one_override_file(tmp_path: Path) -> None:
    left = tmp_path / "gravity_left_2026-10-16.json"
    right = tmp_path / "gravity_right_2026-10-17.json"
    left.write_text(
        json.dumps(
            {
                "created": "2026-10-16T10:00:00",
                "side": "left",
                "links": {
                    "left_gripper": {"mass_kg": 0.9, "com_m": [0.0, 0.0, -0.04]},
                    "right_link3": {"mass_kg": 9.0, "com_m": [0.0, 0.0, 0.0]},  # stale, overridden
                },
            }
        )
    )
    right.write_text(
        json.dumps(
            {
                "created": "2026-10-17T10:00:00",
                "side": "right",
                "links": {"right_link3": {"mass_kg": 1.1, "com_m": [-0.05, -0.03, -0.12]}},
            }
        )
    )
    merged = merge_inertial_json([left, right], datetime(2026, 10, 17, 12, 0))
    assert merged["side"] == "both" and merged["created"] == "2026-10-17T12:00:00"
    assert merged["links"] == {
        "left_gripper": {"mass_kg": 0.9, "com_m": [0.0, 0.0, -0.04]},
        "right_link3": {"mass_kg": 1.1, "com_m": [-0.05, -0.03, -0.12]},
    }
    assert [s["side"] for s in merged["merged_from"]] == ["left", "right"]
    out = tmp_path / "gravity_both.json"
    out.write_text(json.dumps(merged))
    assert set(load_inertial_overrides(out)) == {"left_gripper", "right_link3"}


def test_fit_stays_physical_and_flags_the_bound_when_the_data_want_more(
    arm: ArmModel, pose_set: PoseSet
) -> None:
    # A 2 kg "camera" far out on the gripper: the bounded fit cannot follow it, so
    # the solution sits on a bound, says so, and the error it cannot explain remains.
    truth = perturbed_model(arm, {"left_gripper": (2.0, np.array([0.0, 0.0, -0.4]))})
    measurements = synthetic_measurements(
        arm, truth, [home_pose(), *pose_set.identification], noise_nm=0.0, friction_nm=0.3
    )
    result = fit_gravity(arm, measurements, ridge_nm=0.05)
    assert result.bounds_ok
    assert any("left_gripper" in v and "bound" in v for v in result.at_bounds)
    for link, fitted in result.links.items():
        urdf = result.urdf_links[link]
        assert np.abs(np.asarray(fitted.com_m) - np.asarray(urdf.com_m)).max() <= COM_BOX_M + 1e-9
        assert 0.5 * urdf.mass_kg - 1e-9 <= fitted.mass_kg <= 2.0 * urdf.mass_kg + 1e-9
    after = result.rms_by_joint(holdout=True, after=True)
    assert max(after.values()) > 0.5


def test_fit_rejects_unknown_holdout_and_everything_held_out(
    arm: ArmModel, pose_set: PoseSet
) -> None:
    measurements = synthetic_measurements(
        arm, arm.model, [home_pose(), *pose_set.identification[:2]], noise_nm=0.0, friction_nm=0.3
    )
    with pytest.raises(ValueError, match="not in the recording"):
        fit_gravity(arm, measurements, holdout=["pose_99"])
    with pytest.raises(ValueError, match="every pose is held out"):
        fit_gravity(arm, measurements, holdout=["home", "pose_01", "pose_02"])
    with pytest.raises(ValueError, match="nothing to fit"):
        fit_gravity(arm, [])


def test_default_holdout_is_every_fifth_pose_when_there_are_enough() -> None:
    poses = [f"pose_{i:02d}" for i in range(1, 16)]
    assert default_holdout(poses) == ["pose_05", "pose_10", "pose_15"]
    assert default_holdout(poses[:5]) == []
    assert default_holdout(poses, every=0) == []


def test_friction_flags_mark_outliers_and_negative_values(arm: ArmModel) -> None:
    q = tuple(arm.full_q(home_pose().joints_rad))

    def m(pose: str, friction: float, kp: float = 400.0) -> Measurement:
        return Measurement(pose, "left_joint3", kp, 5.0, 1.0, q, 0.0, friction)

    measurements = [m("a", 0.5), m("b", 0.55), m("c", 0.45), m("d", 1.6), m("e", -0.1)]
    flags = friction_flags(measurements)
    assert set(flags) == {3, 4}
    assert "median" in flags[3] and "negative" in flags[4]
    # Friction of exactly kp*d/2 means the joint never moved for the dither pair.
    stuck = friction_flags([*measurements, m("f", 80.0 * math.radians(1.0) / 2.0, kp=80.0)])
    assert "did not move" in stuck[5]


# Recording pipeline: markers + steps -> memory.db -> report


class GravityErrorMockAdapter(MockWholeBodyAdapter):
    """Mock arm that settles where the PD torque balances a gravity-model error plus friction.

    The real adapter adds g_model(q); the true arm needs g_true(q). At rest
    kp (q_cmd - q) = g_true - g_model +/- friction, the sign following the
    direction the joint last moved in.
    """

    def __init__(
        self,
        arm: ArmModel,
        truth: pin.Model,
        *,
        dof: int,
        initial_positions: list[float],
        friction_nm: float,
    ) -> None:
        super().__init__(dof=dof, initial_positions=initial_positions)
        self._arm = arm
        self._truth = truth
        self._truth_data = truth.createData()
        self._friction = friction_nm
        self._last_cmd = list(initial_positions)
        self._direction = [0.0] * len(initial_positions)

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        if not self._connected or len(commands) != len(self._states):
            return False
        n = len(DUAL_OPENYAM_ARM_JOINTS)
        q_cmd = np.asarray([c.q for c in commands[:n]])
        error = np.asarray(
            pin.computeGeneralizedGravity(self._truth, self._truth_data, q_cmd)
        ) - self._arm.gravity(q_cmd)
        new_states = []
        for i, command in enumerate(commands):
            if i >= n or command.kp <= 0.0:
                new_states.append(MotorState(q=command.q, dq=0.0, tau=0.0))
                continue
            moved = command.q - self._last_cmd[i]
            if abs(moved) > 1e-9:
                # Friction keeps opposing the last direction the joint moved in.
                self._direction[i] = math.copysign(1.0, moved)
                self._last_cmd[i] = command.q
            # Positive direction: stops short below the target -> q_cmd - q > 0 extra.
            q = command.q - (error[i] + self._direction[i] * self._friction) / command.kp
            new_states.append(MotorState(q=q, dq=0.0, tau=command.tau))
        self._states = new_states
        return True


def _record(module: JointStepTestModule, store: SqliteStore) -> None:
    for name, out, kind in (
        ("joint_state", module.joint_state, JointState),
        ("joint_command", module.joint_command, JointState),
        ("joint_gains", module.joint_gains, JointState),
        ("step_events", module.step_events, String),
        ("gravity_events", module.gravity_events, String),
    ):
        stream = store.stream(name, kind)
        out.subscribe(lambda msg, s=stream: s.append(msg, ts=getattr(msg, "ts", None)))


@pytest.mark.parametrize("side", ["left", "right"])
def test_sweep_records_markers_and_the_fit_reports_held_out_error(
    tmp_path: Path, arm: ArmModel, right_arm: ArmModel, side: str
) -> None:
    arm = arm if side == "left" else right_arm
    truth = perturbed_model(arm, {f"{side}_gripper": (0.2, np.array([0.0, 0.0, -0.01]))})
    friction = 0.3
    recording_dir = tmp_path / "run"
    store = SqliteStore(path=str(recording_dir / "memory.db"))
    store.start()
    adapter = GravityErrorMockAdapter(
        arm,
        truth,
        dof=len(DUAL_OPENYAM_JOINTS),
        initial_positions=[*DUAL_OPENYAM_HOME_JOINTS, 0.0, 0.0],
        friction_nm=friction,
    )
    module = JointStepTestModule(adapter=adapter, loop_hz=200.0)
    home = home_pose(side)
    # Small moves around home keep the test short (ramps run at 30 deg/s).
    poses = [
        Pose("pose_a", tuple(np.asarray(home.joints_rad) + np.radians([0, 8, 0, 0, 0, 0])), side),
        Pose("pose_b", tuple(np.asarray(home.joints_rad) + np.radians([0, 0, 8, 0, 0, 0])), side),
        Pose("pose_c", tuple(np.asarray(home.joints_rad) + np.radians([0, -8, 8, 6, 0, 0])), side),
    ]
    stepped = (f"{side}_joint2", f"{side}_joint3")
    other = "right_joint2" if side == "left" else "left_joint2"
    try:
        _record(module, store)
        module.start()
        time.sleep(0.1)
        for index, pose in enumerate(poses, 1):
            module.mark(pose.name, {"index": index})
            module.go_to(pose.as_targets(), 30.0)
            time.sleep(0.1)
            for joint in stepped:
                module.step(joint, 1.0, hold_s=0.25)
                module.step(joint, -1.0, hold_s=0.25)
        module.mark("end")
    finally:
        module.stop()
        store.stop()

    grec = load_gravity_recording(recording_dir)
    assert [m.label for m in grec.markers] == ["pose_a", "pose_b", "pose_c", "end"]
    assert grec.markers[0].targets_deg[stepped[0]] == pytest.approx(
        math.degrees(DUAL_OPENYAM_HOME_JOINTS[1]), abs=1e-6
    )
    # The other arm never left home during the sweep (stop() parks both arms afterwards).
    other_index = DUAL_OPENYAM_ARM_JOINTS.index(other)
    trace = grec.recording.traces[other]
    sweep = trace.t < grec.markers[-1].t0
    assert (
        np.abs(trace.q_deg[sweep] - math.degrees(DUAL_OPENYAM_HOME_JOINTS[other_index])).max() < 0.5
    )
    measurements = extract_measurements(grec, side)
    assert not extract_measurements(grec, "right" if side == "left" else "left")
    assert [(m.pose, m.joint) for m in measurements] == [
        (p.name, j) for p in poses for j in stepped
    ]
    truth_data = truth.createData()
    for m in measurements:
        q = np.asarray(m.q_rad)
        row = DUAL_OPENYAM_ARM_JOINTS.index(m.joint)
        expected = pin.computeGeneralizedGravity(truth, truth_data, q)[row] - arm.gravity(q)[row]
        # Friction cancels in the average and shows up as half the difference.
        assert m.tau_pd_nm == pytest.approx(expected, abs=0.02)
        assert m.friction_nm == pytest.approx(friction, abs=0.02)

    other_json = tmp_path / "other.json"
    other_json.write_text(
        json.dumps({"links": {f"{other[:-7]}_gripper": {"mass_kg": 0.8, "com_m": [0, 0, -0.03]}}})
    )
    result, out_dir = generate_gravity_report(
        recording_dir,
        holdout=["pose_c"],
        arm=arm,
        now=datetime(2026, 10, 16, 12, 0),
        side=side,
        merge=other_json,
    )
    assert out_dir == recording_dir / REPORT_DIR_NAME
    assert result.side == side
    assert result.holdout_poses == ["pose_c"] and result.train_poses == ["pose_a", "pose_b"]
    report = (out_dir / "report.md").read_text()
    assert report.index("## Held-out error (1 pose(s): pose_c)") < report.index("## Fit residual")
    assert "held out" in report and "condition number" in report
    assert (out_dir / "resting_torque.png").stat().st_size > 0
    assert (out_dir / "measurements.csv").read_text().count("pose_") == 6
    json_path = out_dir / f"gravity_{side}_2026-10-16.json"
    both_path = out_dir / "gravity_both_2026-10-16.json"
    if result.bounds_ok:
        data = json.loads(json_path.read_text())
        assert data["side"] == side and set(data["links"]) == set(result.links)
        assert data["fit"]["holdout_poses"] == ["pose_c"]
        assert load_inertial_overrides(json_path)[f"{side}_gripper"].mass_kg > 0.0
        assert "FAILED" not in report
        both = load_inertial_overrides(both_path)
        assert set(both) == set(result.links) | {f"{other[:-7]}_gripper"}
    else:
        assert not json_path.exists() and not both_path.exists()
        assert "FAILED physical bounds" in report


def test_report_compares_with_a_previous_json(
    tmp_path: Path, arm: ArmModel, pose_set: PoseSet
) -> None:
    previous = tmp_path / "gravity_2026-10-01.json"
    previous.write_text(
        json.dumps({"links": {"left_gripper": {"mass_kg": 0.9, "com_m": [0.0, 0.0, -0.05]}}})
    )
    assert find_previous_json([tmp_path]) == previous
    assert find_previous_json([tmp_path], exclude=previous) is None
    assert find_previous_json([tmp_path / "missing"]) is None
    truth = perturbed_model(arm, PERTURBATION)
    measurements = synthetic_measurements(
        arm, truth, [home_pose(), *pose_set.identification], noise_nm=0.0, friction_nm=0.3
    )
    result = fit_gravity(arm, measurements)
    from dimos.robot.manipulators.dual_openyam.calibration.gravity_fit import (
        format_report,
    )

    text = format_report(result, previous, None)
    assert f"## Compared with {previous}" in text
    assert "| left_gripper | 0.900 |" in text


def test_pose_windows_without_markers_cover_everything() -> None:
    assert pose_windows([], 10.0) == [("all", -math.inf, math.inf)]
    markers = [PoseMarker("a", 1.0, {}, {}), PoseMarker("b", 5.0, {}, {})]
    assert pose_windows(markers, 9.0) == [("a", 1.0, 5.0), ("b", 5.0, 9.0)]


@pytest.mark.skipif(not OLD_RECORDING.exists(), reason="rig recording not present")
def test_fit_runs_on_the_single_pose_step_test_recording(arm: ArmModel, tmp_path: Path) -> None:
    # Rank deficient by construction (one pose): it must still run and say so.
    grec = load_gravity_recording(OLD_RECORDING)
    assert not grec.markers
    measurements = extract_measurements(grec)
    assert {m.joint for m in measurements} == set(MEASURED_JOINTS)
    result = fit_gravity(arm, measurements)
    assert result.train_poses == ["all"] and not result.holdout_poses
    assert result.rank < len(result.parameters)
    assert sum(p.determined for p in result.parameters) <= result.rank
