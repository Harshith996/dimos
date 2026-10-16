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

from collections.abc import Iterator
import csv
import json
import math
from pathlib import Path
import time

import pytest

from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
from dimos.hardware.whole_body.spec import MotorCommand, MotorState
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.dual_openyam.blueprints.step_test import dual_openyam_step_test
from dimos.robot.manipulators.dual_openyam.calibration.step_analysis import StepEvent
from dimos.robot.manipulators.dual_openyam.calibration.step_report import (
    REPORT_DIR_NAME,
    generate_report,
)
from dimos.robot.manipulators.dual_openyam.calibration.step_test import (
    JointStepTestModule,
    StepRefusedError,
    check_step,
)
from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_HOME_JOINTS
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_JOINTS,
)

HOME = dict(zip(DUAL_OPENYAM_ARM_JOINTS, DUAL_OPENYAM_HOME_JOINTS, strict=True))
LOOP_HZ = 200.0


class LaggingMockAdapter(MockWholeBodyAdapter):
    """Mock whose joints follow the command as a first-order lag, like a real gearbox."""

    def __init__(self, *, dof: int, initial_positions: list[float], tau_s: float = 0.05) -> None:
        super().__init__(dof=dof, initial_positions=initial_positions)
        self._tau = tau_s
        self._last = time.monotonic()

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        if not self._connected or len(commands) != len(self._states):
            return False
        now = time.monotonic()
        dt = max(now - self._last, 1e-4)
        self._last = now
        alpha = 1.0 - math.exp(-dt / self._tau)
        new_states = []
        for state, command in zip(self._states, commands, strict=True):
            if command.kp > 0.0:
                q = state.q + alpha * (command.q - state.q)
                new_states.append(MotorState(q=q, dq=(q - state.q) / dt, tau=command.tau))
            else:
                new_states.append(MotorState(q=command.q, dq=0.0, tau=0.0))
        self._states = new_states
        return True


def _adapter(offset_rad: float = 0.0) -> MockWholeBodyAdapter:
    positions = [q + offset_rad for q in DUAL_OPENYAM_HOME_JOINTS] + [0.3, 0.3]
    return MockWholeBodyAdapter(dof=len(DUAL_OPENYAM_JOINTS), initial_positions=positions)


@pytest.fixture
def module() -> Iterator[JointStepTestModule]:
    module = JointStepTestModule(adapter=_adapter(), loop_hz=LOOP_HZ)
    try:
        module.start()
        yield module
    finally:
        module.stop()


# Refusal logic


def test_check_step_accepts_a_small_step_near_the_test_pose() -> None:
    check_step("left_joint3", 1.0, dict(HOME), dict(HOME))


def test_check_step_refuses_steps_over_five_degrees() -> None:
    with pytest.raises(StepRefusedError, match="5 deg limit"):
        check_step("left_joint3", 5.01, dict(HOME), dict(HOME))
    check_step("left_joint3", -5.0, dict(HOME), dict(HOME))


@pytest.mark.parametrize("delta", [0.0, math.nan, math.inf])
def test_check_step_refuses_degenerate_steps(delta: float) -> None:
    with pytest.raises(StepRefusedError):
        check_step("left_joint3", delta, dict(HOME), dict(HOME))


def test_set_gains_refuses_what_the_mit_frame_cannot_carry(module: JointStepTestModule) -> None:
    # The Damiao MIT frame packs kp into 0..500 and kd into 0..5; on the rig a
    # kd of 6.1 made the codec refuse every command until the gains were reverted.
    with pytest.raises(StepRefusedError, match="kd 0..5"):
        module.set_gains("left_joint1", 120.0, 6.1)
    with pytest.raises(StepRefusedError, match="kp 0..500"):
        module.set_gains("left_joint1", 600.0, 5.0)
    module.set_gains("left_joint1", 160.0, 5.0)


def test_check_step_refuses_unknown_joints() -> None:
    with pytest.raises(StepRefusedError, match="unknown arm joint"):
        check_step("left_arm/gripper", 1.0, dict(HOME), dict(HOME))


def test_check_step_refuses_when_any_joint_is_far_from_its_target() -> None:
    measured = dict(HOME)
    measured["right_joint5"] += math.radians(10.5)
    with pytest.raises(StepRefusedError, match=r"right_joint5 \+10.5 deg"):
        check_step("left_joint1", 1.0, measured, dict(HOME))
    measured["right_joint5"] = HOME["right_joint5"] + math.radians(9.5)
    check_step("left_joint1", 1.0, measured, dict(HOME))
    # The guard is relative to the target, not to a fixed pose: any ramped-to pose is fine.
    elsewhere = {name: q + math.radians(40.0) for name, q in HOME.items()}
    check_step("left_joint3", 1.0, elsewhere, dict(elsewhere))


# Module behaviour on the mock adapter


def test_park_folds_every_arm_joint_onto_the_supports(module: JointStepTestModule) -> None:
    message = module.park(speed_deg_s=30.0)
    assert "parked" in message
    for info in module.status()["joints"].values():
        assert info["target_deg"] == pytest.approx(0.0, abs=1e-6)


def test_stop_parks_toward_rest_before_disabling() -> None:
    # Started at home (joints 2 and 3 at 60 deg): stop() must ramp toward the
    # rest pose inside its budget before the motors go off.
    module = JointStepTestModule(adapter=_adapter(), loop_hz=LOOP_HZ)
    commands: list[JointState] = []
    module.joint_command.subscribe(commands.append)
    module.start()
    time.sleep(0.1)
    module.stop()
    last = commands[-1].position[: len(DUAL_OPENYAM_HOME_JOINTS)]
    assert max(abs(v) for v in last) < math.radians(1.0)


def test_start_holds_the_measured_pose_and_streams_state(module: JointStepTestModule) -> None:
    states: list[JointState] = []
    commands: list[JointState] = []
    module.joint_state.subscribe(states.append)
    module.joint_command.subscribe(commands.append)
    time.sleep(0.2)
    status = module.status()
    assert status["running"] and status["adapter"] == "MockWholeBodyAdapter"
    assert len(states) >= 10 and len(commands) >= 10
    assert states[-1].name == DUAL_OPENYAM_JOINTS
    for name, info in status["joints"].items():
        assert info["target_deg"] == pytest.approx(math.degrees(HOME[name]))
        assert info["pose_error_deg"] == pytest.approx(0.0, abs=1e-6)
    # Grippers are held at their measured opening with zero gain.
    assert commands[-1].position[-2:] == [0.3, 0.3]


def test_step_moves_the_target_and_publishes_a_marker(module: JointStepTestModule) -> None:
    events: list[String] = []
    module.step_events.subscribe(events.append)
    result = module.step("left_joint2", 1.0, hold_s=0.2)
    assert result.startswith("left_joint2 +1.00 deg at kp=80 kd=5")
    [event] = [StepEvent.from_json(e.data) for e in events]
    assert event.joint == "left_joint2" and event.delta_deg == 1.0 and event.hold_s == 0.2
    assert event.target_after_deg - event.target_before_deg == pytest.approx(1.0)
    assert module.status()["joints"]["left_joint2"]["target_deg"] == pytest.approx(
        math.degrees(HOME["left_joint2"]) + 1.0
    )


def test_step_refuses_large_steps_and_bad_joints(module: JointStepTestModule) -> None:
    with pytest.raises(StepRefusedError, match="exceeds the 5 deg limit"):
        module.step("left_joint1", 6.0, hold_s=0.1)
    with pytest.raises(StepRefusedError, match="unknown arm joint"):
        module.step("left_arm/gripper", 1.0, hold_s=0.1)
    with pytest.raises(StepRefusedError, match="hold_s"):
        module.step("left_joint1", 1.0, hold_s=0.0)


def test_steps_are_allowed_at_any_pose_a_ramp_reached() -> None:
    # Starting 12 deg off the test pose is fine: the target starts at the measured pose.
    module = JointStepTestModule(adapter=_adapter(offset_rad=math.radians(12.0)), loop_hz=LOOP_HZ)
    try:
        module.start()
        module.step("left_joint1", 1.0, hold_s=0.1)
        assert module.go_to_test_pose(speed_deg_s=30.0).startswith("moved 12 joint(s) 13.0 deg")
        assert not module.status()["ramping"]
        module.step("left_joint1", 1.0, hold_s=0.1)
        elsewhere = {"left_joint2": HOME["left_joint2"] + math.radians(20.0)}
        module.go_to(elsewhere, speed_deg_s=30.0)
        assert module.step("left_joint2", -1.0, hold_s=0.1).startswith("left_joint2 -1.00 deg")
        assert module.status()["joints"]["left_joint2"]["target_error_deg"] == pytest.approx(
            0.0, abs=1e-6
        )
    finally:
        module.stop()


class StuckMockAdapter(MockWholeBodyAdapter):
    """Mock whose first joint never moves, like a stalled or blocked motor."""

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        stuck = self._states[0]
        ok = super().write_motor_commands(commands)
        self._states[0] = stuck
        return ok


def test_step_is_refused_when_a_joint_does_not_track_its_target() -> None:
    adapter = StuckMockAdapter(
        dof=len(DUAL_OPENYAM_JOINTS), initial_positions=[*DUAL_OPENYAM_HOME_JOINTS, 0.0, 0.0]
    )
    module = JointStepTestModule(adapter=adapter, loop_hz=LOOP_HZ)
    try:
        module.start()
        module.go_to({"left_joint1": HOME["left_joint1"] + math.radians(11.0)}, speed_deg_s=30.0)
        with pytest.raises(StepRefusedError, match=r"left_joint1 -11.0 deg"):
            module.step("left_joint2", 1.0, hold_s=0.1)
    finally:
        module.stop()


def test_mark_publishes_a_json_marker_with_the_targets(module: JointStepTestModule) -> None:
    events: list[String] = []
    module.gravity_events.subscribe(events.append)
    assert module.mark("pose_01", {"index": 1}) == "marked pose_01"
    with pytest.raises(StepRefusedError):
        module.mark("")
    [event] = events
    data = json.loads(event.data)
    assert data["label"] == "pose_01" and data["data"] == {"index": 1}
    assert data["targets_deg"]["left_joint2"] == pytest.approx(math.degrees(HOME["left_joint2"]))
    assert data["t0"] == pytest.approx(time.time(), abs=5.0)


def test_set_gains_takes_effect_in_the_gains_stream(module: JointStepTestModule) -> None:
    gains: list[JointState] = []
    module.joint_gains.subscribe(gains.append)
    assert module.set_gains("right_joint4", 15.0, 1.8) == "right_joint4: kp=15 kd=1.8"
    time.sleep(0.05)
    index = DUAL_OPENYAM_ARM_JOINTS.index("right_joint4")
    assert gains[-1].position[index] == 15.0 and gains[-1].velocity[index] == 1.8
    assert module.get_gains()["right_joint4"] == (15.0, 1.8)
    with pytest.raises(StepRefusedError):
        module.set_gains("right_joint4", -1.0, 1.0)


def test_go_to_rejects_unknown_joints_and_excess_speed(module: JointStepTestModule) -> None:
    with pytest.raises(StepRefusedError):
        module.go_to({"left_arm/gripper": 0.5}, 10.0)
    with pytest.raises(StepRefusedError):
        module.go_to({"left_joint1": 0.1}, 100.0)


def test_blueprint_pins_recording_and_exposes_the_streams() -> None:
    assert dual_openyam_step_test.global_config_overrides["record"] == "sqlite"
    [atom] = dual_openyam_step_test.blueprints
    assert atom.module is JointStepTestModule
    assert {s.name for s in atom.streams} == {
        "joint_state",
        "joint_command",
        "joint_gains",
        "step_events",
        "gravity_events",
        "float_events",
    }


# End to end on mock hardware: stream -> memory.db -> report


def test_two_steps_record_and_report(tmp_path: Path) -> None:
    recording_dir = tmp_path / "run"
    store = SqliteStore(path=str(recording_dir / "memory.db"))
    store.start()
    adapter = LaggingMockAdapter(
        dof=len(DUAL_OPENYAM_JOINTS),
        initial_positions=[*DUAL_OPENYAM_HOME_JOINTS, 0.0, 0.0],
    )
    module = JointStepTestModule(adapter=adapter, loop_hz=LOOP_HZ)
    try:
        for name, out, kind in (
            ("joint_state", module.joint_state, JointState),
            ("joint_command", module.joint_command, JointState),
            ("joint_gains", module.joint_gains, JointState),
            ("step_events", module.step_events, String),
        ):
            stream = store.stream(name, kind)
            out.subscribe(lambda msg, s=stream: s.append(msg, ts=getattr(msg, "ts", None)))
        module.start()
        time.sleep(0.1)
        module.step("left_joint1", 1.0, hold_s=0.6)
        module.set_gains("left_joint1", 160.0, 5.0)
        module.step("left_joint1", -1.0, hold_s=0.6)
    finally:
        module.stop()
        store.stop()

    reopened = SqliteStore(path=str(recording_dir / "memory.db"), must_exist=True)
    reopened.start()
    try:
        assert set(reopened.list_streams()) >= {
            "joint_state",
            "joint_command",
            "joint_gains",
            "step_events",
        }
        assert reopened.stream("step_events").count() == 2
    finally:
        reopened.stop()

    out_dir = generate_report(recording_dir)
    assert out_dir == recording_dir / REPORT_DIR_NAME
    with (out_dir / "steps.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["joint"], float(r["delta_deg"]), float(r["kp"])) for r in rows] == [
        ("left_joint1", 1.0, 80.0),
        ("left_joint1", -1.0, 160.0),
    ]
    for row in rows:
        # 50 ms lag: settles within 0.1 deg in ~0.12 s, no overshoot, quiet at rest.
        assert float(row["settle_time_s"]) < 0.3
        assert float(row["overshoot_pct"]) == 0.0
        assert row["passed"] == "True"
    assert (out_dir / "report.md").read_text().count("| left_joint1 |") == 2
    assert (out_dir / "left_joint1.png").stat().st_size > 0
    assert (out_dir / "summary_settle_time.png").exists()
    assert (out_dir / "summary_overshoot.png").exists()
