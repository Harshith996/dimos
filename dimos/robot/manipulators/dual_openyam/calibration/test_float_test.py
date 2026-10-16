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
import threading
import time

import numpy as np
import pytest

from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
from dimos.hardware.whole_body.spec import MotorCommand, MotorState
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.dual_openyam.calibration.float_analysis import (
    DRIFT_LIMIT_MM,
    FingertipModel,
    FloatEvent,
    analyze_window,
    segment_windows,
)
from dimos.robot.manipulators.dual_openyam.calibration.float_report import (
    REPORT_DIR_NAME,
    generate_report,
)
from dimos.robot.manipulators.dual_openyam.calibration.step_test import (
    DEFAULT_FLOAT_KD,
    JointStepTestModule,
    StepRefusedError,
)
from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_HOME_JOINTS
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_JOINTS,
    DUAL_OPENYAM_LEFT_ARM_JOINTS,
    DUAL_OPENYAM_RIGHT_ARM_JOINTS,
)

HOME = dict(zip(DUAL_OPENYAM_ARM_JOINTS, DUAL_OPENYAM_HOME_JOINTS, strict=True))
LOOP_HZ = 200.0
LEFT = [DUAL_OPENYAM_ARM_JOINTS.index(n) for n in DUAL_OPENYAM_LEFT_ARM_JOINTS]
RIGHT = [DUAL_OPENYAM_ARM_JOINTS.index(n) for n in DUAL_OPENYAM_RIGHT_ARM_JOINTS]


class DriftingMockAdapter(MockWholeBodyAdapter):
    """Joints follow a kp>0 command at once; with kp=0 they drift at ``drift_rad_s``.

    Models a floated arm sagging under an imperfect gravity model: the drift is
    per joint index, zero where unset.
    """

    def __init__(
        self,
        *,
        dof: int,
        initial_positions: list[float],
        drift_rad_s: dict[int, float] | None = None,
    ) -> None:
        super().__init__(dof=dof, initial_positions=initial_positions)
        self.drift_rad_s = dict(drift_rad_s or {})
        self._last = time.monotonic()
        self.commands: list[list[MotorCommand]] = []

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        if not self._connected or len(commands) != len(self._states):
            return False
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        self.commands.append(commands)
        new_states = []
        for i, (state, command) in enumerate(zip(self._states, commands, strict=True)):
            if command.kp > 0.0:
                new_states.append(MotorState(q=command.q, dq=0.0, tau=0.0))
            else:
                rate = self.drift_rad_s.get(i, 0.0)
                new_states.append(MotorState(q=state.q + rate * dt, dq=rate, tau=0.0))
        self._states = new_states
        return True


def _adapter(drift_rad_s: dict[int, float] | None = None) -> DriftingMockAdapter:
    return DriftingMockAdapter(
        dof=len(DUAL_OPENYAM_JOINTS),
        initial_positions=[*DUAL_OPENYAM_HOME_JOINTS, 0.3, 0.3],
        drift_rad_s=drift_rad_s,
    )


@pytest.fixture
def module() -> Iterator[JointStepTestModule]:
    # left_joint2 sags at 20 deg/s whenever its kp is zero; nothing else moves.
    module = JointStepTestModule(adapter=_adapter({1: math.radians(20.0)}), loop_hz=LOOP_HZ)
    try:
        module.start()
        yield module
    finally:
        module.stop()


@pytest.fixture(scope="module")
def left_model() -> FingertipModel:
    return FingertipModel("left")


# float_arm / hold_arm


def test_float_arm_zeroes_kp_on_that_arm_only(module: JointStepTestModule) -> None:
    gains: list[JointState] = []
    commands: list[JointState] = []
    module.joint_gains.subscribe(gains.append)
    module.joint_command.subscribe(commands.append)
    before = module.get_gains()
    assert module.float_arm("left") == "left arm floating: kp=0 kd=0.5 on 6 joints"
    time.sleep(0.05)
    after = module.get_gains()
    for name in DUAL_OPENYAM_LEFT_ARM_JOINTS:
        assert after[name] == (0.0, DEFAULT_FLOAT_KD)
    for name in DUAL_OPENYAM_RIGHT_ARM_JOINTS:
        assert after[name] == before[name]
    assert [gains[-1].position[i] for i in LEFT] == [0.0] * 6
    assert [gains[-1].velocity[i] for i in LEFT] == [DEFAULT_FLOAT_KD] * 6
    assert [gains[-1].position[i] for i in RIGHT] == [
        before[n][0] for n in DUAL_OPENYAM_RIGHT_ARM_JOINTS
    ]
    status = module.status()
    assert status["floating"] == {"left": True, "right": False}
    # Grippers keep their zero-gain hold at the measured opening.
    assert commands[-1].position[-2:] == [0.3, 0.3]


def test_float_arm_refuses_bad_sides_and_kd(module: JointStepTestModule) -> None:
    with pytest.raises(StepRefusedError, match="side must be one of"):
        module.float_arm("both")
    with pytest.raises(StepRefusedError, match="float kd must be in 0..5"):
        module.float_arm("left", 5.1)
    with pytest.raises(StepRefusedError, match="float kd must be in 0..5"):
        module.float_arm("left", -0.1)
    assert module.status()["floating"] == {"left": False, "right": False}


def test_float_arm_is_refused_while_a_ramp_runs(module: JointStepTestModule) -> None:
    goal = {"right_joint1": HOME["right_joint1"] + math.radians(3.0)}
    ramp = threading.Thread(target=module.go_to, args=(goal, 1.0))
    ramp.start()
    try:
        deadline = time.monotonic() + 1.0
        while not module.status()["ramping"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert module.status()["ramping"]
        with pytest.raises(StepRefusedError, match="ramp is still running"):
            module.float_arm("left")
    finally:
        ramp.join(timeout=5.0)
    assert not module.status()["floating"]["left"]


def test_hold_arm_reseats_targets_at_the_measured_pose(module: JointStepTestModule) -> None:
    commands: list[JointState] = []
    module.joint_command.subscribe(commands.append)
    before = module.get_gains()
    module.float_arm("left", 0.3)
    time.sleep(0.3)  # the floated left_joint2 sags ~6 deg
    floating = module.status()
    assert floating["floating"]["left"]
    sagged = floating["joints"]["left_joint2"]["measured_deg"]
    assert sagged > math.degrees(HOME["left_joint2"]) + 3.0
    # While floating the published target follows the hand, not the old pose.
    assert floating["joints"]["left_joint2"]["target_error_deg"] == pytest.approx(0.0, abs=0.5)
    message = module.hold_arm("left")
    assert message.startswith("left arm held at [") and message.endswith("deg; gains restored")
    held = module.status()
    assert held["floating"] == {"left": False, "right": False}
    for name in DUAL_OPENYAM_LEFT_ARM_JOINTS:
        info = held["joints"][name]
        assert info["target_deg"] == pytest.approx(info["measured_deg"], abs=1e-6)
        assert (info["kp"], info["kd"]) == before[name]
    assert held["joints"]["left_joint2"]["target_deg"] == pytest.approx(sagged, abs=0.2)
    # The first command after hold carries the re-seated target, never the pre-float one.
    time.sleep(0.05)
    index = DUAL_OPENYAM_ARM_JOINTS.index("left_joint2")
    assert math.degrees(commands[-1].position[index]) == pytest.approx(sagged, abs=0.2)
    # Held under kp>0 the mock stops drifting.
    time.sleep(0.1)
    assert module.status()["joints"]["left_joint2"]["measured_deg"] == pytest.approx(
        sagged, abs=0.2
    )
    assert module.hold_arm("left") == "left arm already held"


def test_float_again_changes_kd_but_keeps_the_saved_gains(module: JointStepTestModule) -> None:
    module.set_gains("right_joint4", 15.0, 1.2)
    module.float_arm("right", 0.5)
    module.float_arm("right", 1.0)
    assert module.get_gains()["right_joint4"] == (0.0, 1.0)
    module.hold_arm("right")
    assert module.get_gains()["right_joint4"] == (15.0, 1.2)


def test_steps_and_ramps_are_refused_while_an_arm_floats(module: JointStepTestModule) -> None:
    module.float_arm("right")
    with pytest.raises(StepRefusedError, match="right arm is floating; hold_arm"):
        module.step("left_joint1", 1.0, hold_s=0.1)
    with pytest.raises(StepRefusedError, match="right arm is floating; hold_arm"):
        module.go_to({"left_joint1": HOME["left_joint1"]}, 10.0)
    with pytest.raises(StepRefusedError):
        module.park(speed_deg_s=30.0)
    module.hold_arm("right")
    module.step("left_joint1", 1.0, hold_s=0.05)


def test_stop_holds_a_floating_arm_before_parking() -> None:
    module = JointStepTestModule(adapter=_adapter({1: math.radians(20.0)}), loop_hz=LOOP_HZ)
    commands: list[JointState] = []
    module.joint_command.subscribe(commands.append)
    module.start()
    module.float_arm("left")
    time.sleep(0.1)
    module.stop()
    assert not module.status()["floating"]["left"]
    last = commands[-1].position[: len(DUAL_OPENYAM_ARM_JOINTS)]
    assert max(abs(v) for v in last) < math.radians(1.0)


def test_mark_float_publishes_the_window_marker(module: JointStepTestModule) -> None:
    events: list[String] = []
    module.float_events.subscribe(events.append)
    module.float_arm("left", 0.4)
    assert module.mark_float("left", 2, 10.0, {"kd": 0.4}) == "marked float window 2 on left (10 s)"
    with pytest.raises(StepRefusedError, match="duration_s"):
        module.mark_float("left", 3, 0.0)
    with pytest.raises(StepRefusedError, match="side"):
        module.mark_float("middle", 3, 1.0)
    [event] = events
    data = json.loads(event.data)
    assert data["side"] == "left" and data["index"] == 2 and data["duration_s"] == 10.0
    assert data["floating"] is True and data["kd"] == 0.4 and data["data"] == {"kd": 0.4}
    assert data["t0"] == pytest.approx(time.time(), abs=5.0)
    assert set(data["q0_deg"]) == set(DUAL_OPENYAM_LEFT_ARM_JOINTS)
    parsed = FloatEvent.from_json(event.data)
    assert parsed.index == 2 and parsed.side == "left"


# Analysis


def test_fingertip_model_moves_with_the_shoulder(left_model: FingertipModel) -> None:
    home = list(DUAL_OPENYAM_HOME_JOINTS[:6])
    tip = left_model.fingertip_m(home)
    assert np.isfinite(tip).all() and tip[2] > 0.0
    moved = list(home)
    moved[1] += math.radians(1.0)
    # One degree on joint 2 swings everything downstream: a few mm at least.
    assert 3.0 < np.linalg.norm(left_model.fingertip_m(moved) - tip) * 1000.0 < 15.0
    with pytest.raises(ValueError):
        left_model.fingertip_m(home[:5])
    with pytest.raises(ValueError):
        FingertipModel("middle")


def test_analyze_window_reports_drift_and_the_busiest_joint(left_model: FingertipModel) -> None:
    home = np.asarray(DUAL_OPENYAM_HOME_JOINTS[:6])
    t = np.linspace(0.0, 10.0, 201)
    q = np.tile(home, (t.size, 1))
    q[:, 2] += np.radians(0.02) * t  # 0.2 deg over 10 s on joint 3, nothing else
    event = FloatEvent(side="left", index=1, t0=0.0, duration_s=10.0)
    metrics = analyze_window(left_model, event, t, q)
    assert metrics.max_joint == "left_joint3"
    # Medians over the first and last 0.25 s shave a little off the full 0.2 deg.
    assert 0.17 < metrics.max_joint_drift_deg < 0.2
    assert 0.0 < metrics.fingertip_drift_mm < DRIFT_LIMIT_MM
    assert metrics.fingertip_peak_mm >= metrics.fingertip_drift_mm
    assert metrics.passed and metrics.samples == 201 and metrics.duration_s == 10.0
    still = analyze_window(left_model, event, t, np.tile(home, (t.size, 1)))
    assert still.fingertip_drift_mm == 0.0 and still.passed
    with pytest.raises(ValueError, match="sample"):
        analyze_window(left_model, event, t[:1], q[:1])


def test_segment_windows_cut_at_the_next_marker() -> None:
    a = FloatEvent(side="left", index=1, t0=100.0, duration_s=10.0)
    b = FloatEvent(side="left", index=2, t0=105.0, duration_s=10.0)
    assert segment_windows([b, a], t_end=112.0) == [(a, 100.0, 105.0), (b, 105.0, 112.0)]


# Pipeline: module with a drifting mock -> memory.db -> report


def test_float_windows_record_and_report(tmp_path: Path) -> None:
    recording_dir = tmp_path / "run"
    store = SqliteStore(path=str(recording_dir / "memory.db"))
    store.start()
    # Pose 1 floats with joint 2 sagging 6 deg/s (fails); pose 2 with 0.1 deg/s (passes).
    adapter = _adapter({1: math.radians(6.0)})
    module = JointStepTestModule(adapter=adapter, loop_hz=LOOP_HZ)
    try:
        for name, out, kind in (
            ("joint_state", module.joint_state, JointState),
            ("joint_command", module.joint_command, JointState),
            ("float_events", module.float_events, String),
        ):
            stream = store.stream(name, kind)
            out.subscribe(lambda msg, s=stream: s.append(msg, ts=getattr(msg, "ts", None)))
        module.start()
        time.sleep(0.1)
        for index, rate_deg_s in ((1, 6.0), (2, 0.1)):
            adapter.drift_rad_s[1] = math.radians(rate_deg_s)
            module.float_arm("left", 0.5)
            module.mark_float("left", index, 0.6)
            time.sleep(0.6)
            module.hold_arm("left")
            time.sleep(0.1)
    finally:
        module.stop()
        store.stop()

    out_dir = generate_report(recording_dir, plot=True)
    assert out_dir == recording_dir / REPORT_DIR_NAME
    with (out_dir / "poses.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["side"], int(r["index"])) for r in rows] == [("left", 1), ("left", 2)]
    fast, slow = rows
    assert fast["max_joint"] == "left_joint2" and fast["passed"] == "False"
    # ~6 deg/s for 0.6 s less the two 0.25 s edge medians: a few degrees, tens of mm.
    assert 1.0 < float(fast["max_joint_drift_deg"]) < 4.0
    assert float(fast["fingertip_drift_mm"]) > DRIFT_LIMIT_MM
    assert slow["passed"] == "True" and float(slow["fingertip_drift_mm"]) < DRIFT_LIMIT_MM
    assert float(slow["max_joint_drift_deg"]) < 0.1
    assert int(fast["samples"]) > 50
    report = (out_dir / "report.md").read_text()
    assert "| left | 1 |" in report and "FAIL" in report and "| ok |" in report
    assert (out_dir / "left_fingertip_drift.png").stat().st_size > 0
