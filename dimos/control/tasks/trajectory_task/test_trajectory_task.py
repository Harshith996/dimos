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

from math import radians
from types import SimpleNamespace

import pytest

from dimos.control.task import (
    CoordinatorState,
    JointStateSnapshot,
)
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JointTrajectoryTask,
    JointTrajectoryTaskConfig,
    TrajectoryExecutionStatus,
    create_task,
)
from dimos.hardware.spec import JointLimits
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.msgs.trajectory_msgs.TrajectoryStatus import TrajectoryState


@pytest.mark.parametrize("single_point", [False, True])
@pytest.mark.parametrize("preempted", [False, True])
def test_jtt_does_not_pull_joint_back_after_another_task_moves_it(single_point, preempted):
    task = JointTrajectoryTask(JointTrajectoryTaskConfig(joint_names=["joint"]))
    state = CoordinatorState(
        joints=JointStateSnapshot(joint_positions={"joint": 0.0}), t_now=0.1, dt=0.1
    )
    first_target = 1.0 if preempted else 0.1
    first = JointTrajectory(
        joint_names=["joint"], points=[TrajectoryPoint(positions=[first_target])]
    )
    assert task.execute(first, {}).status is TrajectoryExecutionStatus.ACCEPTED
    output = task.compute(state)
    assert output is not None
    assert output.positions == pytest.approx([0.1])
    if preempted:
        task.on_preempted("other_task", frozenset({"joint"}))
    assert not task.is_active()

    # Another task moves the joint away from JTT's previous command.
    state = CoordinatorState(
        joints=JointStateSnapshot(joint_positions={"joint": -1.0}), t_now=1.0, dt=0.01
    )
    assert task.compute(state) is None
    points = [TrajectoryPoint(positions=[-1.2])]
    if not single_point:
        points = [
            TrajectoryPoint(positions=[-1.0]),
            TrajectoryPoint(positions=[-1.2], time_from_start=0.1),
        ]
    assert (
        task.execute(
            JointTrajectory(joint_names=["joint"], points=points), state.joints.joint_positions
        ).status
        is TrajectoryExecutionStatus.ACCEPTED
    )

    # The new motion goes farther away from the old command. Every output
    # must move toward the new target, bounded from the new starting pose.
    previous = -1.0
    for tick in range(30):
        state.t_now = 1.0 + tick * state.dt
        output = task.compute(state)
        if output is None:
            break
        commanded = output.positions[0]
        assert -state.dt - 1e-9 <= commanded - previous <= 1e-9
        previous = commanded
    assert previous == pytest.approx(-1.2)
    assert not task.is_active()
    assert task.compute(state) is None


def test_completed_joint_reanchors_while_other_joint_keeps_command_continuity():
    task = JointTrajectoryTask(JointTrajectoryTaskConfig(joint_names=["finished", "running"]))
    measured = JointStateSnapshot(joint_positions={"finished": 0.0, "running": 0.0})
    initial = JointTrajectory(
        joint_names=["finished", "running"], points=[TrajectoryPoint(positions=[0.1, 1.0])]
    )
    assert task.execute(initial, {}).status is TrajectoryExecutionStatus.ACCEPTED
    output = task.compute(CoordinatorState(joints=measured, t_now=0.1, dt=0.1))
    assert output is not None
    assert output.positions == pytest.approx([0.1, 0.1])
    measured.joint_positions["finished"] = -1.0
    replacement = JointTrajectory(
        joint_names=["finished"],
        points=[
            TrajectoryPoint(positions=[-1.0]),
            TrajectoryPoint(positions=[0.1], time_from_start=2.0),
        ],
    )
    assert (
        task.execute(replacement, measured.joint_positions).status
        is TrajectoryExecutionStatus.ACCEPTED
    )
    output = task.compute(CoordinatorState(joints=measured, t_now=0.2, dt=0.1))
    assert output is not None
    assert output.positions == pytest.approx([-1.0, 0.2])


@pytest.mark.parametrize(
    "positions, expected",
    [
        ({}, TrajectoryExecutionStatus.START_STATE_UNAVAILABLE),
        ({"joint": -1.0}, TrajectoryExecutionStatus.START_STATE_MISMATCH),
    ],
)
def test_completed_trajectory_does_not_bypass_start_validation(positions, expected):
    task = JointTrajectoryTask(JointTrajectoryTaskConfig(joint_names=["joint"]))
    initial = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.1])])
    assert task.execute(initial, {}).status is TrajectoryExecutionStatus.ACCEPTED
    task.compute(
        CoordinatorState(
            joints=JointStateSnapshot(joint_positions={"joint": 0.0}), t_now=0.1, dt=0.1
        )
    )
    trajectory = JointTrajectory(
        joint_names=["joint"],
        points=[
            TrajectoryPoint(positions=[0.1]),
            TrajectoryPoint(positions=[1.0], time_from_start=2.0),
        ],
    )
    assert task.execute(trajectory, positions).status is expected


def test_measured_goal_does_not_complete_from_command_alone():
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"], goal_tolerance=0.01, goal_time_tolerance=0.5
        )
    )
    target = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.1])])
    assert task.execute(target, {}).status is TrajectoryExecutionStatus.ACCEPTED
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    output = task.compute(CoordinatorState(joints=measured, t_now=1.0, dt=0.1))
    assert output is not None
    assert output.positions == pytest.approx([0.1])
    assert task.get_state() is TrajectoryState.EXECUTING

    output = task.compute(CoordinatorState(joints=measured, t_now=1.05, dt=0.05))
    assert output is not None
    assert output.positions == pytest.approx([0.1])
    assert task.get_state() is TrajectoryState.EXECUTING

    measured.joint_positions["joint"] = 0.095
    task.compute(CoordinatorState(joints=measured, t_now=1.1, dt=0.1))
    assert task.get_state() is TrajectoryState.COMPLETED


def test_measured_goal_times_out_and_reports_error():
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"], goal_tolerance=0.01, goal_time_tolerance=0.5
        )
    )
    target = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.1])])
    assert task.execute(target, {}).status is TrajectoryExecutionStatus.ACCEPTED
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    task.compute(CoordinatorState(joints=measured, t_now=1.0, dt=0.1))
    task.compute(CoordinatorState(joints=measured, t_now=1.6, dt=0.1))
    status = task.get_status(1.6)
    assert status.state is TrajectoryState.FAULT
    assert "joint" in status.error
    assert "0.1000" in status.error
    assert "0.0000" in status.error


def test_tracking_feedback_biases_target_then_releases_when_measured_catches_up():
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"],
            goal_tolerance=0.01,
            tracking_gain=0.5,
            max_tracking_correction=0.02,
        )
    )
    target = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.05])])
    assert task.execute(target, {}).status is TrajectoryExecutionStatus.ACCEPTED
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    first = task.compute(CoordinatorState(joints=measured, t_now=1.0, dt=0.1))
    assert first is not None
    assert first.positions == pytest.approx([0.07])
    assert task.get_state() is TrajectoryState.EXECUTING

    measured.joint_positions["joint"] = 0.05
    second = task.compute(CoordinatorState(joints=measured, t_now=1.1, dt=0.1))
    assert second is not None
    assert second.positions == pytest.approx([0.05])
    assert task.get_state() is TrajectoryState.COMPLETED


def test_tracking_feedback_respects_rate_and_joint_limits():
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"],
            goal_tolerance=0.01,
            tracking_gain=1.0,
            max_tracking_correction=0.1,
            joint_position_limits={"joint": (0.0, 0.06)},
        )
    )
    target = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.05])])
    assert task.execute(target, {}).status is TrajectoryExecutionStatus.ACCEPTED
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    first = task.compute(CoordinatorState(joints=measured, t_now=1.0, dt=0.02))
    assert first is not None
    assert first.positions == pytest.approx([0.02])
    second = task.compute(CoordinatorState(joints=measured, t_now=1.02, dt=0.1))
    assert second is not None
    assert second.positions == pytest.approx([0.06])


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("error_degrees", [0.5, 3.5])
def test_replacement_does_not_apply_tracking_feedback_twice(
    direction: int, error_degrees: float
) -> None:
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"],
            goal_tolerance=radians(0.1),
            tracking_gain=0.5,
            max_tracking_correction=radians(1),
        )
    )
    target = radians(115.6)
    measured = JointStateSnapshot(
        joint_positions={"joint": target + direction * radians(error_degrees)}
    )
    initial = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[target])])
    assert task.execute(initial, {}).status is TrajectoryExecutionStatus.ACCEPTED
    before = task.compute(CoordinatorState(joints=measured, t_now=1.0, dt=0.1))
    assert before is not None
    previous = before.positions[0]

    # Move against the existing feedback bias, as in the near-banana test.
    replacement = JointTrajectory(
        joint_names=["joint"],
        points=[
            TrajectoryPoint(positions=[previous]),
            TrajectoryPoint(positions=[target + direction * radians(3)], time_from_start=16 / 30),
        ],
    )
    assert (
        task.execute(replacement, measured.joint_positions).status
        is TrajectoryExecutionStatus.ACCEPTED
    )
    for tick in range(20):
        output = task.compute(CoordinatorState(joints=measured, t_now=1.01 + tick * 0.01, dt=0.01))
        assert output is not None
        if tick == 0:
            assert output.positions == pytest.approx(before.positions)
        # No wrong-direction kick, and the motor command still respects 1 rad/s.
        assert -1e-9 <= direction * (output.positions[0] - previous) <= 0.01 + 1e-9
        previous = output.positions[0]


def test_repeated_replacements_do_not_accumulate_tracking_feedback() -> None:
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"],
            goal_tolerance=0.001,
            tracking_gain=0.5,
            max_tracking_correction=0.02,
        )
    )
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    initial = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.05])])
    assert task.execute(initial, {}).status is TrajectoryExecutionStatus.ACCEPTED
    before = task.compute(CoordinatorState(joints=measured, t_now=0.0, dt=0.1))
    assert before is not None
    for chunk in range(5):
        replacement = JointTrajectory(
            joint_names=["joint"],
            points=[
                TrajectoryPoint(positions=before.positions),
                TrajectoryPoint(positions=[0.05], time_from_start=1.0),
            ],
        )
        assert (
            task.execute(replacement, measured.joint_positions).status
            is TrajectoryExecutionStatus.ACCEPTED
        )
        for tick in range(20):
            output = task.compute(
                CoordinatorState(joints=measured, t_now=chunk + tick * 0.01, dt=0.01)
            )
            assert output is not None
            assert output.positions == pytest.approx(before.positions)


@pytest.mark.parametrize("dt, upper", [(0.01, None), (0.1, 0.06)])
@pytest.mark.parametrize("new_measurement", [0.0, 0.03])
def test_replacement_anchors_to_actual_output_after_limits(
    dt: float, upper: float | None, new_measurement: float
) -> None:
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"],
            goal_tolerance=0.001,
            tracking_gain=0.5,
            max_tracking_correction=0.02,
            joint_position_limits={"joint": (0.0, upper)},
        )
    )
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    initial = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.05])])
    assert task.execute(initial, {}).status is TrajectoryExecutionStatus.ACCEPTED
    before = task.compute(CoordinatorState(joints=measured, t_now=0.0, dt=dt))
    assert before is not None
    measured.joint_positions["joint"] = new_measurement
    reference = task.get_reference_positions(measured.joint_positions)
    replacement = JointTrajectory(
        joint_names=["joint"],
        points=[
            TrajectoryPoint(positions=[reference["joint"]]),
            TrajectoryPoint(positions=[0.05], time_from_start=1.0),
        ],
    )
    assert (
        task.execute(replacement, measured.joint_positions).status
        is TrajectoryExecutionStatus.ACCEPTED
    )
    after = task.compute(CoordinatorState(joints=measured, t_now=0.1, dt=0.01))
    assert after is not None
    assert after.positions == pytest.approx(before.positions)


@pytest.mark.parametrize("positions", [{}, {"joint": float("nan")}])
def test_feedback_replacement_requires_measurement_without_disturbing_active_motion(
    positions: dict[str, float],
) -> None:
    task = JointTrajectoryTask(
        JointTrajectoryTaskConfig(
            joint_names=["joint"],
            goal_tolerance=0.001,
            tracking_gain=0.5,
            max_tracking_correction=0.02,
        )
    )
    measured = JointStateSnapshot(joint_positions={"joint": 0.0})
    initial = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.05])])
    assert task.execute(initial, {}).status is TrajectoryExecutionStatus.ACCEPTED
    before = task.compute(CoordinatorState(joints=measured, t_now=0.0, dt=0.1))
    assert before is not None
    replacement = JointTrajectory(
        joint_names=["joint"],
        points=[
            TrajectoryPoint(positions=before.positions),
            TrajectoryPoint(positions=[0.1], time_from_start=1.0),
        ],
    )
    assert (
        task.execute(replacement, positions).status
        is TrajectoryExecutionStatus.START_STATE_UNAVAILABLE
    )
    assert task.is_active()
    after = task.compute(CoordinatorState(joints=measured, t_now=0.1, dt=0.1))
    assert after is not None
    assert after.positions == pytest.approx(before.positions)


def test_tracking_factory_reads_connected_hardware_limits():
    hardware = SimpleNamespace(
        joint_names=["joint"],
        get_limits=lambda: JointLimits(
            position_lower=[0.0], position_upper=[0.06], velocity_max=[None]
        ),
    )
    config = SimpleNamespace(
        name="joint_trajectory",
        joint_names=["joint"],
        priority=20,
        params={
            "goal_tolerance": 0.01,
            "tracking_gain": 1.0,
            "max_tracking_correction": 0.1,
        },
    )
    task = create_task(config, {"hw": hardware})
    target = JointTrajectory(joint_names=["joint"], points=[TrajectoryPoint(positions=[0.05])])
    assert task.execute(target, {}).status is TrajectoryExecutionStatus.ACCEPTED
    output = task.compute(
        CoordinatorState(
            joints=JointStateSnapshot(joint_positions={"joint": 0.0}), t_now=1.0, dt=0.1
        )
    )
    assert output is not None
    assert output.positions == pytest.approx([0.06])
