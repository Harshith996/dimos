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

import numpy as np
import pytest

from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.hardware.spec import JointLimits
from dimos.robot.manipulators.openyam.control import YamHoldTask, YamTargetTask
from dimos.robot.manipulators.openyam.device import YamPositionDevice
from dimos.robot.manipulators.openyam.simulation import PUBLIC_JOINTS


@pytest.fixture
def device(mocker):
    device = mocker.create_autospec(YamPositionDevice, instance=True)
    device.get_limits.return_value = JointLimits(
        position_lower=[-2.0] * 6 + [0.0],
        position_upper=[2.0] * 6 + [1.0],
        velocity_max=[0.5] * 6 + [1.0],
    )
    device.observed_generation.return_value = 0
    return device


def state(positions, *, tick=1):
    return CoordinatorState(
        joints=JointStateSnapshot(joint_positions=dict(zip(PUBLIC_JOINTS, positions, strict=True))),
        t_now=tick * 0.01,
        dt=0.01,
    )


def test_hold_reseeds_only_when_its_control_tick_observes_a_reset(device):
    task = YamHoldTask(device)
    task.request("manual")
    before = state([0.8] * 7)
    reset = state([0.0] * 7)

    assert task.compute(before).positions == [0.8] * 7
    assert task.compute(reset).positions == [0.8] * 7

    device.observed_generation.return_value = 1
    assert task.compute(reset).positions == [0.0] * 7
    assert task.compute(before).positions == [0.0] * 7


def test_hold_clamps_physical_overshoot_without_corrupting_feedback(device):
    task = YamHoldTask(device)
    task.request("manual")
    measured = state([2.01, -2.01, 0, 0, 0, 0, 1.0000001])

    assert task.compute(measured).positions == [2.0, -2.0, 0, 0, 0, 0, 1.0]
    assert list(measured.joints.joint_positions.values()) == [2.01, -2.01, 0, 0, 0, 0, 1.0000001]
    assert task.compute(state([0.0] * 7)).positions == [2.0, -2.0, 0, 0, 0, 0, 1.0]


def test_repeated_setpoints_settle_even_when_feedback_never_reaches_them(device):
    task = YamTargetTask(device)
    target = [0.02, 0.4, 0.3, 0.1, 0, 0, 1.0]
    previous = np.zeros(7)
    for tick in range(1, 301):
        # 20 Hz intent, 100 Hz control, gravity/contact keep feedback off target.
        if tick % 5 == 1:
            task.set_target(target)
        output = np.array(task.compute(state([0.0] * 7, tick=tick)).positions)
        assert np.all(np.abs(output - previous) <= np.array([0.005] * 6 + [0.01]) + 1e-12)
        if tick >= 100:
            assert output == pytest.approx(target)
        previous = output
    assert task.is_active()

    # A new goal starts at the previous command, even after reaching the old goal.
    task.set_target([-0.1] * 6 + [0.5])
    assert task.compute(state([0.0] * 7, tick=301)).positions == pytest.approx(
        np.array(target) - np.array([0.005] * 6 + [0.01])
    )


@pytest.mark.parametrize("boundary", ["clear", "preempt", "reset"])
def test_authority_boundaries_forget_old_targets_and_command_references(device, boundary):
    task = YamTargetTask(device)
    task.set_target([0.5] * 7)
    assert task.compute(state([0.0] * 7)).positions == pytest.approx([0.005] * 6 + [0.01])

    if boundary == "clear":
        task.clear()
    elif boundary == "preempt":
        task.on_preempted("operator_hold", frozenset(PUBLIC_JOINTS))
    else:
        device.observed_generation.return_value = 1
        assert task.compute(state([0.8] * 7)) is None
    assert not task.is_active()
    assert task.compute(state([0.8] * 7)) is None

    task.set_target([0.5] * 7)
    assert task.compute(state([0.8] * 7)).positions == pytest.approx([0.795] * 6 + [0.79])


@pytest.mark.parametrize("target", [[0.0], [float("nan")] * 7, [0.0] * 6 + [1.1]])
def test_invalid_setpoints_do_not_replace_an_accepted_goal(device, target):
    task = YamTargetTask(device)
    task.set_target([0.1] * 7)
    with pytest.raises(ValueError, match="finite positions within joint limits"):
        task.set_target(target)
    assert task.compute(state([0.1] * 7)).positions == [0.1] * 7
