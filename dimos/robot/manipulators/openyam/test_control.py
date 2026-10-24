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

from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.robot.manipulators.openyam.control import YamHoldTask
from dimos.robot.manipulators.openyam.device import YamPositionDevice


def test_hold_reseeds_only_when_its_control_tick_observes_a_reset(mocker):
    device = mocker.create_autospec(YamPositionDevice, instance=True)
    device.observed_generation.return_value = 0
    task = YamHoldTask(device)
    task.request("manual")
    before = CoordinatorState(joints=JointStateSnapshot(joint_positions={"joint": 0.8}))
    reset = CoordinatorState(joints=JointStateSnapshot(joint_positions={"joint": 0.0}))

    assert task.compute(before).positions == [0.8]
    assert task.compute(reset).positions == [0.8]

    device.observed_generation.return_value = 1
    assert task.compute(reset).positions == [0.0]
    assert task.compute(before).positions == [0.0]
