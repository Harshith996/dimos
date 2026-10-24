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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from dimos.robot.manipulators.openyam.device import YamPositionDevice
from dimos.robot.manipulators.openyam.simulation import load_yam_artifact
from dimos.sim2.client import SimRobotClient
from dimos.sim2.messages import FrameMetadata, StateFrame, robot_descriptor


@pytest.fixture
def connection(artifact: Path, mocker: MockerFixture):
    model = load_yam_artifact(artifact)
    client = mocker.create_autospec(SimRobotClient, instance=True)
    client.read.return_value = StateFrame(
        FrameMetadata(1, 0, 1, 1, 0.005),
        {
            "position": np.asarray([0, 1, 1, -0.9, 0, 0, 0.5]),
            "velocity": np.zeros(7),
            "effort": np.zeros(7),
        },
    )
    client.write_positions.return_value = 8
    mocker.patch("dimos.robot.manipulators.openyam.device.SimRobotClient", return_value=client)
    device = YamPositionDevice(
        robot_descriptor("test", "yam", "test", model.robot_instance(), 0.005), model
    )
    device.connect()
    device.activate()
    try:
        yield device, client
    finally:
        device.disconnect()


def test_takeover_cannot_relabel_an_inflight_coordinator_tick(connection) -> None:
    device, client = connection
    target = device.read_joint_positions()
    device.transition("correction")
    # An unrelated status/RPC reader cannot refresh the control thread's old input.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(device.read_joint_positions).result(timeout=2)
    assert device.write_joint_positions(target) is False
    client.write_positions.assert_not_called()
    device.read_joint_positions()
    assert device.write_joint_positions(target) is True
    observed = client.write_positions.call_args.kwargs["observed"]
    assert observed is client.read.return_value


def test_only_acknowledged_commands_in_the_same_reset_generation_are_recordable(connection) -> None:
    device, client = connection
    device.transition("human")
    target = device.read_joint_positions()
    assert device.write_joint_positions(target)
    _, accepted = device.snapshot()
    assert accepted is None
    frame = client.read.return_value
    client.read.return_value = replace(
        frame, metadata=replace(frame.metadata, applied_action_sequence=8)
    )
    _, accepted = device.snapshot()
    assert accepted is not None
    assert accepted.positions == tuple(target)
    assert accepted.authority == "human"
    client.read.return_value = replace(
        frame, metadata=replace(frame.metadata, episode_id=1, applied_action_sequence=8)
    )
    assert device.snapshot()[1] is None
