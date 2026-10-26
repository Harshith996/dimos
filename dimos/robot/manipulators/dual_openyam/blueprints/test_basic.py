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


import math
from pathlib import Path

import pytest

from dimos.control.coordinator import ControlCoordinator
from dimos.control.hardware_interface import ConnectedWholeBody
from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
from dimos.robot.assets.model import LoadedRobotModel, RobotModel
from dimos.robot.manipulators.dual_openyam.blueprints.basic import (
    DualOpenYamCoordinator,
    dual_openyam_trajectory_task,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_ARM_JOINTS


def test_trajectory_checks_measured_endpoint():
    params = dual_openyam_trajectory_task().params
    assert params["goal_tolerance"] == pytest.approx(math.radians(0.5))
    assert params["goal_time_tolerance"] == 2.0
    assert params["tracking_gain"] == 0.5
    assert params["max_tracking_correction"] == pytest.approx(math.radians(1.0))


@pytest.mark.parametrize(
    ("config", "adapter_type"),
    [
        ({}, "mock_whole_body"),
        ({"left_can_port": "left", "right_can_port": "right"}, "dual_openyam_damiao"),
        ({"use_transport_hardware": True}, "transport_lcm"),
    ],
)
def test_setup_supplies_model_limits_to_hardware(mocker, config, adapter_type):
    names = DUAL_OPENYAM_ARM_JOINTS
    # Different bounds and reverse document order catch accidental positional mapping.
    bounds = {name: (-float(i + 1), float(i + 2)) for i, name in enumerate(names)}
    xml = (
        '<robot name="test"><link name="base"/>'
        + "".join(
            f'<link name="{name}_link"/><joint name="{name}" type="revolute">'
            f'<parent link="base"/><child link="{name}_link"/>'
            f'<limit lower="{bounds[name][0]}" upper="{bounds[name][1]}"/></joint>'
            for name in reversed(names)
        )
        + "</robot>"
    )
    mocker.patch.object(
        RobotModel, "load", return_value=LoadedRobotModel(xml, Path("model.urdf"), {})
    )
    mocker.patch.object(ControlCoordinator, "_setup_from_config")
    coordinator = DualOpenYamCoordinator(**config)
    try:
        coordinator._setup_from_config()
        component = coordinator.config.hardware[0]
        assert component.adapter_type == adapter_type
        adapter = MockWholeBodyAdapter(dof=len(component.joints))
        hardware = ConnectedWholeBody(adapter, component)
        limits = hardware.get_limits()
        assert limits is not None
        assert limits.position_lower == [*[bounds[name][0] for name in names], 0.0, 0.0]
        assert limits.position_upper == [*[bounds[name][1] for name in names], 1.0, 1.0]
    finally:
        coordinator.stop()


def test_transport_hardware_rejects_can_ports() -> None:
    coordinator = DualOpenYamCoordinator(
        use_transport_hardware=True,
        left_can_port="left",
        right_can_port="right",
    )
    try:
        with pytest.raises(ValueError, match="cannot also use CAN ports"):
            coordinator._setup_from_config()
    finally:
        coordinator.stop()
