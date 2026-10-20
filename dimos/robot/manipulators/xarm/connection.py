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

"""One xArm connection for the SDK and sim2, with identical typed joint ports."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from dimos.control.contract.keys import Interface, Unit
from dimos.control.contract.ports import JointPort
from dimos.control.contract.presets import GripperSpec
from dimos.robot.manipulators.common.connection import ArmConnection, ArmConnectionConfig
from dimos.robot.manipulators.xarm.config import make_xarm_model_config
from dimos.robot.manipulators.xarm.sim2 import xarm_definition
from dimos.sim2.scene import scene_robot
from dimos.sim2.spec import RobotInstance

if TYPE_CHECKING:
    from dimos.hardware.manipulators.xarm.adapter import XArmAdapter


class XArmConnectionConfig(ArmConnectionConfig):
    robot_type: Literal["xarm", "lite"] = "xarm"
    position_command: JointPort = JointPort()

    @model_validator(mode="after")
    def _arm(self) -> Self:
        self.model = make_xarm_model_config(
            self.dof, robot_type=self.robot_type, add_gripper=False, joint_prefix=""
        ).model
        self.position_command = JointPort(
            joints=[f"joint{i}" for i in range(1, self.dof + 1)],
            gripper=GripperSpec(unit=Unit.UNITLESS, lo=0, hi=850) if self.gripper else None,
        )
        self.velocity_command = JointPort(
            interface=Interface.VELOCITY,
            joints=[f"joint{i}" for i in range(1, self.dof + 1)],
        )
        return self


class XArmConnection(ArmConnection):
    config: XArmConnectionConfig
    adapter_type = "xarm"

    def write_gripper_position(self, position: float) -> None:
        device = cast("XArmAdapter", self._device)
        if not device.write_gripper_position(position):
            raise RuntimeError("xArm rejected gripper command")

    @staticmethod
    def describe_simulated_robot(config: BaseModel, scene: Path) -> RobotInstance:
        config = cast("XArmConnectionConfig", config)
        if config.robot_type != "xarm":
            raise ValueError("Lite6 has no simulated asset; it cannot use an xArm6 body")
        definition = xarm_definition(
            config.dof,
            source=config.source,
            gripper=config.gripper,
            home=config.initial_positions,
            camera_pointcloud=config.camera_pointcloud,
        )
        if config.base_pose is not None:
            return RobotInstance(
                definition,
                xyz=config.base_pose.position.to_tuple(),
                rpy=config.base_pose.orientation.to_euler().to_tuple(),
            )
        return scene_robot(
            scene,
            definition,
            config.spawn,
            default=(0.0, 0.0, 0.12),
        )
