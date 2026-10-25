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

"""Compose existing coordinator tasks for the single-arm learning example."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from threading import RLock
import time
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator
from dimos.control.task import (
    CoordinatorState,
    JointCommandOutput,
)
from dimos.control.tasks.gripper_task.gripper_task import (
    GripperControlTask,
    GripperControlTaskConfig,
)
from dimos.control.tasks.operator_hold_task.operator_hold_task import OperatorHoldTask
from dimos.control.tasks.teleop_ik_task.teleop_ik_task import (
    TeleopHandBinding,
    TeleopIKTask,
    TeleopIKTaskConfig,
)
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JointTrajectoryTask,
    JointTrajectoryTaskConfig,
)
from dimos.core.rpc_client import RPCClient
from dimos.core.transport import ZenohTransport
from dimos.hardware.manipulators.spec import ManipulatorAdapter
from dimos.hardware.whole_body.spec import WholeBodyAdapter
from dimos.imitation.collection.episode import EpisodeStatus
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.msgs.std_msgs.String import String
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.robot.manipulators.openyam.config import make_openyam_model_config
from dimos.robot.manipulators.openyam.device import AuthorityMode, LearningControlDevice
from dimos.robot.manipulators.openyam.simulation import PUBLIC_JOINTS
from dimos.robot.manipulators.openyam.teleop_ik import OpenYamPinkPoseTargetSolver
from dimos.teleop.webxr.controller_types import Buttons
from dimos.teleop.webxr.extensions import VideoArmTeleopModule


class YamTrajectoryTask(JointTrajectoryTask):
    """Use the shared trajectory controller, fencing simulation world resets."""

    def __init__(self, device: LearningControlDevice, hold: OperatorHoldTask) -> None:
        limits = device.get_limits()
        velocities = {
            name: value
            for name, value in zip(PUBLIC_JOINTS, limits.velocity_max, strict=True)
            if value is not None
        }
        super().__init__(
            JointTrajectoryTaskConfig(
                joint_names=PUBLIC_JOINTS,
                velocity_limits=velocities,
                hold_final_position=True,
            )
        )
        self._device = device
        self._hold = hold
        self._generation: int | None = None

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        generation = self._device.observed_generation()
        if self._generation is not None and generation != self._generation:
            self.cancel()
            self._hold.request("manual", "Simulation reset; explicit resume required")
            self._generation = generation
            return None
        self._generation = generation
        return super().compute(state)


class YamHoldTask(OperatorHoldTask):
    """Reseed the existing hold from the same tick that observes a world reset."""

    def __init__(self, device: LearningControlDevice) -> None:
        super().__init__()
        self._device = device
        self._generation: int | None = None
        limits = device.get_limits()
        self._bounds = {
            name: (lower, upper)
            for name, lower, upper in zip(
                PUBLIC_JOINTS, limits.position_lower, limits.position_upper, strict=True
            )
            if lower is not None and upper is not None
        }
        if len(self._bounds) != len(PUBLIC_JOINTS):
            raise ValueError("YAM hold requires position limits for every joint")

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        if not self.is_active():
            return None
        generation = self._device.observed_generation()
        if generation != self._generation:
            self.acknowledge()
            self.request("manual", "Hold the current simulation generation")
            self._generation = generation
        output = super().compute(state)
        if output is None or output.positions is None:
            return output
        # Feedback is physical state, not necessarily a legal command: contact
        # or servo overshoot can put a joint beyond its narrower target limits.
        return replace(
            output,
            positions=[
                min(self._bounds[name][1], max(self._bounds[name][0], value))
                for name, value in zip(output.joint_names, output.positions, strict=True)
            ],
        )


class YamControl:
    """Own coordinator/teleop lifetimes; the coordinator alone writes the device."""

    def __init__(
        self,
        device: LearningControlDevice,
        component: HardwareComponent | None = None,
        *,
        hand: Literal["left", "right"] = "right",
    ) -> None:
        self.device = device
        self.hand = hand
        self.component = component or HardwareComponent(
            hardware_id="openyam",
            hardware_type=HardwareType.MANIPULATOR,
            joints=list(PUBLIC_JOINTS),
        )
        self._stack = ExitStack()
        self._teleop_stack = ExitStack()
        self._lock = RLock()
        self._mode: AuthorityMode = "hold"
        self._grip = False
        self._await_release = False
        self._ik: TeleopIKTask | None = None
        self._buttons_at = 0.0
        self.on_authority: Callable[[AuthorityMode], None] | None = None
        self.on_record: Callable[[Literal["toggle", "discard"]], None] | None = None
        self._record_buttons = (False, False)
        self._module: VideoArmTeleopModule | None = None
        self._status: EpisodeStatus | None = None
        self._id = f"yam-learning-{uuid4().hex}"
        self.teleop_port: int | None = None
        self.coordinator = ControlCoordinator(publish_joint_state=False, rpc_name=self._id)
        self._hold = YamHoldTask(device)
        self._target = YamTrajectoryTask(device, self._hold)
        self._gripper = GripperControlTask(
            "yam_gripper",
            GripperControlTaskConfig([PUBLIC_JOINTS[-1]], priority=20, hold_duration=0.2),
            [(0.0, 1.0)],
        )

    def start(self, *, enable: bool = True) -> None:
        try:
            if not self.device.connect():
                raise RuntimeError("OpenYAM connection failed")
            self._stack.callback(self.device.disconnect)
            if enable and not self.device.activate():
                raise RuntimeError("OpenYAM activation failed")
            self.coordinator.add_hardware(
                cast("ManipulatorAdapter | WholeBodyAdapter", self.device), self.component
            )
            self.coordinator.add_task(self._target)
            self.coordinator.add_task(self._hold)
            self.coordinator.add_task(self._gripper)
            self._hold.request("manual", "Start held")
            self._stack.callback(self.coordinator.stop)
            self.coordinator.start()
        except BaseException:
            self.close()
            raise

    def transition(self, mode: AuthorityMode) -> int:
        with self._lock:
            if mode == "hold" and self._grip:
                self._await_release = True
            epoch = self.device.transition(mode)
            self.coordinator.cancel_trajectory()
            self._hold.request("manual", "Authority changed")
            if self._ik is not None:
                self._ik.set_estop(True)
                self._ik.set_estop(False)
            self._gripper.set_estop(True)
            self._gripper.set_estop(False)
            self._mode = mode
            self._grip = False
            if mode == "policy":
                self._hold.acknowledge()
            return epoch

    @property
    def teleop_engaged(self) -> bool:
        with self._lock:
            return self._grip and time.monotonic() - self._buttons_at < 0.5

    def target(self, positions: list[float]) -> None:
        with self._lock:
            if self._mode == "hold" or self._grip:
                return
            values = np.asarray(positions)
            limits = self.device.get_limits()
            if (
                values.shape != (len(PUBLIC_JOINTS),)
                or not np.isfinite(values).all()
                or np.any(values < limits.position_lower)
                or np.any(values > limits.position_upper)
            ):
                raise ValueError("target must contain seven finite positions within joint limits")
            self._hold.acknowledge()
            self.coordinator.execute_trajectory(
                JointTrajectory(
                    joint_names=list(PUBLIC_JOINTS), points=[TrajectoryPoint(positions=positions)]
                )
            )

    def policy_control(self) -> RPCClient:
        """Bind the isolated policy to this coordinator's unique RPC endpoint."""
        return RPCClient.remote(ControlCoordinator, remote_name=self._id)

    def start_teleop(self, port: int) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("WebXR port must be between 1 and 65535")
        with self._lock:
            if self.teleop_port is not None:
                raise RuntimeError("WebXR is already running")
            model = make_openyam_model_config()
            ik = TeleopIKTask(
                "yam_quest",
                TeleopIKTaskConfig(
                    joint_names=PUBLIC_JOINTS[:6],
                    robot_model=model,
                    bindings=(TeleopHandBinding(self.hand, "gripper_tip"),),
                    pink=PinkKinematicsConfig(
                        posture_cost=0.01, joint_limit_posture_margin=0.3, lm_damping=0.01, gain=1.0
                    ),
                    priority=20,
                    max_joint_velocity_rad_s=0.5,
                ),
                solver_type=OpenYamPinkPoseTargetSolver,
            )
            module = VideoArmTeleopModule(
                server_port=port, input_timeout_s=0.5, rpc_name=f"{self._id}-quest"
            )
            stack = ExitStack()
            try:
                for name, message_type in (("status", String), ("color_image", Image)):
                    transport: ZenohTransport[Any] = ZenohTransport(
                        f"{self._id}/{name}", message_type
                    )
                    stack.callback(transport.stop)
                    module.set_transport(name, transport)
                self.coordinator.add_task(ik)
                stack.callback(self.coordinator.remove_task, ik.name)
                self._ik = ik
                poses = (
                    module.left_controller_output
                    if self.hand == "left"
                    else module.right_controller_output
                )
                gripper = (
                    module.left_gripper_command
                    if self.hand == "left"
                    else module.right_gripper_command
                )
                stack.callback(poses.subscribe(self._pose))
                stack.callback(gripper.subscribe(self._gripper_target))
                stack.callback(module.teleop_buttons.subscribe(self._buttons))
                # Stop the producer before disconnecting its subscriptions.
                stack.callback(module.stop)
                module.build()
                module.start()
                self._teleop_stack = stack.pop_all()
                self._module = module
                self.teleop_port = port
                if self._status is not None:
                    self.report_recording(self._status)
            except BaseException:
                stack.close()
                self._ik = None
                raise

    def _pose(self, pose: PoseStamped) -> None:
        with self._lock:
            if self._ik is not None:
                if self.hand == "left":
                    self._ik.on_left_cartesian_command(pose, time.monotonic())
                else:
                    self._ik.on_right_cartesian_command(pose, time.monotonic())

    def _buttons(self, buttons: Buttons) -> None:
        with self._lock:
            pressed = buttons.left_grip if self.hand == "left" else buttons.right_grip
            if self._await_release:
                if pressed:
                    return
                self._await_release = False
            previous = self._grip
            callback = self.on_authority
            record_callback = self.on_record
            primary, secondary = (
                (buttons.left_primary, buttons.left_secondary)
                if self.hand == "left"
                else (buttons.right_primary, buttons.right_secondary)
            )
            start_save = primary and not self._record_buttons[0]
            discard = secondary and not self._record_buttons[1]
            self._record_buttons = (primary, secondary)
        # The application lock must never be acquired while holding the control lock.
        if callback is not None and pressed != previous:
            callback("human" if pressed else "hold")
        if record_callback is not None:
            if discard:
                record_callback("discard")
            elif start_save:
                record_callback("toggle")
        with self._lock:
            self._buttons_at = time.monotonic()
            self._grip = pressed and self._mode in ("human", "correction")
            if self._ik is not None:
                self._ik.on_teleop_buttons(buttons, self._buttons_at)
            if self._grip:
                self._hold.acknowledge()

    def check_input(self) -> None:
        with self._lock:
            expired = self._grip and time.monotonic() - self._buttons_at >= 0.5
            if expired:
                self._await_release = True
            callback = self.on_authority
        if expired and callback is not None:
            callback("hold")

    def _gripper_target(self, message: Float32) -> None:
        with self._lock:
            if self._grip:
                self._gripper.set_position([1.0 - message.data], time.monotonic())

    def publish_camera(self, image: Image) -> None:
        with self._lock:
            if self._module is not None:
                self._module.color_image.transport.publish(image)

    def report_recording(self, status: EpisodeStatus) -> None:
        with self._lock:
            self._status = status
            if self._module is not None:
                self._module.status.transport.publish(String(data=status.to_json()))

    def close(self) -> None:
        self._teleop_stack.close()
        self._module = None
        self.teleop_port = None
        self._stack.close()
