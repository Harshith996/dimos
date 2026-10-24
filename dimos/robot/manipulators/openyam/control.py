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
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator
from dimos.control.task import (
    BaseControlTask,
    ControlMode,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
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
from dimos.core.transport import ZenohTransport
from dimos.imitation.collection.episode import EpisodeStatus
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.openyam.config import make_openyam_model_config
from dimos.robot.manipulators.openyam.device import AuthorityMode, YamPositionDevice
from dimos.robot.manipulators.openyam.simulation import PUBLIC_JOINTS
from dimos.robot.manipulators.openyam.teleop_ik import OpenYamPinkPoseTargetSolver
from dimos.teleop.webxr.controller_types import Buttons
from dimos.teleop.webxr.extensions import VideoArmTeleopModule


class YamTargetTask(BaseControlTask):
    """Persistent setpoints, rate-limited on the coordinator's clock.

    A completed finite trajectory releases its command reference. Streaming
    that same trajectory again would then ramp from gravity-deflected feedback.
    Keep this reference until authority changes, including after reaching a goal.
    Native artifact servos remain responsible for tracking these positions.
    """

    def __init__(self, device: YamPositionDevice) -> None:
        self._name = "yam_target"
        self._device = device
        limits = device.get_limits()
        self._lower = np.asarray(limits.position_lower, dtype=np.float64)
        self._upper = np.asarray(limits.position_upper, dtype=np.float64)
        self._velocity = np.asarray(limits.velocity_max, dtype=np.float64)
        if (
            any(
                values.shape != (len(PUBLIC_JOINTS),)
                for values in (self._lower, self._upper, self._velocity)
            )
            or not np.isfinite([self._lower, self._upper, self._velocity]).all()
            or np.any(self._lower >= self._upper)
            or np.any(self._velocity <= 0)
        ):
            raise ValueError("YAM target control requires finite position and velocity limits")
        self._lock = RLock()
        self._target: NDArray[np.float64] | None = None
        self._command: NDArray[np.float64] | None = None
        self._generation: int | None = None

    def claim(self) -> ResourceClaim:
        return ResourceClaim(frozenset(PUBLIC_JOINTS), 10, ControlMode.SERVO_POSITION)

    def is_active(self) -> bool:
        return self._target is not None

    def set_target(self, positions: list[float]) -> None:
        values = np.asarray(positions, dtype=np.float64)
        if (
            values.shape != self._lower.shape
            or not np.isfinite(values).all()
            or np.any(values < self._lower)
            or np.any(values > self._upper)
        ):
            raise ValueError("target must contain seven finite positions within joint limits")
        with self._lock:
            self._target = values

    def clear(self) -> None:
        with self._lock:
            self._target = None
            self._command = None
            self._generation = None

    def on_preempted(self, by_task: str, joints: frozenset[str]) -> None:
        self.clear()

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        with self._lock:
            if self._target is None:
                return None
            generation = self._device.observed_generation()
            if self._generation is not None and generation != self._generation:
                self.clear()
                return None
            self._generation = generation
            if self._command is None:
                measured = np.asarray(
                    [state.joints.get_position(name) for name in PUBLIC_JOINTS], dtype=np.float64
                )
                if not np.isfinite(measured).all():
                    return None
                self._command = np.clip(measured, self._lower, self._upper)
            step = self._velocity * max(0.0, state.dt)
            self._command += np.clip(self._target - self._command, -step, step)
            return JointCommandOutput(
                joint_names=list(PUBLIC_JOINTS),
                positions=self._command.tolist(),
                mode=ControlMode.SERVO_POSITION,
            )


class YamHoldTask(OperatorHoldTask):
    """Reseed the existing hold from the same tick that observes a world reset."""

    def __init__(self, device: YamPositionDevice) -> None:
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

    def __init__(self, device: YamPositionDevice) -> None:
        self.device = device
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
        self._target = YamTargetTask(device)
        self._hold = YamHoldTask(device)
        self._gripper = GripperControlTask(
            "yam_gripper",
            GripperControlTaskConfig([PUBLIC_JOINTS[-1]], priority=20, hold_duration=0.2),
            [(0.0, 1.0)],
        )

    def start(self) -> None:
        try:
            self.device.connect()
            self._stack.callback(self.device.disconnect)
            self.device.activate()
            self.coordinator.add_hardware(
                self.device,
                HardwareComponent(
                    hardware_id="openyam",
                    hardware_type=HardwareType.MANIPULATOR,
                    joints=list(PUBLIC_JOINTS),
                ),
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
            self._target.clear()
            self._hold.request("manual", "Authority changed")
            if self._ik is not None:
                self._ik.set_estop(True)
                self._ik.set_estop(False)
            self._gripper.set_estop(True)
            self._gripper.set_estop(False)
            self._mode = mode
            self._grip = False
            return epoch

    @property
    def teleop_engaged(self) -> bool:
        with self._lock:
            return self._grip and time.monotonic() - self._buttons_at < 0.5

    def target(self, positions: list[float]) -> None:
        with self._lock:
            if self._mode == "hold" or self._grip:
                return
            self._hold.acknowledge()
            self._target.set_target(positions)

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
                    bindings=(TeleopHandBinding("right", "gripper_tip"),),
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
                stack.callback(module.right_controller_output.subscribe(self._pose))
                stack.callback(module.right_gripper_command.subscribe(self._gripper_target))
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
                self._ik.on_right_cartesian_command(pose, time.monotonic())

    def _buttons(self, buttons: Buttons) -> None:
        with self._lock:
            if self._await_release:
                if buttons.right_grip:
                    return
                self._await_release = False
            previous = self._grip
            pressed = buttons.right_grip
            callback = self.on_authority
            record_callback = self.on_record
            primary, secondary = buttons.right_primary, buttons.right_secondary
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
