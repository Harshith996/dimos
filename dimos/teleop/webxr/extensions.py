# Copyright 2025-2026 Dimensional Inc.
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

"""WebXR teleop module extensions and subclasses.

Available subclasses:
    - ArmTeleopModule: Raw arm poses with middle-finger grip deadman buttons
    - HandTeleopModule: Pinch-to-toggle arm teleop using WebXR hand tracking
    - TwistTeleopModule: Outputs Twist instead of PoseStamped
    - VideoArmTeleopModule: ArmTeleopModule + JPEG frames pushed to the headset over /ws
    - DualArmVideoTeleopModule: ArmTeleopModule + egocentric and wrist camera panels
    - Go2TeleopModule: Thumbstick → Twist velocity for the Go2 + camera over /ws
"""

import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.manipulation.manipulation_spec import ManipulationSpec
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.teleop.utils.teleop_transforms import webxr_point_to_robot
from dimos.teleop.webxr.body_tracking import ARM_BODY_JOINTS, ArmPosture, BodyTrackingSnapshot
from dimos.teleop.webxr.controller_types import (
    BUTTON_ALIASES,
    Buttons,
    Hand,
    WebXRControllerState,
)
from dimos.teleop.webxr.module import CameraPanel, WebXRTeleopConfig, WebXRTeleopModule
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _encode_panel_frame(panel: int, msg: Image, quality: int, max_width: int) -> bytes:
    """One headset video message: the panel index byte, then the JPEG."""
    if max_width > 0:
        msg, _ = msg.resize_to_fit(max_width, msg.height)
    return bytes([panel]) + msg.to_jpeg_bytes(quality=quality)


def _push_jpeg(
    module: WebXRTeleopModule,
    msg: Image,
    *,
    quality: int,
    panel: int = 0,
    max_width: int = 0,
) -> None:
    """JPEG-encode a camera frame and push it to connected camera clients.

    Called from auto-bound handlers on the module loop; their latest-only
    mailbox drops frames that arrive while an earlier one is still encoding.
    """
    # Skip the encode entirely if nobody is listening.
    if not module._has_clients():
        return
    try:
        data = _encode_panel_frame(panel, msg, quality, max_width)
    except Exception:
        logger.exception("Failed to encode camera frame", panel=panel)
        return
    module._broadcast_bytes(data)


class TwistTeleopConfig(WebXRTeleopConfig):
    """Configuration for TwistTeleopModule."""

    linear_scale: float = 1.0
    angular_scale: float = 1.0


# Example implementation to show how to extend WebXRTeleopModule for different teleop behaviors and outputs.
class TwistTeleopModule(WebXRTeleopModule):
    """WebXR teleop that outputs TwistStamped instead of PoseStamped.

    Config:
        - linear_scale: Scale factor for linear (position) values. Default 1.0.
        - angular_scale: Scale factor for angular (orientation) values. Default 1.0.

    Outputs:
        - left_twist: TwistStamped (linear + angular velocity)
        - right_twist: TwistStamped (linear + angular velocity)
        - buttons: Buttons (inherited)
    """

    config: TwistTeleopConfig

    left_twist: Out[TwistStamped]
    right_twist: Out[TwistStamped]

    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def stop(self) -> None:
        super().stop()

    def _publish_msg(self, hand: Hand, output_msg: PoseStamped) -> None:
        """Convert PoseStamped to TwistStamped, apply scaling, and publish."""
        twist = TwistStamped(
            ts=output_msg.ts,
            frame_id=output_msg.frame_id,
            linear=output_msg.position * self.config.linear_scale,
            angular=output_msg.orientation.to_euler() * self.config.angular_scale,
        )
        if hand == Hand.LEFT:
            self.left_twist.publish(twist)
        else:
            self.right_twist.publish(twist)


def arm_posture_from_body(snapshot: BodyTrackingSnapshot, hand: Hand) -> ArmPosture | None:
    """Pick one arm's joints out of a body snapshot and express them in the robot frame.

    Returns None while the body source is unavailable or the arm's elbow and
    wrist are not both resolved.
    """
    joints = snapshot.joints
    if not joints:
        return None
    side = hand.name.lower()
    names = ARM_BODY_JOINTS[side]

    def first(candidates: tuple[str, ...]) -> tuple[float, float, float] | None:
        for name in candidates:
            joint = joints.get(name)
            if joint is not None:
                return webxr_point_to_robot(joint.position)
        return None

    elbow = first(names["elbow"])
    wrist = first(names["wrist"])
    if elbow is None or wrist is None:
        return None
    return ArmPosture(
        ts=snapshot.capture_time_s,
        hand="left" if hand is Hand.LEFT else "right",
        wrist=wrist,
        elbow=elbow,
        shoulder=first(names["shoulder"]),
    )


class ArmTeleopModule(WebXRTeleopModule):
    """Quest arm input that leaves engagement to the coordinator task.

    Controller poses and buttons are published as raw operator input. The
    consuming TeleopIKTask owns the middle-finger grip deadman and reference
    capture, leaving the face buttons available for stack controls.

    Unlike the base module, this publishes absolute controller poses. The
    control task owns controller-to-robot reference capture so one task can
    establish a coherent reference for both arms.

    With body tracking enabled, each body snapshot also yields the operator's
    arm joints per hand so the task can steer the robot elbow.

    Outputs:
        - left_controller_output: PoseStamped (inherited)
        - right_controller_output: PoseStamped (inherited)
        - teleop_buttons: Buttons (inherited)
        - left_gripper_command: Float32 normalized opening
        - right_gripper_command: Float32 normalized opening
        - left_arm_posture: ArmPosture operator arm joints in the robot frame
        - right_arm_posture: ArmPosture operator arm joints in the robot frame
    """

    left_gripper_command: Out[Float32]
    right_gripper_command: Out[Float32]
    left_arm_posture: Out[ArmPosture]
    right_arm_posture: Out[ArmPosture]

    _manipulation: ManipulationSpec | None = None

    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_buttons_pressed(self, pressed: Buttons) -> None:
        """Move to the zero joint pose, off the control loop, when zero_button is pressed."""
        button = self.config.zero_button
        if button is not None and getattr(pressed, BUTTON_ALIASES.get(button, button)):
            threading.Thread(target=self._move_to_zero, daemon=True).start()

    def _move_to_zero(self) -> None:
        if self._manipulation is None:
            return
        reset = self._manipulation.reset()
        if not reset.succeeded:
            logger.error("Could not reset manipulation before moving to zero", error=reset.message)
            return
        targets = {
            group.id: JointState(
                name=list(group.joint_names), position=[0.0] * len(group.joint_names)
            )
            for group in self._manipulation.list_planning_groups()
        }
        plan = self._manipulation.plan_to_joints(targets)
        if plan.plan is not None:
            self._manipulation.execute(blocking=False, plan_id=plan.plan.plan_id)

    def _on_body_snapshot(self, snapshot: BodyTrackingSnapshot) -> None:
        super()._on_body_snapshot(snapshot)
        outputs = {Hand.LEFT: self.left_arm_posture, Hand.RIGHT: self.right_arm_posture}
        for hand, output in outputs.items():
            posture = arm_posture_from_body(snapshot, hand)
            if posture is not None:
                output.publish(posture)

    def _get_output_pose(self, hand: Hand) -> PoseStamped | None:
        """Return the current absolute controller pose."""
        return self._current_poses.get(hand)

    def _handle_engage(self) -> None:
        """Leave arm engagement to TeleopIKTask."""

    def _should_publish(self, hand: Hand) -> bool:
        """Publish every fresh absolute controller pose."""
        return self._controllers.get(hand) is not None

    def _publish_button_state(
        self,
        left: WebXRControllerState | None,
        right: WebXRControllerState | None,
    ) -> None:
        """Publish Buttons with analog triggers packed into bits 16-29."""
        buttons = Buttons.from_controllers(left, right)
        buttons.pack_analog_triggers(
            left=left.trigger if left is not None else 0.0,
            right=right.trigger if right is not None else 0.0,
        )
        self._publish_buttons(buttons)
        self._publish_gripper_commands(left, right, buttons)

    def _publish_gripper_commands(
        self,
        left: WebXRControllerState | None,
        right: WebXRControllerState | None,
        buttons: Buttons,
    ) -> None:
        """Publish normalized opening while the classified deadman is held."""
        controllers = {Hand.LEFT: left, Hand.RIGHT: right}
        outputs = {
            Hand.LEFT: self.left_gripper_command,
            Hand.RIGHT: self.right_gripper_command,
        }
        for hand, controller in controllers.items():
            if controller is None or not self._gripper_is_enabled(hand, buttons):
                continue
            outputs[hand].publish(Float32(data=1.0 - float(controller.trigger)))

    def _gripper_is_enabled(self, hand: Hand, buttons: Buttons) -> bool:
        """Use the same classified grip bit consumed by TeleopIKTask."""
        return buttons.left_grip if hand == Hand.LEFT else buttons.right_grip


class HandTeleopModule(ArmTeleopModule):
    """WebXR hand teleop with pinch-to-toggle engage and task name routing.

    A thumb-and-index pinch is sent as the primary button by the browser. Each
    pinch edge toggles that hand between engaged and disengaged, so movement
    continues after releasing the pinch.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._primary_was_pressed: dict[Hand, bool] = {Hand.LEFT: False, Hand.RIGHT: False}

    def _handle_engage(self) -> None:
        """Toggle each hand on the rising edge of its primary button."""
        for hand in Hand:
            controller = self._controllers.get(hand)
            is_pressed = controller is not None and controller.primary
            if is_pressed and not self._primary_was_pressed[hand]:
                if self._is_engaged[hand]:
                    self._disengage(hand)
                else:
                    self._engage(hand)
            self._primary_was_pressed[hand] = is_pressed

    def _should_publish(self, hand: Hand) -> bool:
        return self._is_engaged[hand]

    def _gripper_is_enabled(self, hand: Hand, buttons: Buttons) -> bool:
        return self._is_engaged[hand]

    def _publish_button_state(
        self,
        left: WebXRControllerState | None,
        right: WebXRControllerState | None,
    ) -> None:
        """Keep downstream press-and-hold teleop tasks engaged between pinches."""
        buttons = Buttons.from_controllers(left, right)
        buttons.pack_analog_triggers(
            left=left.trigger if left is not None else 0.0,
            right=right.trigger if right is not None else 0.0,
        )
        buttons.left_grip = self._is_engaged[Hand.LEFT]
        buttons.right_grip = self._is_engaged[Hand.RIGHT]
        self._publish_buttons(buttons)
        self._publish_gripper_commands(left, right, buttons)


class VideoArmTeleopConfig(WebXRTeleopConfig):
    """Configuration for VideoArmTeleopModule."""

    video_jpeg_quality: int = 70


class VideoArmTeleopModule(ArmTeleopModule):
    """ArmTeleopModule + camera frames pushed to the headset as JPEG over /ws.

    Subscribes to color_image, JPEG-encodes each frame, and broadcasts raw
    JPEG bytes to every connected /ws client as a binary message. The client
    decodes via createObjectURL and uploads to a WebGL texture.

    Inputs:
        - color_image: In[Image] (required — wire to a camera output)

    Outputs:
        - left_controller_output: PoseStamped (inherited)
        - right_controller_output: PoseStamped (inherited)
        - buttons: Buttons (inherited)
    """

    config: VideoArmTeleopConfig
    camera_panels = (CameraPanel("color_image", "center"),)

    color_image: In[Image]

    async def handle_color_image(self, msg: Image) -> None:
        _push_jpeg(self, msg, quality=self.config.video_jpeg_quality)


class DualArmVideoTeleopConfig(WebXRTeleopConfig):
    """Configuration for DualArmVideoTeleopModule."""

    video_jpeg_quality: int = 70
    # Frames wider than this are downscaled before encoding; 0 sends the source size.
    video_max_width: int = 640
    # Per-panel cap on frames pushed to the headset; 0 pushes every frame.
    video_max_fps: float = 0.0


_EGOCENTRIC_PANEL, _LEFT_WRIST_PANEL, _RIGHT_WRIST_PANEL = range(3)


class DualArmVideoTeleopModule(ArmTeleopModule):
    """ArmTeleopModule + egocentric and wrist camera panels in the headset.

    Each camera input is JPEG-encoded and pushed over /ws tagged with its
    panel index. The client floats the egocentric view in a large center panel
    above the workspace and each wrist view in a smaller panel on that arm's
    side, turned toward the operator.

    Inputs:
        - egocentric_image: In[Image] (scene camera, center panel)
        - left_wrist_image: In[Image] (left wrist camera, left panel)
        - right_wrist_image: In[Image] (right wrist camera, right panel)

    Outputs: as ArmTeleopModule.
    """

    config: DualArmVideoTeleopConfig
    camera_panels = (
        CameraPanel("egocentric", "center"),
        CameraPanel("left_wrist", "left"),
        CameraPanel("right_wrist", "right"),
    )

    egocentric_image: In[Image]
    left_wrist_image: In[Image]
    right_wrist_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Only touched from the module loop, where the auto-bound handlers run.
        self._panel_next_push_at: dict[int, float] = {}

    async def handle_egocentric_image(self, msg: Image) -> None:
        self._push_panel(_EGOCENTRIC_PANEL, msg)

    async def handle_left_wrist_image(self, msg: Image) -> None:
        self._push_panel(_LEFT_WRIST_PANEL, msg)

    async def handle_right_wrist_image(self, msg: Image) -> None:
        self._push_panel(_RIGHT_WRIST_PANEL, msg)

    def _push_panel(self, panel: int, msg: Image) -> None:
        """Apply the per-panel frame-rate cap, then encode and push the frame."""
        max_fps = self.config.video_max_fps
        if max_fps > 0:
            now = time.monotonic()
            if now < self._panel_next_push_at.get(panel, 0.0):
                return
            self._panel_next_push_at[panel] = now + 1.0 / max_fps
        _push_jpeg(
            self,
            msg,
            quality=self.config.video_jpeg_quality,
            panel=panel,
            max_width=self.config.video_max_width,
        )


class Go2TeleopConfig(WebXRTeleopConfig):
    """Configuration for Go2TeleopModule."""

    linear_speed: float = 0.5  # m/s at full stick deflection
    angular_speed: float = 0.8  # rad/s at full stick deflection
    deadzone: float = 0.1
    video_jpeg_quality: int = 70


class Go2TeleopModule(WebXRTeleopModule):
    """WebXR teleop for the Unitree Go2: thumbstick driving + camera in the headset.

    Velocity is derived from the controller thumbsticks as each Joy message
    arrives (left stick → forward/strafe, right stick → yaw) and published on
    cmd_vel for GO2Connection.move. The Go2 camera (color_image) is JPEG-encoded
    and pushed to the headset over /ws. A deadzone suppresses stick drift.

    Inputs:
        - color_image: In[Image] (wire to the Go2 camera output)

    Outputs:
        - cmd_vel: Twist (base velocity command)
    """

    config: Go2TeleopConfig
    camera_panels = (CameraPanel("color_image", "center"),)

    color_image: In[Image]
    cmd_vel: Out[Twist]

    def _publish_safe_command(self) -> None:
        self.cmd_vel.publish(Twist.zero())

    def _deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.config.deadzone else v

    def _on_joy_bytes(self, data: bytes) -> bool:
        try:
            valid = super()._on_joy_bytes(data)
        except ValueError:
            self._publish_safe_command()
            raise
        if not valid:
            self._publish_safe_command()
            return False
        with self._lock:
            left = self._controllers.get(Hand.LEFT)
            right = self._controllers.get(Hand.RIGHT)
        twist = Twist()
        twist.linear = Vector3(0.0, 0.0, 0.0)
        twist.angular = Vector3(0.0, 0.0, 0.0)
        if left is not None:
            twist.linear.x = -self._deadzone(left.thumbstick.y) * self.config.linear_speed
            twist.linear.y = -self._deadzone(left.thumbstick.x) * self.config.linear_speed
        if right is not None:
            twist.angular.z = -self._deadzone(right.thumbstick.x) * self.config.angular_speed
        self.cmd_vel.publish(twist)
        return True

    async def handle_color_image(self, msg: Image) -> None:
        _push_jpeg(self, msg, quality=self.config.video_jpeg_quality)
