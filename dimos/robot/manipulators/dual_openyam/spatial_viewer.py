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

"""Sensor-derived immersive view on the existing WebXR server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from typing import Any, Literal

from fastapi import HTTPException, WebSocket
from fastapi.responses import HTMLResponse, Response
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.imitation.collection.spec import EpisodeCollectionSpec
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.perception.experimental.object import Object
from dimos.protocol.tf.tf import MultiTBuffer
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS
from dimos.robot.manipulators.dual_openyam.spatial_model import (
    KnownUrdfYamModel,
    KnownYamModel,
    artifact_directory,
)
from dimos.robot.manipulators.openyam.simulation import load_yam_artifact
from dimos.teleop.webxr.controller_types import Buttons, Hand, WebXRControllerState
from dimos.teleop.webxr.extensions import ArmTeleopModule
from dimos.teleop.webxr.module import STATIC_DIR, WebXRTeleopConfig
from dimos.teleop.webxr.rgbd import RGBDEncoder, SurfaceFrame
from dimos.types.timestamped import align_timestamped
from dimos.utils.reactive import backpressure


class SpatialHand(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    pinched: bool
    action: Literal["move", "opening"] | None = None
    arm: Literal["left", "right"] | None = None

    @model_validator(mode="after")
    def tool_target(self) -> SpatialHand:
        if (self.action is None) != (self.arm is None):
            raise ValueError("a tool action requires an explicit robot arm")
        return self

    @field_validator("orientation")
    @classmethod
    def unit_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if abs(float(np.linalg.norm(value)) - 1) > 0.01:
            raise ValueError("orientation must be a unit quaternion")
        return value


class SpatialInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["spatial_input"]
    sequence: int = Field(ge=0)
    observed_at: float
    hands: dict[Literal["left", "right"], SpatialHand]
    focus: tuple[float, float, float] | None = None


class CollectionAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["record", "save", "discard"]
    task: str = Field(default="OpenYAM manipulation demonstration", min_length=1, max_length=256)


class SpatialViewerConfig(WebXRTeleopConfig):
    artifact_directory: Path = Field(default_factory=artifact_directory)
    robot_model: RobotModelConfig | None = None
    input_timeout_s: float = 0.3
    observation_timeout_s: float = Field(default=0.5, gt=0)
    max_pixels: int = Field(default=60000, ge=4, le=200000)
    max_depth_m: float = Field(default=2.0, gt=0, le=60)
    depth_unit_m: float = Field(default=0.001, gt=0)
    sync_tolerance_s: float = Field(default=0.03, gt=0, le=0.1)
    max_edge_m: float = Field(default=0.035, gt=0, le=0.1)
    opening_handle_offset: tuple[float, float, float] = (0.0, 0.14, 0.0)
    opening_handle_travel_m: float = Field(default=0.12, gt=0)


class OpenYamSpatialTeleop(ArmTeleopModule):
    """No sim world access: camera observations and measured joint state are the inputs.

    `objects` accepts the same world-frame detections as ManipulationModule.
    `perception_focus` names an operator-selected observed surface for a perception
    consumer. This blueprint does not silently load a recognizer or collision map.
    """

    config: SpatialViewerConfig
    _collection: EpisodeCollectionSpec
    coordinator_joint_state: In[JointState]
    color_image: In[Image]
    depth_image: In[Image]
    camera_info: In[CameraInfo]
    left_wrist: In[Image]
    right_wrist: In[Image]
    tf: In[TFMessage]
    objects: In[list[Object]]
    perception_focus: Out[PointStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._robot: KnownYamModel | KnownUrdfYamModel | None = None
        self._feedback: dict[str, Any] = {}
        self._surface: SurfaceFrame | None = None
        self._camera_info: CameraInfo | None = None
        self._observation_error = "Waiting for registered RGB-D and camera calibration"
        self._encoder = RGBDEncoder(
            **{
                name: getattr(self.config, name)
                for name in (
                    "max_pixels",
                    "max_depth_m",
                    "depth_unit_m",
                    "sync_tolerance_s",
                    "max_edge_m",
                )
            }
        )
        self._images: dict[str, tuple[float, bytes]] = {}
        self._objects: list[dict[str, Any]] = []
        self._objects_at = 0.0
        self._transforms = MultiTBuffer()
        self._sequence = -1
        self._released = {hand: False for hand in Hand}
        self._claims: dict[Hand, tuple[Hand, str]] = {}
        self._last_input_update: float | None = None
        self._opening_refs: dict[Hand, tuple[np.ndarray, np.ndarray, float]] = {}

    @rpc
    def build(self) -> None:
        self._robot = (
            KnownUrdfYamModel(self.config.robot_model)
            if self.config.robot_model
            else KnownYamModel(load_yam_artifact(self.config.artifact_directory))
        )
        super().build()
        subscriptions: list[tuple[In[Any], Callable[[Any], None]]] = [
            (self.coordinator_joint_state, self._on_feedback),
            (self.camera_info, self._on_camera_info),
            (self.tf, self._transforms.receive_tfmessage),
            (self.left_wrist, lambda image: self._on_image("left", image)),
            (self.right_wrist, lambda image: self._on_image("right", image)),
            (self.color_image, lambda image: self._on_image("workspace", image)),
            (self.objects, self._on_objects),
        ]
        for port, callback in subscriptions:
            self.register_disposable(Disposable(port.subscribe(callback)))
        frames = align_timestamped(
            self.color_image.observable(),
            self.depth_image.observable(),
            buffer_size=0.3,
            match_tolerance=self.config.sync_tolerance_s,
        )
        self.register_disposable(backpressure(frames).subscribe(self._on_rgbd))

    def _setup_routes(self) -> None:
        super()._setup_routes()
        assert self._web_server is not None
        app = self._web_server.app

        @app.get("/spatial", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse((STATIC_DIR / "spatial.html").read_text())

        @app.get("/spatial/model")
        def model() -> dict[str, Any]:
            assert self._robot is not None
            return self._robot.geometry

        @app.get("/spatial/state")
        def state() -> dict[str, Any]:
            with self._lock:
                now = time.time()
                return {
                    **self._feedback,
                    "server_time": now,
                    "surface_ts": self._surface.ts if self._surface else 0.0,
                    "observation_error": self._observation_error,
                    "opening_handle_offset": self.config.opening_handle_offset,
                    "opening_handle_travel_m": self.config.opening_handle_travel_m,
                    "actions": {
                        arm.name.lower(): {
                            action: hand.name.lower()
                            for hand, (target, action) in self._claims.items()
                            if target == arm
                        }
                        for arm in Hand
                    },
                    "objects": self._objects if now - self._objects_at < 2 else [],
                    "objects_active": now - self._objects_at < 2,
                    "observation_timeout_s": self.config.observation_timeout_s,
                }

        @app.get("/spatial/surface")
        def surface(after: float = 0) -> Response:
            with self._lock:
                frame = self._surface
            if frame is None:
                raise HTTPException(503, self._observation_error)
            if frame.ts <= after:
                return Response(status_code=204, headers={"Cache-Control": "no-store"})
            return Response(
                frame.payload,
                media_type="application/octet-stream",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/spatial/wrist/{side}")
        @app.get("/spatial/camera/{side}")
        def camera(side: str) -> Response:
            with self._lock:
                image = self._images.get(side)
            if image is None or time.time() - image[0] > self.config.observation_timeout_s:
                raise HTTPException(503, "Camera unavailable")
            return Response(
                image[1],
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store", "X-Frame-Timestamp": str(image[0])},
            )

        @app.get("/spatial/collection")
        def collection() -> dict[str, Any]:
            return self._collection.get_status()

        @app.post("/spatial/collection")
        def collection_action(command: CollectionAction) -> dict[str, Any]:
            try:
                if command.action == "record":
                    return self._collection.record(command.task)
                return self._collection.finish(command.action == "save")
            except (RuntimeError, ValueError) as error:
                raise HTTPException(409, str(error)) from error

    @rpc
    def get_web_views(self) -> list[dict[str, Any]]:
        return [
            {
                "title": "Spatial workstation",
                "scheme": "https",
                "port": self.config.server_port,
                "path": "/spatial",
                "embed": True,
                "description": "Pinch the round tool handle to move the arm. Pinch and slide its opening control to adjust the jaws. Release to hold.",
            }
        ]

    def _on_feedback(self, message: JointState) -> None:
        if len(message.name) != len(message.position) or not np.isfinite(message.ts):
            return
        positions = dict(zip(message.name, message.position, strict=True))
        if (
            not set(DUAL_OPENYAM_JOINTS) <= positions.keys()
            or not np.isfinite(list(positions.values())).all()
        ):
            return
        with self._lock:
            assert self._robot is not None
            self._feedback = {"ts": message.ts, "arms": self._robot.feedback(positions)}

    def _on_image(self, side: str, message: Image) -> None:
        data = message.to_jpeg_bytes(quality=75)
        with self._lock:
            self._images[side] = (message.ts, data)

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _on_rgbd(self, frames: tuple[Image, ...]) -> None:
        color, depth = frames
        info = self._camera_info
        if (
            info is None
            or time.time() - min(color.ts, depth.ts) > self.config.observation_timeout_s
        ):
            return
        transform = self._transforms.get("world", depth.frame_id, depth.ts, 0.1, warn=False)
        if transform is None:
            self._observation_error = "Camera-to-world calibration unavailable"
            return
        try:
            frame = self._encoder.encode(color, depth, info, transform)
        except ValueError as error:
            self._observation_error = str(error)
            return
        with self._lock:
            if self._surface is None or frame.ts > self._surface.ts:
                self._surface = frame
                self._observation_error = ""

    def _on_objects(self, objects: list[Object]) -> None:
        # Match ManipulationModule's world-frame object contract. Do not guess a frame.
        result = [
            {
                "id": obj.object_id,
                "label": obj.name,
                "confidence": obj.confidence,
                "position": obj.pose.position.to_list(),
                "size": obj.size.to_list(),
                "orientation": obj.pose.orientation.to_list(),
            }
            for obj in objects
            if obj.pose.frame_id == "world"
        ]
        with self._lock:
            self._objects, self._objects_at = result, time.time()

    def _dispatch_binary_message(self, data: bytes) -> bool:
        # This viewer's world-space input must not take the legacy Quest conversion path.
        return False

    def _client_connected(self, ws: WebSocket) -> bool:
        accepted = super()._client_connected(ws)
        if accepted:
            self._sequence = -1
        return accepted

    def _dispatch_text_message(self, payload: str) -> bool:
        try:
            message = SpatialInput.model_validate_json(payload)
        except ValidationError:
            self._reset_controller_state()
            return False
        with self._lock:
            now = time.time()
            fresh = (
                0 <= now - message.observed_at <= self.config.observation_timeout_s
                and 0 <= now - self._feedback.get("ts", 0) <= self.config.observation_timeout_s
                and self._surface is not None
                and 0 <= now - self._surface.ts <= self.config.observation_timeout_s
            )
            if message.sequence <= self._sequence or not fresh:
                self._reset_controller_state()
                return False
            self._sequence = message.sequence
            # Claims belong to human hands; downstream controller slots belong to
            # robot arms. Reserve previous owners through this release frame.
            reserved = set(self._claims.values())
            previous_movers = {arm: self._owner(arm, "move") for arm in Hand}
            for hand in Hand:
                sample = message.hands.get("left" if hand == Hand.LEFT else "right")
                if sample is None or not sample.pinched:
                    self._released[hand] = sample is not None
                    self._claims.pop(hand, None)
                    self._opening_refs.pop(hand, None)
                    continue
                target = Hand.LEFT if sample.arm == "left" else Hand.RIGHT
                claim = (target, sample.action) if sample.action is not None else None
                if hand in self._claims:
                    if claim != self._claims[hand]:
                        self._claims.pop(hand)
                        self._opening_refs.pop(hand, None)
                    continue
                if not self._released[hand]:
                    continue
                self._released[hand] = False
                if claim is None or claim in reserved:
                    continue
                arm = self._feedback["arms"][sample.arm]
                position = np.asarray(sample.position)
                opening = float(arm["opening"])
                rotation = Rotation.from_quat(arm["orientation"])
                axis = rotation.apply([1.0, 0.0, 0.0])
                knob = (
                    np.asarray(arm["grasp"])
                    + rotation.apply(self.config.opening_handle_offset)
                    + axis * (opening - 0.5) * self.config.opening_handle_travel_m
                )
                if sample.action == "opening" and np.linalg.norm(position - knob) < 0.055:
                    self._claims[hand] = (target, "opening")
                    self._opening_refs[hand] = (position, axis, opening)
                elif sample.action == "move" and np.linalg.norm(position - arm["grasp"]) < 0.12:
                    self._claims[hand] = (target, "move")
                if hand in self._claims:
                    reserved.add(self._claims[hand])

            # Publish releases immediately, even if release/regrab packets arrive
            # between control ticks. IK must recapture for a different human hand.
            released = False
            for arm in Hand:
                if (
                    previous_movers[arm] is not None
                    and self._owner(arm, "move") != previous_movers[arm]
                ):
                    controller = self._controllers[arm]
                    if controller is not None:
                        controller.grip = 0.0
                    released = True
            if released:
                self._publish_buttons(
                    Buttons.from_controllers(
                        self._controllers[Hand.LEFT], self._controllers[Hand.RIGHT]
                    )
                )

            self._last_input_update = time.monotonic()
            for arm in Hand:
                side = "left" if arm == Hand.LEFT else "right"
                measured = self._feedback["arms"][side]
                mover, opener = self._owner(arm, "move"), self._owner(arm, "opening")
                if mover is None and opener is None:
                    self._controllers[arm] = None
                    self._current_poses[arm] = None
                    self._last_pose_update[arm] = self._last_controller_update[arm] = None
                    continue
                pose = (
                    message.hands["left" if mover == Hand.LEFT else "right"]
                    if mover is not None
                    else None
                )
                opening = float(measured["opening"])
                if opener is not None:
                    sample = message.hands["left" if opener == Hand.LEFT else "right"]
                    origin, axis, initial = self._opening_refs[opener]
                    opening = float(
                        np.clip(
                            initial
                            + np.dot(np.asarray(sample.position) - origin, axis)
                            / self.config.opening_handle_travel_m,
                            0,
                            1,
                        )
                    )
                self._current_poses[arm] = PoseStamped(
                    position=pose.position if pose else measured["grasp"],
                    orientation=pose.orientation if pose else measured["orientation"],
                    frame_id="world",
                    ts=now,
                )
                self._controllers[arm] = WebXRControllerState(
                    is_left=arm == Hand.LEFT,
                    grip=float(mover is not None),
                    trigger=1 - opening,
                )
                self._last_pose_update[arm] = self._last_controller_update[arm] = (
                    self._last_input_update
                )
            if message.focus is not None:
                assert self._surface is not None
                points = self._surface.points
                if len(points) and np.min(np.linalg.norm(points - message.focus, axis=1)) < 0.04:
                    self.perception_focus.publish(
                        PointStamped(*message.focus, frame_id="world", ts=now)
                    )
            return True

    def _owner(self, arm: Hand, action: str) -> Hand | None:
        return next((hand for hand, claim in self._claims.items() if claim == (arm, action)), None)

    def _gripper_is_enabled(self, hand: Hand, buttons: Buttons) -> bool:
        # Either human hand may own this robot arm's independent opening control.
        return self._owner(hand, "opening") is not None

    def _reset_controller_state(self) -> None:
        with self._lock:
            self._released = {hand: False for hand in Hand}
            self._claims.clear()
            self._opening_refs.clear()
            self._last_input_update = None
            super()._reset_controller_state()

    def _expire_stale_state(self, now: float) -> None:
        # Each packet is a complete snapshot; omitted hands release immediately.
        if (
            self._last_input_update is not None
            and now - self._last_input_update > self.config.input_timeout_s
        ) or (
            time.time() - min(self._feedback.get("ts", 0), self._surface.ts if self._surface else 0)
            > self.config.observation_timeout_s
        ):
            self._reset_controller_state()
        super()._expire_stale_state(now)
