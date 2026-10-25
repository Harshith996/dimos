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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.perception.experimental.object import Object
from dimos.protocol.tf.tf import MultiTBuffer
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS
from dimos.robot.manipulators.dual_openyam.spatial_model import KnownYamModel, artifact_directory
from dimos.robot.manipulators.openyam.simulation import load_yam_artifact
from dimos.teleop.webxr.controller_types import Hand, WebXRControllerState
from dimos.teleop.webxr.extensions import ArmTeleopModule
from dimos.teleop.webxr.module import STATIC_DIR, WebXRTeleopConfig


class SpatialHand(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    grab: bool
    opening: float = Field(ge=0, le=1)

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


class SpatialViewerConfig(WebXRTeleopConfig):
    artifact_directory: Path = Field(default_factory=artifact_directory)
    input_timeout_s: float = 0.3
    observation_timeout_s: float = Field(default=0.5, gt=0)
    max_points: int = Field(default=12000, gt=0, le=50000)
    max_depth_m: float = Field(default=2.0, gt=0)


class OpenYamSpatialTeleop(ArmTeleopModule):
    """No sim world access: camera observations and measured joint state are the inputs.

    `objects` accepts the same world-frame detections as ManipulationModule.
    `perception_focus` names an operator-selected observed surface for a perception
    consumer. This blueprint does not silently load a recognizer or collision map.
    """

    config: SpatialViewerConfig
    coordinator_joint_state: In[JointState]
    pointcloud: In[PointCloud2]
    left_wrist: In[Image]
    right_wrist: In[Image]
    tf: In[TFMessage]
    objects: In[list[Object]]
    perception_focus: Out[PointStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._robot: KnownYamModel | None = None
        self._feedback: dict[str, Any] = {}
        self._cloud: dict[str, Any] = {"ts": 0.0, "positions": [], "colors": []}
        self._images: dict[str, tuple[float, bytes]] = {}
        self._objects: list[dict[str, Any]] = []
        self._objects_at = 0.0
        self._transforms = MultiTBuffer()
        self._sequence = -1
        self._released = {hand: False for hand in Hand}
        self._grabbing = {hand: False for hand in Hand}

    @rpc
    def build(self) -> None:
        self._robot = KnownYamModel(load_yam_artifact(self.config.artifact_directory))
        super().build()
        subscriptions: list[tuple[In[Any], Callable[[Any], None]]] = [
            (self.coordinator_joint_state, self._on_feedback),
            (self.pointcloud, self._on_cloud),
            (self.tf, self._transforms.receive_tfmessage),
            (self.left_wrist, lambda image: self._on_image("left", image)),
            (self.right_wrist, lambda image: self._on_image("right", image)),
            (self.objects, self._on_objects),
        ]
        for port, callback in subscriptions:
            self.register_disposable(Disposable(port.subscribe(callback)))

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
                    "cloud_ts": self._cloud["ts"],
                    "objects": self._objects if now - self._objects_at < 2 else [],
                    "objects_active": now - self._objects_at < 2,
                    "observation_timeout_s": self.config.observation_timeout_s,
                }

        @app.get("/spatial/cloud")
        def cloud() -> dict[str, Any]:
            with self._lock:
                return self._cloud

        @app.get("/spatial/wrist/{side}")
        def wrist(side: str) -> Response:
            with self._lock:
                image = self._images.get(side)
            if image is None or time.time() - image[0] > self.config.observation_timeout_s:
                raise HTTPException(503, "Camera unavailable")
            return Response(
                image[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"}
            )

    @rpc
    def get_web_views(self) -> list[dict[str, Any]]:
        return [
            {
                "title": "Spatial workstation",
                "scheme": "https",
                "port": self.config.server_port,
                "path": "/spatial",
                "embed": True,
                "description": "Pinch a gripper handle with thumb and index finger to move that arm. Release to hold. Thumb and middle finger close the gripper.",
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

    def _on_cloud(self, message: PointCloud2) -> None:
        # Camera optical coordinates require a measured/calibrated camera transform.
        transform = self._transforms.get("world", message.frame_id, message.ts, 0.1, warn=False)
        if transform is None:
            return
        points, colors = message.as_numpy()
        valid = (
            np.isfinite(points).all(axis=1)
            & (points[:, 2] > 0)
            & (points[:, 2] < self.config.max_depth_m)
        )
        points = points[valid]
        colors = colors[valid] if colors is not None else np.full_like(points, 0.65)
        step = max(1, int(np.ceil(len(points) / self.config.max_points)))
        points, colors = points[::step], colors[::step]
        matrix = transform.to_matrix()
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        packet = {
            "ts": message.ts,
            "frame": "world",
            "positions": np.round(points, 4).ravel().tolist(),
            "colors": np.round(np.nan_to_num(colors).clip(0, 1), 3).ravel().tolist(),
        }
        with self._lock:
            self._cloud = packet

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
                and 0 <= now - self._cloud["ts"] <= self.config.observation_timeout_s
            )
            if message.sequence <= self._sequence or not fresh:
                self._reset_controller_state()
                return False
            self._sequence = message.sequence
            for hand in Hand:
                side: Literal["left", "right"] = "left" if hand == Hand.LEFT else "right"
                sample = message.hands.get(side)
                if sample is None:
                    self._controllers[hand] = None
                    self._current_poses[hand] = None
                    self._released[hand] = self._grabbing[hand] = False
                    continue
                if not sample.grab:
                    self._released[hand] = True
                    self._grabbing[hand] = False
                elif self._released[hand] and not self._grabbing[hand]:
                    grasp = self._feedback["arms"][side]["grasp"]
                    self._grabbing[hand] = bool(
                        np.linalg.norm(np.array(sample.position) - grasp) < 0.12
                    )
                    self._released[hand] = False
                self._current_poses[hand] = PoseStamped(
                    position=sample.position,
                    orientation=sample.orientation,
                    frame_id="world",
                    ts=now,
                )
                self._controllers[hand] = WebXRControllerState(
                    is_left=hand == Hand.LEFT,
                    grip=float(self._grabbing[hand]),
                    trigger=1 - sample.opening,
                )
                self._last_pose_update[hand] = self._last_controller_update[hand] = time.monotonic()
            if message.focus is not None:
                points = np.asarray(self._cloud["positions"]).reshape(-1, 3)
                if len(points) and np.min(np.linalg.norm(points - message.focus, axis=1)) < 0.04:
                    self.perception_focus.publish(
                        PointStamped(*message.focus, frame_id="world", ts=now)
                    )
            return True

    def _reset_controller_state(self) -> None:
        with self._lock:
            self._released = {hand: False for hand in Hand}
            self._grabbing = {hand: False for hand in Hand}
            super()._reset_controller_state()

    def _expire_stale_state(self, now: float) -> None:
        for hand in Hand:
            updated = self._last_pose_update[hand]
            if updated is None or now - updated > self.config.input_timeout_s:
                self._released[hand] = self._grabbing[hand] = False
        if (
            time.time() - min(self._feedback.get("ts", 0), self._cloud["ts"])
            > self.config.observation_timeout_s
        ):
            self._reset_controller_state()
        super()._expire_stale_state(now)
