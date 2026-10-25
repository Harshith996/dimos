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

"""Episode collection beside dual-arm control, with one owner for each camera."""

from functools import partial
import json
import math
from pathlib import Path
from threading import RLock
import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field
import reactivex as rx
from reactivex.disposable import Disposable

from dimos.constants import STATE_DIR
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.camera.webcam import Webcam, discover_webcams
from dimos.imitation.collection.capture import EpisodeCapture
from dimos.imitation.collection.episode import EpisodeEvent, EpisodeStatus
from dimos.imitation.collection.profile import CollectionProfile
from dimos.imitation.dataprep.schema import FeatureSpec, SyncConfig
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS
from dimos.teleop.webxr.controller_types import Buttons


class DualOpenYamCollectionConfig(ModuleConfig):
    directory: Path = STATE_DIR / "recordings" / "dual-openyam"
    camera_ids: tuple[str, ...] = ()
    camera_source: Literal["devices", "streams"] = "devices"
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    fps: int = Field(default=15, gt=0)
    task: str = "OpenYAM manipulation demonstration"


class DualOpenYamCollection(Module):
    dedicated_worker = True
    config: DualOpenYamCollectionConfig
    coordinator_joint_state: In[JointState]
    coordinator_joint_command: In[JointState]
    button_pressed: In[Buttons]
    left_image: In[Image]
    right_image: In[Image]
    camera1: Out[Image]
    camera2: Out[Image]
    status: Out[String]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = RLock()
        self._capture: EpisodeCapture | None = None
        self._metadata: dict[str, Any] = {}
        self._seen: dict[str, float] = {}
        self._accepted: dict[str, tuple[float, float]] = {}
        self._camera_ids: tuple[str, ...] = self.config.camera_ids
        self._saved = 0
        self._discarded = 0
        self._started = 0.0
        self._event: EpisodeEvent = "init"
        self._task = self.config.task
        self._error: str | None = None

    def profile(self) -> CollectionProfile:
        state = FeatureSpec(
            stream="state",
            message_type=JointState,
            field="position",
            dtype="float32",
            shape=(len(DUAL_OPENYAM_JOINTS),),
            names=DUAL_OPENYAM_JOINTS,
        )
        return CollectionProfile(
            name="dual-openyam",
            robot_type="dual_openyam",
            observations={
                "observation.state": state,
                **{
                    f"observation.images.{name}": FeatureSpec(
                        stream=name,
                        message_type=Image,
                        dtype="video",
                        shape=(self.config.height, self.config.width, 3),
                        names=["height", "width", "channels"],
                    )
                    for name in ("camera1", "camera2")
                },
            },
            actions={
                "action": state.model_copy(
                    update={
                        "stream": "action",
                        "source_kind": "joint_position_updates",
                    },
                    deep=True,
                )
            },
            sync=SyncConfig(
                anchor="observation.images.camera1", rate_hz=self.config.fps, tolerance_ms=75
            ),
        )

    @rpc
    def start(self) -> None:
        super().start()
        self.config.directory.mkdir(parents=True, exist_ok=True)
        self.register_disposable(
            Disposable(self.coordinator_joint_state.subscribe(partial(self._joints, "state")))
        )
        self.register_disposable(
            Disposable(self.coordinator_joint_command.subscribe(partial(self._joints, "action")))
        )
        self.register_disposable(Disposable(self.button_pressed.subscribe(self._buttons)))
        if self.config.camera_source == "streams":
            for port, name in ((self.left_image, "camera1"), (self.right_image, "camera2")):
                self.register_disposable(Disposable(port.subscribe(partial(self._image, name))))
        else:
            self._open_cameras()
        self.register_disposable(rx.interval(0.2).subscribe(lambda _: self._health()))
        self._publish_status()

    def _open_cameras(self) -> None:
        self._camera_ids = self.config.camera_ids or tuple(
            sorted(camera.id for camera in discover_webcams() if "usb" in camera.name.lower())
        )
        if len(self._camera_ids) != 2 or len(set(self._camera_ids)) != 2:
            self._error = (
                "Collection requires two distinct cameras; set camera_ids to their USB IDs."
            )
        else:
            for name, device in zip(("camera1", "camera2"), self._camera_ids, strict=True):
                camera = Webcam(
                    camera_index=device,
                    width=self.config.width,
                    height=self.config.height,
                    fps=self.config.fps,
                )
                self.register_disposable(
                    camera.image_stream().subscribe(
                        partial(self._image, name),
                        on_error=partial(self._camera_error, name),
                    )
                )

    def _joints(self, name: str, message: JointState) -> None:
        if (
            not message.name
            or len(set(message.name)) != len(message.name)
            or len(message.name) != len(message.position)
            or not all(math.isfinite(value) for value in message.position)
        ):
            return
        if name == "state" and set(message.name) != set(DUAL_OPENYAM_JOINTS):
            return
        if not set(message.name).issubset(DUAL_OPENYAM_JOINTS):
            return
        with self._lock:
            if abs(time.time() - message.ts) >= 1.0 or message.ts < self._seen.get(name, 0):
                return
            self._seen[name] = message.ts
            if name == "action":
                self._accepted.update(
                    (joint, (message.ts, value))
                    for joint, value in zip(message.name, message.position, strict=True)
                )
            if self._capture is not None and message.ts >= self._started:
                self._capture.write(name, message)

    def _image(self, name: str, image: Image) -> None:
        if self.config.camera_source == "devices":
            output = self.camera1 if name == "camera1" else self.camera2
            output.publish(image)
        with self._lock:
            if image.shape != (self.config.height, self.config.width, 3):
                self._camera_error(name, ValueError(f"Unexpected image shape: {image.shape}"))
                return
            if abs(time.time() - image.ts) >= 1.0 or image.ts < self._seen.get(name, 0):
                return
            self._seen[name] = image.ts
            if self._capture is not None and image.ts >= self._started:
                self._capture.write(name, image)

    def _camera_error(self, name: str, error: Exception) -> None:
        with self._lock:
            self._seen.pop(name, None)
            self._error = f"{name}: {error}"

    def _ready(self) -> bool:
        now = time.time()
        return all(
            abs(now - self._seen.get(name, 0.0)) < 1.0 for name in ("state", "camera1", "camera2")
        )

    def _buttons(self, buttons: Buttons) -> None:
        try:
            with self._lock:
                if buttons.left_secondary or buttons.right_secondary:
                    if self._capture is not None:
                        self.finish(keep=False)
                elif buttons.right_primary:
                    if self._capture is None:
                        self.record(task=self._task)
                    else:
                        self.finish(keep=True)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)

    def _publish_status(self) -> None:
        status = EpisodeStatus(
            ts=self._started if self._capture is not None else time.time(),
            state="recording" if self._capture is not None else "idle",
            episodes_saved=self._saved,
            episodes_discarded=self._discarded,
            last_event=self._event,
            task_label=self._task,
        )
        self.status.publish(String(status.to_json()))

    @rpc
    def record(self, task: str, split: str = "train") -> dict[str, Any]:
        with self._lock:
            if self._capture is not None:
                raise RuntimeError("Save or discard the current episode first")
            if not task.strip() or split not in ("train", "eval"):
                raise ValueError("Enter a task and choose train or eval")
            if not self._ready():
                raise RuntimeError("Wait for fresh joint feedback and both camera feeds")
            if set(self._accepted) != set(DUAL_OPENYAM_JOINTS):
                raise RuntimeError("Engage both arms and their grippers once before recording")
            self._task = task.strip()
            directory = self.config.directory / uuid4().hex
            capture = EpisodeCapture(directory, self.profile())
            self._started = time.time()
            self._metadata = {
                "id": directory.name,
                "task": self._task,
                "split": split,
                "robot_type": "dual_openyam",
                "joint_names": DUAL_OPENYAM_JOINTS,
                "camera_ids": list(self._camera_ids),
                "camera_source": self.config.camera_source,
                "fps": self.config.fps,
                "action_semantics": "driver_accepted_joint_position_updates",
                "started_at": self._started,
            }
            self._capture = capture
            self._error = None
            self._event = "start"
            self._write_boundary("start")
            # Retain the actual acceptance timestamps. Sparse commands can then
            # reconstruct the held targets for an arm that stays still in this take.
            seeds: dict[float, dict[str, float]] = {}
            for joint, (timestamp, value) in self._accepted.items():
                seeds.setdefault(timestamp, {})[joint] = value
            for timestamp, values in sorted(seeds.items()):
                capture.write(
                    "action",
                    JointState(ts=timestamp, name=list(values), position=list(values.values())),
                )
            self._publish_status()
            return self.get_status()

    def _write_boundary(self, event: EpisodeEvent) -> None:
        assert self._capture is not None
        self._capture.write(
            "status",
            String(
                EpisodeStatus(
                    ts=time.time(),
                    state="recording" if event == "start" else "idle",
                    episodes_saved=self._saved + int(event == "save"),
                    episodes_discarded=self._discarded + int(event == "discard"),
                    last_event=event,
                    task_label=self._task,
                ).to_json()
            ),
        )

    @rpc
    def finish(self, keep: bool, outcome: str = "ungraded") -> dict[str, Any]:
        with self._lock:
            capture = self._capture
            if capture is None:
                raise RuntimeError("No episode is recording")
            if outcome not in ("ungraded", "success", "failure"):
                raise ValueError("Unknown episode outcome")
            missing = [
                name
                for name in ("state", "action", "camera1", "camera2")
                if capture.counts.get(name, 0) == 0
            ]
            if keep and (missing or not self._ready()):
                raise RuntimeError(
                    f"Episode is missing fresh streams: {missing}; discard this take"
                )
            event: EpisodeEvent = "save" if keep else "discard"
            self._write_boundary(event)
            self._capture = None
            try:
                counts = capture.finish()
                metadata = {
                    **self._metadata,
                    "kept": keep,
                    "outcome": outcome,
                    "finished_at": time.time(),
                    "stream_counts": counts,
                    "frames": min(counts.get("camera1", 0), counts.get("camera2", 0)),
                    "training_frames": 0,
                    "compatible": True,
                    "path": str(capture.directory / "recording.mcap"),
                }
                manifest = capture.directory / "metadata.json.partial"
                manifest.write_text(json.dumps(metadata, indent=2) + "\n")
                manifest.replace(capture.directory / "metadata.json")
            except Exception as exc:
                self._error = f"Episode was not saved: {exc}"
                raise
            finally:
                self._event = "init"
                self._publish_status()
            self._saved += int(keep)
            self._discarded += int(not keep)
            self._event = event
            self._publish_status()
            return self.get_status()

    @rpc
    def get_status(self) -> dict[str, Any]:
        with self._lock:
            episodes = []
            for path in sorted(self.config.directory.glob("*/metadata.json")):
                metadata = json.loads(path.read_text())
                if metadata["kept"]:
                    episodes.append(metadata)
            return {
                "collection_available": True,
                "collection_ready": self._ready()
                and set(self._accepted) == set(DUAL_OPENYAM_JOINTS),
                "collection_waiting": (
                    "Waiting for fresh joint feedback and both camera feeds."
                    if not self._ready()
                    else "Engage both arms and their grippers once before recording."
                    if set(self._accepted) != set(DUAL_OPENYAM_JOINTS)
                    else None
                ),
                "recording": self._capture.directory.name if self._capture is not None else None,
                "started_at": self._started if self._capture is not None else None,
                "episodes_saved": self._saved,
                "episodes_discarded": self._discarded,
                "last_event": self._event,
                "recorded_frames": min(
                    self._capture.counts.get("camera1", 0), self._capture.counts.get("camera2", 0)
                )
                if self._capture
                else 0,
                "recording_directory": str(self.config.directory),
                "episodes": episodes,
                "task": self._task,
                "collection_error": self._error,
            }

    def _health(self) -> None:
        with self._lock:
            if abs(time.time() - self._seen.get("state", 0)) >= 1.0:
                self._accepted.clear()
            if self._capture is not None and not self._ready():
                self._error = "Recording discarded because joint feedback or a camera stopped."
                try:
                    self.finish(keep=False)
                except Exception as exc:
                    self._error = str(exc)
            self._publish_status()

    @rpc
    def stop(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.close()
                self._capture = None
        super().stop()
