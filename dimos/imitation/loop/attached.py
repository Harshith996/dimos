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

"""Desktop attachment to advertised capabilities of a running blueprint.

The blueprint owns devices, control and services. This client owns only its
subscriptions and any jog it explicitly requested. It has no robot imports.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import math
from threading import Lock
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus
from dimos.core.global_config import global_config
from dimos.core.run_registry import RunEntry, is_pid_alive, list_runs
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.porcelain.dimos import Dimos

if TYPE_CHECKING:
    from dimos.control.coordinator import ControlCoordinator


class EpisodeRecorder(Protocol):
    def get_status(self) -> dict[str, Any]: ...
    def record(self, task: str, split: str = "train") -> dict[str, Any]: ...
    def finish(self, keep: bool, outcome: str = "ungraded") -> dict[str, Any]: ...


class WebViewProvider(Protocol):
    def get_web_views(self) -> list[dict[str, Any]]: ...


JOG_SPEED = 0.25
FEEDBACK_TIMEOUT = 1.0


@dataclass
class Attachment:
    run: RunEntry
    resources: ExitStack
    coordinator: ControlCoordinator
    joint_names: list[str]
    lower: list[float | None]
    upper: list[float | None]
    controls: dict[str, dict[str, str]]
    views: list[dict[str, Any]]
    simulation: bool
    collection: EpisodeRecorder | None = None
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)


class AttachedBlueprint:
    """Discover capabilities, follow run replacement, and fence stale user input."""

    def __init__(self) -> None:
        self.attachment: Attachment | None = None
        self.available_runs: list[RunEntry] = []
        self.selected_run_id: str | None = None
        self.session = None
        self.last_error: str | None = None
        self.frames: dict[str, str] = {}
        self.poses: list[list[float]] = []
        self._lock = Lock()
        self._images: dict[str, Image] = {}
        self._positions: dict[str, float] = {}
        self._updated_at = 0.0
        self._next_update = 0.0
        self._next_discovery = 0.0
        self._config: dict[str, Any] = {}
        self._collection_state: dict[str, Any] = {}
        self._next_collection_update = 0.0
        self._generation: int | None = None
        self._heartbeat = time.monotonic()
        self._jog_until = 0.0

    @property
    def run(self) -> RunEntry | None:
        return self.attachment.run if self.attachment else None

    def start(self) -> None:
        self._config = global_config.model_dump()
        self._follow_run()

    def _follow_run(self) -> None:
        self.available_runs = list_runs()
        candidates = self.available_runs
        if self.selected_run_id is not None:
            candidates = [r for r in candidates if r.run_id == self.selected_run_id]
        if self.run is not None and any(r.run_id == self.run.run_id for r in candidates):
            return
        self._disconnect()
        self.last_error = None
        if len(candidates) > 1:
            self.last_error = "Several blueprints are running. Select the run to view."
            return
        if not candidates:
            return
        try:
            self._attach(candidates[0])
        except (OSError, RuntimeError, LookupError, ValueError) as exc:
            self._disconnect()
            self.last_error = f"Waiting for blueprint connection: {exc}"

    def _attach(self, run: RunEntry) -> None:
        global_config.update(**{**self._config, **run.config_overrides})
        with ExitStack() as resources:
            app = Dimos.connect(timeout=1.0)
            resources.callback(app.stop)
            modules = app.list_modules()
            methods = {m.instance_name or m.name: {rpc.name for rpc in m.rpcs} for m in modules}
            owners = [
                name
                for name, rpcs in methods.items()
                if {"list_joints", "get_joint_positions"} <= rpcs
            ]
            if len(owners) != 1:
                raise RuntimeError(f"Expected one joint-control owner; found {len(owners)}")
            owner = owners[0]
            handle = app.get_module(owner)
            coordinator = cast("ControlCoordinator", handle)
            names = coordinator.list_joints()
            description = (
                coordinator.describe_controller() if "describe_controller" in methods[owner] else {}
            )
            if description and description.get("run_id") != run.run_id:
                raise RuntimeError("The responding controller belongs to a different run")
            if not description and len(self.available_runs) > 1:
                raise RuntimeError("This older controller cannot confirm its run identity")
            bounds = {j["name"]: j for j in description.get("joints", [])}
            controls: dict[str, dict[str, str]] = {}
            for task in description.get("tasks", []):
                if task["trajectory"]:
                    controls.update({name: {"kind": "trajectory"} for name in task["joints"]})
                elif "set_position" in task["commands"] and len(task["joints"]) == 1:
                    controls[task["joints"][0]] = {"kind": "position", "task": task["name"]}
            for name in list(controls):
                limits = [bounds.get(name, {}).get(key) for key in ("lower", "upper")]
                if any(value is None or not math.isfinite(value) for value in limits):
                    del controls[name]
            views = []
            recorder = None
            for name, rpcs in methods.items():
                handle = app.get_module(name)
                if "get_web_views" in rpcs:
                    views.extend(cast("WebViewProvider", handle).get_web_views())
                if {"record", "finish", "get_status"} <= rpcs:
                    if recorder is not None:
                        raise RuntimeError(
                            "Multiple recorders are available; collection is ambiguous"
                        )
                    recorder = cast("EpisodeRecorder", handle)
            binding = Attachment(
                run,
                resources,
                coordinator,
                names,
                [bounds.get(name, {}).get("lower") for name in names],
                [bounds.get(name, {}).get("upper") for name in names],
                controls,
                views,
                any({"describe_run", "scene_state"} <= rpcs for rpcs in methods.values()),
                recorder,
            )
            streams = [s for s in app.list_streams() if s.direction == "out"]
            feedback = [
                s for s in streams if s.module_name == owner and s.type_name == "JointState"
            ]
            canonical = [s for s in feedback if s.name == "coordinator_joint_state"]
            feedback = canonical or feedback
            if len(feedback) != 1:
                raise RuntimeError(
                    "The joint-control owner must publish an unambiguous JointState stream"
                )
            with self._lock:
                self.attachment = binding
            resources.callback(
                app.get_module(owner)
                .get_stream(feedback[0].name)
                .subscribe(partial(self._joint_state, run.run_id))
            )
            for index, stream in enumerate(s for s in streams if s.type_name == "Image"):
                key = f"camera{index}"
                binding.cameras[key] = {"key": key, "label": stream.name, "id": stream.channel}
                handle = app.get_module(str(stream.module_name))
                resources.callback(
                    handle.get_stream(stream.name).subscribe(partial(self._image, run.run_id, key))
                )
            if recorder is not None:
                self._collection_state = recorder.get_status()
            binding.resources = resources.pop_all()

    def _disconnect(self) -> None:
        with self._lock:
            old, self.attachment = self.attachment, None
            self._images.clear()
            self._positions.clear()
            self._updated_at = 0.0
        if old is not None:
            old.resources.close()
        self.frames.clear()
        self._collection_state.clear()
        self._generation = None
        self._next_collection_update = 0.0

    def _image(self, run_id: str, key: str, image: Image) -> None:
        if image.format in (ImageFormat.DEPTH, ImageFormat.DEPTH16):
            return
        with self._lock:
            if self.attachment is not None and self.attachment.run.run_id == run_id:
                self._images[key] = image

    def _joint_state(self, run_id: str, state: JointState) -> None:
        if (
            len(state.name) != len(state.position)
            or not math.isfinite(state.ts)
            or abs(time.time() - state.ts) > FEEDBACK_TIMEOUT
        ):
            return
        positions = dict(zip(state.name, state.position, strict=True))
        with self._lock:
            binding = self.attachment
            if binding is None or binding.run.run_id != run_id:
                return
            if any(
                name not in positions or not math.isfinite(positions[name])
                for name in binding.joint_names
            ):
                return
            self._positions = positions
            self._updated_at = time.monotonic()

    def state(self) -> dict[str, Any]:
        with self._lock:
            binding = self.attachment
            positions = dict(self._positions)
            fresh = bool(positions) and time.monotonic() - self._updated_at < FEEDBACK_TIMEOUT
            names = binding.joint_names if binding else []
            cameras = (
                [
                    dict(camera, ready=key in self.frames)
                    for key, camera in binding.cameras.items()
                    if key in self._images
                ]
                if binding
                else []
            )
        return {
            "attached": {
                "blueprint": binding.run.blueprint if binding else None,
                "pid": binding.run.pid if binding else None,
            },
            "available_runs": [
                {"run_id": r.run_id, "blueprint": r.blueprint} for r in self.available_runs
            ],
            "run_id": binding.run.run_id if binding else None,
            "running": fresh,
            "environment": "simulation" if binding and binding.simulation else "hardware",
            "mode": "Waiting for blueprint"
            if binding is None
            else "Attached"
            if fresh
            else "Feedback unavailable",
            "station": None,
            "learning_available": False,
            "collection_available": False,
            "collection_ready": False,
            "joint_names": names,
            "joints": [positions[n] for n in names] if positions else [],
            "targets": [positions[n] for n in names] if positions else [],
            "lower": binding.lower if binding else [],
            "upper": binding.upper if binding else [],
            "commandable": [n in binding.controls for n in names] if binding else [],
            "cameras": cameras,
            "web_views": binding.views if binding else [],
            "teleop_port": None,
            "episodes": [],
            "models": [],
            "job": None,
            "recording": None,
            "policy": None,
            **self._collection_state,
            "error": self.last_error
            or (
                "The blueprint is not publishing fresh joint feedback"
                if binding and not fresh
                else None
            ),
        }

    def _require_run(self, run_id: str | None) -> Attachment:
        binding = self.attachment
        if binding is None or run_id != binding.run.run_id:
            raise RuntimeError("The displayed run changed; Workcell reconnects automatically")
        if not is_pid_alive(binding.run.pid) or not binding.run.registry_path.exists():
            raise RuntimeError("Waiting for the blueprint. Workcell reconnects automatically.")
        return binding

    def dispatch(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if action == "heartbeat":
            self._heartbeat = time.monotonic()
            return {"ok": True}
        if action == "scene":
            return {"meshes": {}, "geometries": []}
        if action == "select_run":
            run_id = params.get("selected_run_id")
            if run_id not in {r.run_id for r in list_runs()}:
                raise ValueError("The selected run is no longer available")
            self.close()
            self.selected_run_id = run_id
            self._follow_run()
            return self.state()
        binding = self._require_run(params.get("run_id"))
        coordinator = binding.coordinator
        if action in ("record", "save", "discard"):
            if binding.collection is None:
                raise RuntimeError("This run does not advertise episode recording")
            if action == "record":
                self._collection_state = binding.collection.record(
                    task=params["task"], split=params.get("split", "train")
                )
            else:
                self._collection_state = binding.collection.finish(
                    keep=action == "save", outcome=params.get("outcome", "ungraded")
                )
        elif action == "target":
            index, value = params.get("index"), params.get("value")
            if type(index) is not int or not 0 <= index < len(binding.joint_names):
                raise ValueError("Unknown joint")
            name = binding.joint_names[index]
            lower, upper = binding.lower[index], binding.upper[index]
            if name not in binding.controls or lower is None or upper is None:
                raise ValueError("This joint has no advertised manual control")
            if (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("Joint target is outside its limits")
            with self._lock:
                if time.monotonic() - self._updated_at >= FEEDBACK_TIMEOUT or not self._positions:
                    raise RuntimeError("The blueprint is not publishing fresh joint feedback")
                start = self._positions[name]
            control = binding.controls[name]
            if control["kind"] == "position":
                if not coordinator.task_invoke(
                    control["task"], "set_position", {"values": [value]}
                ):
                    raise RuntimeError("The joint task rejected the command")
            else:
                generation = coordinator.get_trajectory_generation()
                trajectory = JointTrajectory(
                    joint_names=[name],
                    points=[
                        TrajectoryPoint(positions=[start]),
                        TrajectoryPoint(
                            positions=[float(value)],
                            time_from_start=max(0.15, abs(value - start) / JOG_SPEED),
                        ),
                    ],
                )
                result = coordinator.execute_trajectory(trajectory, expected_generation=generation)
                if result.status is not TrajectoryExecutionStatus.ACCEPTED:
                    raise RuntimeError(result.message or result.status.name)
                self._generation = generation
                self._jog_until = time.monotonic() + trajectory.duration + 0.2
            self._heartbeat = time.monotonic()
        elif action == "hold":
            if self._generation is not None:
                coordinator.cancel_trajectory(expected_generation=self._generation)
                self._generation = None
        else:
            raise RuntimeError(f"{action} is not available on this blueprint")
        self.last_error = None
        return self.state()

    def update(self) -> None:
        now = time.monotonic()
        if now < self._next_update:
            return
        self._next_update = now + 0.1
        if self.run is not None and (
            not is_pid_alive(self.run.pid) or not self.run.registry_path.exists()
        ):
            self._disconnect()
            self._next_discovery = 0.0
        if now >= self._next_discovery:
            self._next_discovery = now + 1.0
            self._follow_run()
        binding = self.attachment
        if binding is None:
            return
        if binding.collection is not None and now >= self._next_collection_update:
            self._next_collection_update = now + 0.5
            try:
                self._collection_state = binding.collection.get_status()
            except (RuntimeError, OSError) as exc:
                self._collection_state.update(collection_ready=False, collection_error=str(exc))
        if now > self._jog_until:
            self._generation = None
        elif now - self._heartbeat > 2.0:
            self.dispatch("hold", {"run_id": binding.run.run_id})

    def images(self) -> dict[str, Image]:
        with self._lock:
            return {
                key: image
                for key, image in self._images.items()
                if time.time() - image.ts < FEEDBACK_TIMEOUT
            }

    def close(self) -> None:
        try:
            if self.run is not None and self._generation is not None and is_pid_alive(self.run.pid):
                self.dispatch("hold", {"run_id": self.run.run_id})
        finally:
            self._disconnect()
