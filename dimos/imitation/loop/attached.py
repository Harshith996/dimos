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

"""Workcell's client of an already-running dual-arm blueprint."""

from contextlib import ExitStack
from functools import partial
import math
from threading import Lock
import time
from typing import TYPE_CHECKING, Any, cast

from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus
from dimos.core.global_config import global_config
from dimos.core.run_registry import RunEntry, is_pid_alive, list_runs
from dimos.hardware.sensors.camera.webcam import Webcam, discover_webcams
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.porcelain.dimos import Dimos
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_JOINTS,
)
from dimos.robot.manipulators.dual_openyam.model import DUAL_OPENYAM_MODEL
from dimos.teleop.webxr.module import WebXRTeleopConfig

if TYPE_CHECKING:
    from dimos.control.coordinator import ControlCoordinator

JOG_SPEED = 0.25  # rad/s; explicit Workcell input still uses coordinator arbitration.
BLUEPRINT = "teleop-webxr-dual-openyam"


def running_dual_arm() -> RunEntry | None:
    runs = [run for run in list_runs() if run.blueprint == BLUEPRINT]
    if len(runs) > 1:
        raise RuntimeError("Multiple dual-arm blueprints are running; select one before attaching")
    return runs[0] if runs else None


class AttachedDualArm:
    """No arm drivers or robot lifecycle: only the existing blueprint's RPC handle."""

    def __init__(self) -> None:
        self.run: RunEntry | None = None
        self.stack = ExitStack()
        self.coordinator: ControlCoordinator | None = None
        self.session = None
        self.last_error: str | None = None
        self.frames: dict[str, str] = {}
        self.poses: list[list[float]] = []
        self.cameras: dict[str, dict[str, Any]] = {}
        self._images: dict[str, Image] = {}
        self._camera_lock = Lock()
        self._positions: dict[str, float] = {}
        self._joint_lock = Lock()
        self._feedback_run_id: str | None = None
        self._targets: dict[str, float] = {}
        self._lower: list[float] = []
        self._upper: list[float] = []
        self._updated_at = 0.0
        self._next_update = 0.0
        self._next_discovery = 0.0
        self._config: dict[str, Any] = {}
        self._generation: int | None = None
        self._heartbeat = time.monotonic()
        self._jog_until = 0.0
        self.teleop_port = WebXRTeleopConfig.model_fields["server_port"].default

    def start(self) -> None:
        self._config = global_config.model_dump()
        model = DUAL_OPENYAM_MODEL.load()
        for name in DUAL_OPENYAM_JOINTS:
            joint = model.get_joint(name)
            if name not in DUAL_OPENYAM_ARM_JOINTS:
                self._lower.append(0.0)
                self._upper.append(1.0)
            elif joint is None or joint.lower is None or joint.upper is None:
                raise ValueError(f"Missing joint limits for {name}")
            else:
                self._lower.append(joint.lower)
                self._upper.append(joint.upper)
        self._follow_run()

    def _follow_run(self) -> None:
        try:
            run = running_dual_arm()
            if run is not None and self.run is not None and run.run_id == self.run.run_id:
                return
            self._disconnect()
            self.last_error = None
            if run is not None:
                self._attach(run)
        except (OSError, RuntimeError, LookupError) as exc:
            self._disconnect()
            self.last_error = f"Waiting for blueprint connection: {exc}"

    def _attach(self, run: RunEntry) -> None:
        # Each run may use a different bus. Never inherit the previous run's overrides.
        global_config.update(**{**self._config, **run.config_overrides})
        app = Dimos.connect(timeout=1.0)
        self.stack.callback(app.stop)  # Remote Dimos.stop closes only the client.
        handle = app.get_module("ControlCoordinator")
        self.coordinator = cast("ControlCoordinator", handle)
        self.run = run
        with self._joint_lock:
            self._feedback_run_id = run.run_id
        self.stack.callback(
            handle.get_stream("coordinator_joint_state").subscribe(
                partial(self._joint_state, run.run_id)
            )
        )
        self.teleop_port = WebXRTeleopConfig.model_fields["server_port"].default
        for index, arg in enumerate(run.cli_args):
            option, separator, value = arg.partition("=")
            if option in ("--server-port", "--armteleopmodule.server-port"):
                self.teleop_port = int(value if separator else run.cli_args[index + 1])
        # Prefer image streams already published by the blueprint. The existing
        # office teleop blueprint has none, so view its USB cameras locally.
        streams = [s for s in app.list_streams() if s.direction == "out" and s.type_name == "Image"]
        for index, stream in enumerate(streams, 1):
            key = f"camera{index}"
            self.cameras[key] = {"key": key, "label": stream.name, "id": stream.channel}
            handle = app.get_module(str(stream.module_name))
            self.stack.callback(handle.get_stream(stream.name).subscribe(partial(self._image, key)))
        if not streams:
            devices = [camera for camera in discover_webcams() if "usb" in camera.name.lower()]
            for index, device in enumerate(devices, 1):
                key = f"camera{index}"
                self.cameras[key] = {"key": key, "label": f"Camera {index}", "id": device.id}
                camera = Webcam(camera_index=device.id, width=640, height=480, fps=15)
                subscription = camera.image_stream().subscribe(
                    partial(self._image, key), on_error=partial(self._camera_error, key)
                )
                self.stack.callback(subscription.dispose)

    def _disconnect(self) -> None:
        self.run = None
        self.coordinator = None
        self._generation = None
        self._targets.clear()
        with self._joint_lock:
            self._feedback_run_id = None
            self._positions.clear()
            self._updated_at = 0.0
        self.stack.close()
        with self._camera_lock:
            self._images.clear()
            self.cameras.clear()
        self.frames.clear()

    def _image(self, key: str, image: Image) -> None:
        with self._camera_lock:
            self._images[key] = image
            self.cameras[key].pop("error", None)

    def _camera_error(self, key: str, error: Exception) -> None:
        with self._camera_lock:
            self._images.pop(key, None)
            self.cameras[key]["error"] = str(error)

    def _require_run(self) -> "ControlCoordinator":
        if (
            self.run is None
            or not is_pid_alive(self.run.pid)
            or not self.run.registry_path.exists()
        ):
            raise RuntimeError(
                "Waiting for the dual-arm blueprint. Workcell reconnects automatically."
            )
        assert self.coordinator is not None
        return self.coordinator

    def _joint_state(self, run_id: str, state: JointState) -> None:
        positions = dict(zip(state.name, state.position, strict=True))
        if any(
            name not in positions or not math.isfinite(positions[name])
            for name in DUAL_OPENYAM_JOINTS
        ):
            return
        if abs(time.time() - state.ts) > 1.0:
            return
        with self._joint_lock:
            if run_id != self._feedback_run_id:
                return
            self._positions = positions
            self._updated_at = time.monotonic()

    def _read_positions(self) -> dict[str, float]:
        with self._joint_lock:
            if not self._positions or time.monotonic() - self._updated_at >= 1.0:
                raise RuntimeError("The blueprint is not publishing fresh joint feedback")
            return dict(self._positions)

    def state(self) -> dict[str, Any]:
        with self._joint_lock:
            positions = dict(self._positions)
            fresh = bool(positions) and time.monotonic() - self._updated_at < 1.0
        with self._camera_lock:
            cameras = [
                dict(camera, ready=key in self.frames) for key, camera in self.cameras.items()
            ]
        return {
            "attached": {"blueprint": BLUEPRINT, "pid": self.run.pid if self.run else None},
            "run_id": self.run.run_id if self.run else None,
            "running": fresh,
            "environment": "hardware",
            "mode": "Waiting for blueprint"
            if self.run is None
            else "Attached"
            if fresh
            else "Feedback unavailable",
            "controller_hand": "right",
            "station": None,
            "learning_available": False,
            "joint_names": DUAL_OPENYAM_JOINTS,
            "joints": [positions[name] for name in DUAL_OPENYAM_JOINTS] if positions else [],
            "targets": [self._targets.get(name, positions[name]) for name in DUAL_OPENYAM_JOINTS]
            if positions
            else [],
            "lower": self._lower,
            "upper": self._upper,
            "cameras": cameras,
            "teleop_port": self.teleop_port,
            "episodes": [],
            "models": [],
            "job": None,
            "recording": None,
            "policy": None,
            "error": self.last_error
            or (
                None
                if fresh or self.run is None
                else "The blueprint is not publishing fresh joint feedback"
            ),
        }

    def dispatch(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if action == "heartbeat":
            self._heartbeat = time.monotonic()
            return {"ok": True}
        if action == "scene":
            return {"meshes": {}, "geometries": []}
        if self.run is None or params.get("run_id") != self.run.run_id:
            raise RuntimeError("The displayed run changed; Workcell reconnects automatically")
        coordinator = self._require_run()
        if action == "target":
            index, value = params.get("index"), params.get("value")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(DUAL_OPENYAM_JOINTS)
            ):
                raise ValueError("Unknown joint")
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not self._lower[index] <= value <= self._upper[index]
            ):
                raise ValueError("Joint target is outside its limits")
            positions = self._read_positions()
            name = DUAL_OPENYAM_JOINTS[index]
            if name not in DUAL_OPENYAM_ARM_JOINTS:
                side = name.split("_", 1)[0]
                accepted = coordinator.task_invoke(
                    f"{side}_arm_gripper", "set_normalized", {"values": [value]}
                )
                if not accepted:
                    raise RuntimeError("The gripper task rejected the command")
            else:
                generation = coordinator.get_trajectory_generation()
                start = positions[name]
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
            self._targets[name] = float(value)
            self._heartbeat = time.monotonic()
        elif action == "hold":
            if self._generation is not None:
                coordinator.cancel_trajectory(expected_generation=self._generation)
                self._generation = None
            self._targets.clear()
        else:
            raise RuntimeError(f"{action} is not available on the attached dual-arm blueprint")
        self.last_error = None
        return self.state()

    def update(self) -> None:
        if time.monotonic() < self._next_update:
            return
        self._next_update = time.monotonic() + 0.1
        if self.run is not None and (
            not is_pid_alive(self.run.pid) or not self.run.registry_path.exists()
        ):
            self._disconnect()
            self._next_discovery = 0.0
        if time.monotonic() >= self._next_discovery:
            self._next_discovery = time.monotonic() + 1.0
            self._follow_run()
        if self.run is None:
            return
        self._targets.clear()
        if time.monotonic() > self._jog_until:
            self._generation = None
        elif time.monotonic() - self._heartbeat > 2.0:
            self.dispatch("hold", {"run_id": self.run.run_id})

    def images(self) -> dict[str, Image]:
        with self._camera_lock:
            return {
                key: image for key, image in self._images.items() if time.time() - image.ts < 1.0
            }

    def close(self) -> None:
        try:
            if self.run is not None and self._generation is not None and is_pid_alive(self.run.pid):
                self.dispatch("hold", {"run_id": self.run.run_id})
        finally:
            self._disconnect()
