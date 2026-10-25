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

"""Single-arm office station using the existing CAN and camera drivers.

The shared controller uses closure (0=open, 1=closed). The CAN adapter uses
opening. Only this device boundary converts that coordinate. A successful
write acknowledges driver acceptance, not a measured motor target sequence.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
from importlib import import_module
import json
from threading import RLock, local
import time
from typing import Any, Literal, cast
from uuid import uuid4

import cv2
import numpy as np
from pydantic import BaseModel, Field

from dimos.core.run_registry import list_runs
from dimos.core.transport import ZenohTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.hardware.sensors.camera.webcam import Webcam, discover_webcams
from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.damiao.config import DamiaoRuntimeConfig
from dimos.hardware.whole_body.spec import IMUState, MotorCommand, MotorState
from dimos.imitation.loop.device import Observation
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.robot.manipulators.openyam.config import openyam_hardware
from dimos.robot.manipulators.openyam.control import YamControl
from dimos.robot.manipulators.openyam.device import AcceptedTarget, AuthorityMode
from dimos.robot.manipulators.openyam.simulation import YamArtifact
from dimos.robot.manipulators.openyam.workcell import FPS, HEIGHT, WIDTH


class StationConfig(BaseModel):
    name: str = Field(min_length=1)
    can_device: str = Field(min_length=1)
    camera_serial: str = Field(min_length=1)
    camera_kind: Literal["realsense", "webcam"] = "realsense"
    controller_hand: Literal["left", "right"] = "right"
    camera_width: int = Field(default=848, gt=0)
    camera_height: int = Field(default=480, gt=0)
    camera_fps: int = Field(default=30, gt=0)

    @property
    def observation_profile(self) -> str:
        camera = self.model_dump(exclude={"can_device", "controller_hand"})
        identity = hashlib.sha256(json.dumps(camera, sort_keys=True).encode()).hexdigest()[:16]
        return f"openyam-physical-wrist-v2:{self.name}:{identity}"


def station_owners(can_device: str | None = None) -> list[dict[str, Any]]:
    """Registered OpenYAM runs must be stopped by their operator before handoff."""
    owners = []
    for run in list_runs():
        if "openyam" not in run.blueprint:
            continue
        ports = [
            str(value) for key, value in run.config_overrides.items() if "can" in key and value
        ]
        argv = run.original_argv or run.cli_args
        for index, arg in enumerate(argv):
            option, separator, value = arg.partition("=")
            if option in ("--can-port", "--left-can-port", "--right-can-port"):
                if separator:
                    ports.append(value)
                elif index + 1 < len(argv):
                    ports.append(argv[index + 1])
        if can_device is None or not ports or can_device in ports:
            owners.append({"run_id": run.run_id, "blueprint": run.blueprint, "pid": run.pid})
    return owners


def discover_station() -> dict[str, Any]:
    """Read USB inventory and run metadata without connecting to motors."""
    errors = []
    adapters = []
    cameras = []
    try:
        driver = import_module("can_motor_control")
        adapters = [
            {"serial": device.serial_number, "index": device.index}
            for device in driver.list_gs_usb_devices(vendor_id=0x1D50, product_id=0x606F)
        ]
    except (ImportError, RuntimeError, OSError) as error:
        errors.append(f"CAN discovery: {error}")
    try:
        cameras = [asdict(camera) for camera in discover_webcams()]
    except (RuntimeError, OSError) as error:
        errors.append(f"Camera discovery: {error}")
    return {"adapters": adapters, "cameras": cameras, "owners": station_owners(), "errors": errors}


class PhysicalYamDevice:
    def __init__(self, adapter: Any, limits: JointLimits) -> None:
        self.adapter, self.limits = adapter, limits
        self.enabled = False
        self._lock = RLock()
        self._tick = local()
        self._epoch = 0
        self.generation = 0
        self._authority: AuthorityMode = "hold"
        self._sequence = 0
        self._sample: tuple[list[MotorState], float, AcceptedTarget | None] | None = None
        self._accepted: AcceptedTarget | None = None

    def connect(self) -> bool:
        return bool(self.adapter.connect())

    def disconnect(self) -> None:
        with self._lock:
            self.enabled = False
            self.adapter.disconnect()

    def is_connected(self) -> bool:
        return bool(self.adapter.is_connected())

    def activate(self) -> bool:
        with self._lock:
            self.enabled = bool(self.adapter.activate())
            if self.enabled:
                self.generation += 1
                self._sample = self._accepted = None
            return self.enabled

    def has_motor_states(self) -> bool:
        return bool(self.adapter.has_motor_states())

    def get_limits(self) -> JointLimits:
        return self.limits

    def read_imu(self) -> IMUState:
        return cast("IMUState", self.adapter.read_imu())

    def read_motor_states(self) -> list[MotorState]:
        with self._lock:
            states = self.adapter.read_motor_states()
            states = [*states[:-1], replace(states[-1], q=1 - states[-1].q, dq=-states[-1].dq)]
            self._tick.epoch = self._epoch
            self._tick.generation = self.generation
            self._sample = (states, time.time(), self._accepted)
            return states

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        with self._lock:
            if not self.enabled or getattr(self._tick, "epoch", -1) != self._epoch:
                return False
            values = np.asarray([command.q for command in commands])
            if (
                len(values) != 7
                or not np.isfinite(values).all()
                or np.any(values < self.limits.position_lower)
                or np.any(values > self.limits.position_upper)
            ):
                return False
            converted = [
                *commands[:-1],
                replace(commands[-1], q=1 - commands[-1].q, dq=-commands[-1].dq),
            ]
            if not self.adapter.write_motor_commands(converted):
                return False
            self._sequence += 1
            self._accepted = AcceptedTarget(
                self._sequence,
                time.time(),
                tuple(values),
                self._authority,
                self._epoch,
                self.generation,
            )
            return True

    def transition(self, mode: AuthorityMode) -> int:
        with self._lock:
            self._epoch += 1
            self._authority = mode
            self._accepted = None
            return self._epoch

    def observed_generation(self) -> int:
        return int(self._tick.generation)

    def snapshot(self) -> tuple[list[MotorState], float, AcceptedTarget | None] | None:
        with self._lock:
            return self._sample if self.enabled else None


class OpenYamHardware:
    """Opening a station connects sensors; Enable is a separate operator action."""

    def __init__(self, artifact: YamArtifact, station: StationConfig) -> None:
        self.station = station
        joints = artifact.robot_instance().config.joints
        limits = JointLimits(
            position_lower=[j.lower for j in joints],
            position_upper=[j.upper for j in joints],
            velocity_max=[min(j.max_velocity, 0.5) for j in joints],
        )
        adapter_type = import_module(
            "dimos.hardware.whole_body.openyam_damiao.adapter"
        ).OpenYamDamiaoAdapter
        self.device = PhysicalYamDevice(
            adapter_type(
                runtime_config=DamiaoRuntimeConfig(
                    bus_devices={"openyam": station.can_device}, gravity_comp=True
                )
            ),
            limits,
        )
        component = openyam_hardware(bus_devices={"openyam": station.can_device})
        component.auto_enable = False
        component.limits = limits
        self.control = YamControl(self.device, component, hand=station.controller_hand)
        self._stack = ExitStack()
        self._image: Image | None = None
        self._camera_error: str | None = None
        self._lock = RLock()
        identity = {
            "station": station.model_dump(exclude={"can_device", "controller_hand"}),
            "model": artifact.contract_sha256,
            "coordinates": "joint-radians-and-gripper-closure",
            "acknowledgment": "driver-accepted",
        }
        self.contract_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.observation_profile = station.observation_profile

    def __enter__(self) -> OpenYamHardware:
        try:
            owners = station_owners(self.station.can_device)
            if owners:
                owner = owners[0]
                raise RuntimeError(
                    f"Arm is in use by {owner['blueprint']} (PID {owner['pid']}). "
                    "Stop that run with Ctrl-C in its terminal, then Connect station here. "
                    "Workcell has not connected or changed motor state."
                )
            self._stack.callback(self.control.close)
            self.control.start(enable=False)
            if self.station.camera_kind == "webcam":
                camera = Webcam(
                    camera_index=self.station.camera_serial,
                    width=self.station.camera_width,
                    height=self.station.camera_height,
                    fps=self.station.camera_fps,
                )
                subscription = camera.image_stream().subscribe(
                    self._on_image, on_error=self._on_camera_error
                )
                self._stack.callback(subscription.dispose)
                if self._camera_error:
                    raise RuntimeError(self._camera_error)
                return self
            realsense = RealSenseCamera(
                serial_number=self.station.camera_serial,
                width=self.station.camera_width,
                height=self.station.camera_height,
                fps=self.station.camera_fps,
                enable_depth=False,
                enable_pointcloud=False,
                rpc_name=f"yam-camera-{uuid4().hex}",
            )
            transport: ZenohTransport[Image] = ZenohTransport(f"yam-camera/{uuid4().hex}", Image)
            self._stack.callback(transport.stop)
            realsense.set_transport("color_image", transport)
            self._stack.callback(realsense.color_image.subscribe(self._on_image))
            self._stack.callback(realsense.stop)
            realsense.build()
            realsense.start()
            return self
        except BaseException:
            self._stack.close()
            raise

    def _on_camera_error(self, error: Exception) -> None:
        with self._lock:
            self._camera_error = str(error)
            self._image = None

    def _on_image(self, image: Image) -> None:
        rgb = cv2.resize(image.to_rgb().data, (WIDTH, HEIGHT))
        sample = Image.from_numpy(rgb, format=ImageFormat.RGB, ts=image.ts)
        with self._lock:
            self._image = sample
        self.control.publish_camera(sample)

    @property
    def overview(self) -> None:
        return None

    @property
    def preview(self) -> Image | None:
        with self._lock:
            return self._image

    def preflight(self) -> dict[str, Any]:
        image = self.preview
        return {
            "station": self.station.name,
            "arm_connected": self.device.is_connected(),
            "camera_ready": image is not None and time.time() - image.ts < 0.5,
            "enabled": self.device.enabled,
            "camera_error": self._camera_error,
            "can_device": self.station.can_device,
            "camera_id": self.station.camera_serial,
            "controller_hand": self.station.controller_hand,
            "acknowledgment": "driver-accepted",
        }

    def enable(self) -> None:
        if not self.preflight()["camera_ready"]:
            raise RuntimeError("Wait for the configured wrist camera before enabling the arm")
        if not self.device.activate():
            raise RuntimeError("Arm activation or gripper calibration failed")
        self.control.transition("hold")

    def observe(self) -> Observation | None:
        sample, image = self.device.snapshot(), self.preview
        if sample is None or image is None:
            return None
        states, timestamp, accepted = sample
        if time.time() - min(timestamp, image.ts) > 0.5 or abs(timestamp - image.ts) > 0.1:
            return None
        return Observation(
            np.asarray([state.q for state in states]),
            timestamp,
            image.data,
            image.ts,
            self.device.generation,
            accepted.sequence if accepted else 0,
            accepted,
            time.time(),
        )

    def command(self, action: np.ndarray, timestamp: float) -> None:
        if not self.device.enabled or time.time() - timestamp > 1 / FPS:
            raise RuntimeError("Arm is disabled or target expired")
        self.control.target(action.tolist())

    def __exit__(self, *_: Any) -> None:
        self._stack.close()
