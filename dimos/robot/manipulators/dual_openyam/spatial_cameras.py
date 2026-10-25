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

"""Serialize the station's three native cameras in one dedicated sensor worker.

Each CameraDevice retains its acquisition clock, optical frame and intrinsics.
This avoids concurrent graphics initialization on macOS without changing sim2
or its measurement implementation. Physics and control use separate workers.
"""

from typing import Any

from dimos.core.core import rpc
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.manipulators.dual_openyam.spatial_simulation import WRIST_CAMERA
from dimos.sim2.sensors.camera.device import CameraDevice
from dimos.sim2.sensors.camera.module import SimRGBDCameraModule


class OpenYamSpatialCameras(SimRGBDCameraModule):
    left_wrist: Out[Image]
    right_wrist: Out[Image]
    left_wrist_info: Out[CameraInfo]
    right_wrist_info: Out[CameraInfo]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._wrists: dict[str, CameraDevice] = {}

    def open(self) -> None:
        super().open()
        try:
            for side in ("left", "right"):
                self._wrists[side] = CameraDevice(self.reader, WRIST_CAMERA, side)
        except BaseException:
            self.close_sensor()
            raise

    def capture(self) -> None:
        super().capture()
        for side, camera in self._wrists.items():
            for measurement in camera.poll():
                if self.reader.channel.episode_id != measurement.episode:
                    continue
                port = {"color_image": f"{side}_wrist", "camera_info": f"{side}_wrist_info"}.get(
                    measurement.port, measurement.port
                )
                getattr(self, port).publish(measurement.value)
                camera.published(measurement)

    def close_sensor(self) -> None:
        for camera in self._wrists.values():
            camera.close()
        self._wrists.clear()
        super().close_sensor()

    @rpc
    def sensor_status(self) -> dict[str, Any]:
        return {
            **super().sensor_status(),
            "wrists": {side: camera.status() for side, camera in self._wrists.items()},
        }
