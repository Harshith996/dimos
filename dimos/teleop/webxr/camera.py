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

"""A webcam feeding one headset video panel, selected on the command line."""

from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.hardware.sensors.camera.webcam import CameraDevice, Webcam
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class TeleopCameraConfig(ModuleConfig):
    """Webcam capture settings; an unset `device` leaves the panel blank."""

    # OpenCV index, or a /dev/v4l/by-id path that survives device renumbering.
    device: CameraDevice | None = None
    width: int = 640
    height: int = 480
    fps: float = 30.0


class TeleopCamera(Module):
    """Publish a webcam as `color_image` for a headset panel.

    Several instances share one blueprint (`instance_name` tells them apart), so
    each device is picked per instance: `--left-wrist-camera.device=...`. A
    camera left unset only blanks its panel; teleoperation does not depend on it.
    """

    config: TeleopCameraConfig

    color_image: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._webcam: Webcam | None = None

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.device is None:
            logger.warning(
                "Headset camera has no device configured; its panel stays blank",
                camera=self.config.instance_name or self.name,
            )
            return
        self._webcam = Webcam(
            camera_index=self.config.device,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        self.register_disposable(self._webcam.image_stream().subscribe(self.color_image.publish))

    @rpc
    def stop(self) -> None:
        if self._webcam is not None:
            self._webcam.stop()
            self._webcam = None
        super().stop()
