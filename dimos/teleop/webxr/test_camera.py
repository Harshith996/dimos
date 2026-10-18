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

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import pytest_mock
import reactivex as rx

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.teleop.webxr.camera import TeleopCamera


@pytest.fixture
def make_camera() -> Iterator[Callable[..., TeleopCamera]]:
    cameras: list[TeleopCamera] = []

    def make(**kwargs: Any) -> TeleopCamera:
        camera = TeleopCamera(**kwargs)
        cameras.append(camera)
        return camera

    yield make
    for camera in cameras:
        camera.stop()


def test_unconfigured_camera_starts_without_opening_a_device(
    make_camera: Callable[..., TeleopCamera], mocker: pytest_mock.MockerFixture
) -> None:
    webcam = mocker.patch("dimos.teleop.webxr.camera.Webcam")
    camera = make_camera(instance_name="left_wrist_camera")

    camera.start()

    webcam.assert_not_called()


def test_configured_camera_publishes_webcam_frames_until_stopped(
    make_camera: Callable[..., TeleopCamera], mocker: pytest_mock.MockerFixture
) -> None:
    frame = Image.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), format=ImageFormat.RGB)
    webcam = mocker.patch("dimos.teleop.webxr.camera.Webcam")
    webcam.return_value.image_stream.return_value = rx.of(frame)
    camera = make_camera(device="/dev/v4l/by-id/usb-left", width=320, height=240, fps=15.0)
    published: list[Image] = []
    camera.color_image.subscribe(published.append)

    camera.start()
    camera.stop()

    webcam.assert_called_once_with(
        camera_index="/dev/v4l/by-id/usb-left", width=320, height=240, fps=15.0
    )
    assert published == [frame]
    webcam.return_value.stop.assert_called_once_with()
