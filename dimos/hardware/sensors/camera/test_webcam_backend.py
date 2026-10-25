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

import cv2
import pytest

from dimos.hardware.sensors.camera.webcam import Webcam, WebcamDevice, discover_webcams


@pytest.fixture
def webcam(mocker, request):
    camera = Webcam(camera_index=request.param)
    mocker.patch("dimos.hardware.sensors.camera.webcam.threading.Thread")
    yield camera
    camera.stop()


@pytest.mark.parametrize(
    ("platform", "webcam", "device", "backend"),
    [
        ("linux", "/dev/video0", "/dev/video0", cv2.CAP_V4L2),
        ("linux", "/dev/v4l/by-id/usb-camera", "/dev/v4l/by-id/usb-camera", cv2.CAP_V4L2),
        ("linux", 2, 2, cv2.CAP_ANY),
        ("linux", "2", 2, cv2.CAP_ANY),
        ("linux", "-1", -1, cv2.CAP_ANY),
        ("linux", "-2", -2, cv2.CAP_ANY),
        ("linux", "+2", 2, cv2.CAP_ANY),
        ("linux", " 2 ", 2, cv2.CAP_ANY),
        ("linux", " -1 ", -1, cv2.CAP_ANY),
        ("linux", "rtsp://camera/stream", "rtsp://camera/stream", cv2.CAP_ANY),
        ("darwin", 0, 0, cv2.CAP_AVFOUNDATION),
        ("darwin", "/dev/video0", "/dev/video0", cv2.CAP_ANY),
    ],
    indirect=["webcam"],
)
def test_capture_backend(platform, webcam, device, backend, mocker):
    video_capture = mocker.patch("cv2.VideoCapture")
    mocker.patch("sys.platform", platform)

    webcam.start()

    video_capture.assert_called_once_with(device, backend)


@pytest.mark.parametrize("webcam", ["avfoundation:chosen"], indirect=True)
def test_saved_camera_identity_is_resolved_again_after_usb_reordering(webcam, mocker):
    mocker.patch("sys.platform", "darwin")
    inventory = mocker.patch(
        "dimos.hardware.sensors.camera.webcam.discover_webcams",
        return_value=[WebcamDevice("avfoundation:chosen", "USB camera", 2)],
    )
    capture = mocker.patch("cv2.VideoCapture")
    webcam.start()
    capture.assert_called_once_with(2, cv2.CAP_AVFOUNDATION)
    webcam.stop()
    webcam._capture_thread = None
    inventory.return_value = [WebcamDevice("avfoundation:chosen", "USB camera", 1)]
    webcam.start()
    assert capture.call_args.args == (1, cv2.CAP_AVFOUNDATION)


@pytest.mark.parametrize("webcam", ["avfoundation:missing"], indirect=True)
def test_missing_saved_camera_never_opens_a_different_device(webcam, mocker):
    mocker.patch("dimos.hardware.sensors.camera.webcam.discover_webcams", return_value=[])
    capture = mocker.patch("cv2.VideoCapture")
    with pytest.raises(RuntimeError, match="unavailable"):
        webcam.start()
    capture.assert_not_called()


def test_macos_discovery_matches_opencv_order_including_muxed_devices(mocker):
    mocker.patch("sys.platform", "darwin")
    objc = mocker.patch("dimos.hardware.sensors.camera.webcam.objc", create=True)
    first, second, muxed = [mocker.Mock() for _ in range(3)]
    for device, identity in zip((first, second, muxed), ("0x13", "0x14", "0x12"), strict=True):
        device.uniqueID.return_value = identity
        device.localizedName.return_value = "Camera"
    native = objc.lookUpClass.return_value.devicesWithMediaType_
    native.side_effect = [[second, first], [muxed], [first, second], [muxed]]

    expected = [
        WebcamDevice("avfoundation:0x12", "Camera", 0),
        WebcamDevice("avfoundation:0x13", "Camera", 1),
        WebcamDevice("avfoundation:0x14", "Camera", 2),
    ]
    assert discover_webcams() == expected
    assert discover_webcams() == expected
