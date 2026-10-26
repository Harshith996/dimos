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

"""Translate pinhole calibration to/from MuJoCo's native projection.

MuJoCo principal offsets shift the optical axis left/up in the rendered image.
Our calibration uses top-left pixel centres, matching CameraInfo/unprojection.
"""

import math

import mujoco

from dimos.sim2.sensors.camera.models.pinhole import PinholeIntrinsics


def configure_projection(
    camera: mujoco.MjsCamera,  # type: ignore[name-defined]  # MuJoCo dynamically exports spec types.
    intrinsics: PinholeIntrinsics,
) -> None:
    """Override optics without modifying the named camera's mechanical mount."""
    camera.resolution = (intrinsics.width, intrinsics.height)
    # A unit sensor is sufficient: only focal/sensor ratios affect projection.
    camera.sensor_size = (1.0, 1.0)
    camera.focal_length = (0.0, 0.0)
    camera.principal_length = (0.0, 0.0)
    camera.focal_pixel = (intrinsics.fx, intrinsics.fy)
    camera.principal_pixel = (
        (intrinsics.width - 1) / 2 - intrinsics.cx,
        (intrinsics.height - 1) / 2 - intrinsics.cy,
    )


def camera_intrinsics(
    model: mujoco.MjModel, camera_id: int, width: int, height: int
) -> PinholeIntrinsics:
    """Read the actual compiled projection at the renderer's output resolution."""
    sx, sy = model.cam_sensorsize[camera_id]
    if sx > 0 and sy > 0:
        fx, fy, px, py = model.cam_intrinsic[camera_id]
        return PinholeIntrinsics(
            width,
            height,
            float(width * fx / sx),
            float(height * fy / sy),
            float((width - 1) / 2 - width * px / sx),
            float((height - 1) / 2 - height * py / sy),
        )
    focal = height / (2 * math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2))
    return PinholeIntrinsics(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2)
