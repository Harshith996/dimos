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

from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from dimos.sim2.sensors.camera.models.pinhole import PinholeIntrinsics
from dimos.sim2.sensors.camera.projection import camera_intrinsics
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.sensors.spec import Mount
from dimos.sim2.spec import WorldConfig
from dimos.sim2.world.scene import load_scene


def test_resize_preserves_rays_through_corresponding_pixel_centres():
    source = PinholeIntrinsics(640, 480, 390, 410, 323, 237)
    resized = source.resized(320, 120)
    point = np.array([0.2, -0.1])
    before = point * [source.fx, source.fy] + [source.cx, source.cy]
    after = point * [resized.fx, resized.fy] + [resized.cx, resized.cy]
    np.testing.assert_allclose(after, (before + 0.5) * [0.5, 0.25] - 0.5)
    assert resized.resized(640, 480) == source


@pytest.mark.parametrize(
    "field,value",
    [("fx", 0), ("fy", -1), ("cx", math.nan), ("cy", math.inf), ("width", 0), ("height", -1)],
)
def test_invalid_intrinsics_cannot_reach_renderer(field, value):
    with pytest.raises(ValueError):
        replace(PinholeIntrinsics(640, 480, 390, 410, 323, 237), **{field: value})


@pytest.mark.parametrize("named", [False, True])
def test_calibration_applies_to_added_and_named_cameras_without_changing_mount(tmp_path, named):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><worldbody><camera name="existing" pos=".1 .2 .3" fovy="90"/></worldbody></mujoco>'
    )
    calibration = PinholeIntrinsics(640, 480, 390, 410, 323, 237)
    sensor = Camera(
        "inspection",
        "existing" if named else Mount("world", xyz=(0.1, 0.2, 0.3)),
        width=320,
        height=240,
        intrinsics=calibration,
    )
    model = load_scene(WorldConfig(scene=scene, robots={}, sensors=(sensor,)))
    camera = model.camera(sensor.model_name)
    actual = camera_intrinsics(model, camera.id, 320, 240)
    assert (actual.fx, actual.fy, actual.cx, actual.cy) == pytest.approx((195, 205, 161.25, 118.25))
    np.testing.assert_allclose(camera.pos, [0.1, 0.2, 0.3])


def test_native_sensor_dimensions_preserve_unequal_focal_lengths():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="wrist" resolution="1280 720" sensorsize=".003896 .00214" focal=".00193 .00193"/></worldbody></mujoco>'
    )
    actual = camera_intrinsics(model, 0, 320, 240)
    assert actual.fx == pytest.approx(158.5215606)
    assert actual.fy == pytest.approx(216.4485981)
    assert (actual.cx, actual.cy) == (159.5, 119.5)


def test_legacy_fovy_camera_uses_square_pixels_at_output_resolution():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera fovy="60"/></worldbody></mujoco>'
    )
    actual = camera_intrinsics(model, 0, 320, 240)
    assert (actual.fx, actual.fy) == pytest.approx((120 * math.sqrt(3),) * 2)
    assert (actual.cx, actual.cy) == (159.5, 119.5)
