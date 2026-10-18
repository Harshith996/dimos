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

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import math
from typing import Any, cast

import numpy as np
import pytest

from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter, inside_boxes
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.tf.tf import MultiTBuffer
from dimos.utils.transform_utils import pose_to_matrix

CUBE = Vector3(0.2, 0.2, 0.2)


@pytest.fixture
def module() -> Iterator[PointCloudBoxFilter]:
    instance = PointCloudBoxFilter()
    # A local transform buffer stands in for the tf stream.
    cast("dict[str, Any]", instance.__dict__)["_tf"] = MultiTBuffer()
    yield instance
    cast("dict[str, Any]", instance.__dict__)["_tf"] = None
    instance.dispose()


@pytest.fixture
def sent(module: PointCloudBoxFilter) -> list[PointCloud2]:
    clouds: list[PointCloud2] = []
    module.filtered_cloud.subscribe(clouds.append)
    return clouds


def _cloud(points: list[list[float]]) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray(points, dtype=np.float32), frame_id="world", timestamp=1.0
    )


def _feed(module: PointCloudBoxFilter, cloud: PointCloud2) -> None:
    asyncio.run(module.handle_cloud(cloud))


def _place_tool(module: PointCloudBoxFilter, at: Vector3, yaw: float, ts: float) -> None:
    module.tfbuffer.receive_transform(
        Transform(
            translation=at,
            rotation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
            frame_id="world",
            child_frame_id="tool",
            ts=ts,
        )
    )


def test_inside_boxes_axis_aligned() -> None:
    box = (pose_to_matrix(Pose(position=Vector3(1.0, 0.0, 0.0))), np.array([0.2, 0.2, 0.2]))
    points = np.array([[1.0, 0.0, 0.0], [1.09, 0.09, -0.09], [1.11, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert inside_boxes(points, [box]).tolist() == [True, True, False, False]


def test_inside_boxes_rotated() -> None:
    # A 40 cm long, 10 cm wide box turned 45 degrees about z.
    pose = Pose(orientation=Quaternion.from_euler(Vector3(0.0, 0.0, math.pi / 4)))
    box = (pose_to_matrix(pose), np.array([0.4, 0.1, 0.1]))
    # On the turned long axis, and on the world x axis where an unturned box would reach.
    points = np.array([[0.12, 0.12, 0.0], [0.17, 0.0, 0.0]])

    assert inside_boxes(points, [box]).tolist() == [True, False]


def test_no_boxes_passes_the_cloud_through(
    module: PointCloudBoxFilter, sent: list[PointCloud2]
) -> None:
    _feed(module, _cloud([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    np.testing.assert_allclose(sent[-1].points_f32(), [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_a_fixed_box_drops_the_points_inside(
    module: PointCloudBoxFilter, sent: list[PointCloud2]
) -> None:
    module.set_box("target", Pose(position=Vector3(1.0, 0.0, 0.0)), CUBE)

    _feed(module, _cloud([[1.0, 0.0, 0.0], [1.05, -0.05, 0.05], [2.0, 0.0, 0.0]]))

    np.testing.assert_allclose(sent[-1].points_f32(), [[2.0, 0.0, 0.0]])
    assert sent[-1].frame_id == "world"


def test_a_box_on_a_moving_frame_follows_it(
    module: PointCloudBoxFilter, sent: list[PointCloud2]
) -> None:
    # 20 cm along the tool's x axis; the tool is turned 90 degrees, so in the
    # world the box sits 20 cm along y from the tool.
    module.set_box("held", Pose(position=Vector3(0.2, 0.0, 0.0)), Vector3(0.1, 0.1, 0.1), "tool")
    cloud = _cloud([[1.0, 0.2, 0.0], [1.2, 0.0, 0.0], [2.0, 0.2, 0.0]])

    _place_tool(module, Vector3(1.0, 0.0, 0.0), math.pi / 2, ts=1.0)
    _feed(module, cloud)
    np.testing.assert_allclose(sent[-1].points_f32(), [[1.2, 0.0, 0.0], [2.0, 0.2, 0.0]])

    _place_tool(module, Vector3(2.0, 0.0, 0.0), math.pi / 2, ts=2.0)
    _feed(module, cloud)
    np.testing.assert_allclose(sent[-1].points_f32(), [[1.0, 0.2, 0.0], [1.2, 0.0, 0.0]])


def test_a_box_whose_frame_is_unknown_keeps_the_points(
    module: PointCloudBoxFilter, sent: list[PointCloud2]
) -> None:
    module.set_box("held", Pose(), CUBE, "tool")

    _feed(module, _cloud([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    np.testing.assert_allclose(sent[-1].points_f32(), [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_set_and_clear_resend_the_last_cloud_at_once(
    module: PointCloudBoxFilter, sent: list[PointCloud2]
) -> None:
    # Nothing to re-send before the first cloud.
    assert module.set_box("target", Pose(), CUBE)
    assert sent == []
    _feed(module, _cloud([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    np.testing.assert_allclose(sent[-1].points_f32(), [[2.0, 0.0, 0.0]])

    module.set_box("target", Pose(position=Vector3(2.0, 0.0, 0.0)), CUBE)
    assert len(sent) == 2
    np.testing.assert_allclose(sent[-1].points_f32(), [[0.0, 0.0, 0.0]])

    assert module.clear_box("target")
    assert len(sent) == 3
    np.testing.assert_allclose(sent[-1].points_f32(), [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    assert not module.clear_box("target")
    assert len(sent) == 3


@pytest.mark.parametrize(
    "name, size",
    [
        ("", CUBE),
        ("target", Vector3(0.0, 0.2, 0.2)),
        ("target", Vector3(0.2, -0.1, 0.2)),
        ("target", Vector3(0.2, 0.2, math.inf)),
        ("target", Vector3(0.2, 0.2, math.nan)),
    ],
)
def test_invalid_boxes_are_refused(module: PointCloudBoxFilter, name: str, size: Vector3) -> None:
    with pytest.raises(ValueError):
        module.set_box(name, Pose(), size)
