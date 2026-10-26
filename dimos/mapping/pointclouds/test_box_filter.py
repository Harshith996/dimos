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

from dimos.mapping.pointclouds import box_filter
from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.tf.tf import MultiTBuffer


def _module(**config: Any) -> Iterator[PointCloudBoxFilter]:
    instance = PointCloudBoxFilter(**config)
    # A local transform buffer stands in for the tf stream.
    cast("dict[str, Any]", instance.__dict__)["_tf"] = MultiTBuffer()
    yield instance
    cast("dict[str, Any]", instance.__dict__)["_tf"] = None
    instance.dispose()


@pytest.fixture
def module() -> Iterator[PointCloudBoxFilter]:
    yield from _module()


@pytest.fixture
def re_expressing() -> Iterator[PointCloudBoxFilter]:
    """A filter that sends its clouds in ``base``, without waiting for a late transform."""
    yield from _module(target_frame="base", tf_forward_tolerance_s=0.0)


@pytest.fixture
def sent(module: PointCloudBoxFilter) -> list[PointCloud2]:
    clouds: list[PointCloud2] = []
    module.filtered_cloud.subscribe(clouds.append)
    return clouds


def _cloud(points: list[list[float]], frame_id: str = "world", ts: float = 1.0) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray(points, dtype=np.float32), frame_id=frame_id, timestamp=ts
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


def test_a_fixed_box_drops_the_points_inside(
    module: PointCloudBoxFilter, sent: list[PointCloud2]
) -> None:
    module.set_box("target", Pose(position=Vector3(1.0, 0.0, 0.0)), Vector3(0.2, 0.2, 0.2))

    _feed(module, _cloud([[1.0, 0.0, 0.0], [1.05, -0.05, 0.05], [2.0, 0.0, 0.0]]))

    np.testing.assert_allclose(sent[-1].points_f32(), [[2.0, 0.0, 0.0]])


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


def _place_base(module: PointCloudBoxFilter, at: Vector3, yaw: float, ts: float) -> None:
    """Where the base stands in odom; the filter looks the inverse up."""
    module.tfbuffer.receive_transform(
        Transform(
            translation=at,
            rotation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
            frame_id="odom",
            child_frame_id="base",
            ts=ts,
        )
    )


def test_a_target_frame_re_expresses_the_cloud_at_its_stamp(
    re_expressing: PointCloudBoxFilter,
) -> None:
    sent: list[PointCloud2] = []
    re_expressing.filtered_cloud.subscribe(sent.append)
    # The base stands 1 m along odom x, turned 90 degrees: odom (1, 1, 0) is 1 m ahead of it.
    _place_base(re_expressing, Vector3(1.0, 0.0, 0.0), math.pi / 2, ts=1.0)
    # Where the base went next does not move a cloud stamped before it.
    _place_base(re_expressing, Vector3(5.0, 0.0, 0.0), 0.0, ts=1.5)
    cloud = _cloud([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], frame_id="odom", ts=1.0)
    cloud.seq = 7

    _feed(re_expressing, cloud)

    (out,) = sent
    assert (out.frame_id, out.ts, out.seq) == ("base", 1.0, 7)
    np.testing.assert_allclose(out.points_f32(), [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], atol=1e-6)


def test_a_box_without_a_frame_is_in_the_target_frame(
    re_expressing: PointCloudBoxFilter,
) -> None:
    sent: list[PointCloud2] = []
    re_expressing.filtered_cloud.subscribe(sent.append)
    _place_base(re_expressing, Vector3(1.0, 0.0, 0.0), math.pi / 2, ts=1.0)
    # 1 m ahead of the base: odom (1, 1, 0), not odom (1, 0, 0).
    re_expressing.set_box("target", Pose(position=Vector3(1.0, 0.0, 0.0)), Vector3(0.2, 0.2, 0.2))

    _feed(re_expressing, _cloud([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0]], frame_id="odom"))

    np.testing.assert_allclose(sent[-1].points_f32(), [[2.0, 0.0, 0.0]], atol=1e-6)


def test_a_cloud_with_no_transform_is_dropped_until_one_arrives(
    re_expressing: PointCloudBoxFilter, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        box_filter.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    sent: list[PointCloud2] = []
    re_expressing.filtered_cloud.subscribe(sent.append)
    cloud = _cloud([[1.0, 1.0, 0.0]], frame_id="odom", ts=1.0)

    for _ in range(3):
        _feed(re_expressing, cloud)
    assert sent == []
    assert len(warnings) == 1, "one warning for a run of dropped clouds, not one per cloud"
    assert warnings[0].startswith("No transform from base to odom")

    # The transform lands, and a box change re-sends the last cloud, now placed.
    _place_base(re_expressing, Vector3(1.0, 0.0, 0.0), math.pi / 2, ts=1.0)
    re_expressing.set_box("held", Pose(position=Vector3(9.0, 0.0, 0.0)), Vector3(0.1, 0.1, 0.1))
    (out,) = sent
    assert out.frame_id == "base"
    np.testing.assert_allclose(out.points_f32(), [[1.0, 0.0, 0.0]], atol=1e-6)


def test_a_cloud_too_far_from_any_transform_is_dropped(
    re_expressing: PointCloudBoxFilter,
) -> None:
    sent: list[PointCloud2] = []
    re_expressing.filtered_cloud.subscribe(sent.append)
    _place_base(re_expressing, Vector3(1.0, 0.0, 0.0), 0.0, ts=1.0)

    _feed(re_expressing, _cloud([[1.0, 1.0, 0.0]], frame_id="odom", ts=1.0 + 0.5))

    assert sent == [], "0.5 s from the only edge, outside the 0.1 s tolerance"


def test_an_empty_cloud_is_relabelled_without_a_transform(
    re_expressing: PointCloudBoxFilter,
) -> None:
    sent: list[PointCloud2] = []
    re_expressing.filtered_cloud.subscribe(sent.append)

    _feed(re_expressing, PointCloud2(frame_id="odom", ts=1.0))

    (out,) = sent
    assert (out.frame_id, len(out.points_f32())) == ("base", 0)
