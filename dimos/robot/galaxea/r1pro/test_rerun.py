# Copyright 2025-2026 Dimensional Inc.
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


import numpy as np
import pytest
import rerun as rr

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.galaxea.r1pro.head_cameras import HeadLeftCameraConfig
from dimos.robot.galaxea.r1pro.lio import R1ProLioConfig
from dimos.robot.galaxea.r1pro.rerun import render_cloud

HEAD_FRAME = HeadLeftCameraConfig.model_fields["frame_id"].default
LIDAR_FRAME = R1ProLioConfig.model_fields["lidar_frame"].default


def _cloud(frame_id: str) -> PointCloud2:
    points = np.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.2], [2.0, 0.5, 1.0]], dtype=np.float32)
    return PointCloud2.from_numpy(points, frame_id=frame_id, timestamp=1.0)


@pytest.mark.parametrize("frame_id", [HEAD_FRAME, LIDAR_FRAME, "some_other_frame"])
def test_each_cloud_hangs_off_its_own_frame(frame_id: str) -> None:
    (points_path, points), (transform_path, transform) = render_cloud(_cloud(frame_id))

    assert points_path == transform_path == f"world/lidar/{frame_id}"
    assert isinstance(points, rr.Points3D)
    assert isinstance(transform, rr.Transform3D)


@pytest.mark.parametrize("frame_id", [HEAD_FRAME, LIDAR_FRAME])
def test_sensor_clouds_are_coloured_by_height_and_never_black(frame_id: str) -> None:
    (_, points), _ = render_cloud(_cloud(frame_id))

    colors = points.colors.as_arrow_array().to_numpy(zero_copy_only=False)
    assert len(np.unique(colors)) > 1
    assert colors.min() > 0
