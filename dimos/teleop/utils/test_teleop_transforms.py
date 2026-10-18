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

import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.teleop.utils.teleop_transforms import webxr_point_to_robot, webxr_to_robot


def test_webxr_point_matches_the_pose_transform_position() -> None:
    point = (1.0, 2.0, 3.0)

    robot_point = webxr_point_to_robot(point)
    robot_pose = webxr_to_robot(PoseStamped(position=point), is_left_controller=True)

    assert robot_point == (-3.0, -1.0, 2.0)
    assert robot_point == pytest.approx(
        (robot_pose.position.x, robot_pose.position.y, robot_pose.position.z)
    )
