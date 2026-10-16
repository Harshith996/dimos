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

from dimos.mapping.dim_slam.dim_slam import DimSlam
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.robot.galaxea.r1pro.blueprints.navigation.r1pro_nav import r1pro_nav
from dimos.robot.galaxea.r1pro.lio import (
    R1ProLioConfig,
    base_odometry,
    base_pose,
    mount_transforms,
)


def test_mount_under_base_reverses_the_lidar_edge() -> None:
    over, chassis = mount_transforms(R1ProLioConfig())
    under, same_chassis = mount_transforms(R1ProLioConfig(mount_under_base=True))
    assert (over.frame_id, over.child_frame_id) == ("lidar_pointlio_link", "base_link")
    assert (under.frame_id, under.child_frame_id) == ("base_link", "lidar_pointlio_link")
    assert (over + under).translation.length() == pytest.approx(0.0, abs=1e-9)
    assert chassis.child_frame_id == same_chassis.child_frame_id == "lidar_chassis_left_link"


def test_base_odometry_is_base_pose() -> None:
    lidar_to_base = mount_transforms(R1ProLioConfig())[0]
    lidar = Odometry(
        ts=3.0,
        frame_id="odom",
        child_frame_id="lidar_pointlio_link",
        pose=Pose(Vector3(1.0, 2.0, 0.5), Quaternion.from_euler(Vector3(0.0, 0.0, 0.7))),
    )
    pose = base_pose(lidar, lidar_to_base)
    odometry = base_odometry(lidar, lidar_to_base, "base_link")
    assert (odometry.frame_id, odometry.child_frame_id, odometry.ts) == ("odom", "base_link", 3.0)
    assert (odometry.position - pose.position).length() == pytest.approx(0.0, abs=1e-12)
    assert odometry.orientation.to_euler().z == pytest.approx(pose.orientation.to_euler().z)


def topic_users(direction: str, topic: str) -> set[str]:
    """Modules with a ``direction`` ("in"/"out") stream on ``topic`` in r1pro_nav."""
    users = set()
    for atom in r1pro_nav.active_blueprints:
        for stream in atom.streams:
            name = r1pro_nav.remapping_map.get((atom.name, stream.name), stream.name)
            if name == topic and stream.direction == direction:
                users.add(atom.module.__name__)
    return users


def test_dimslam_takes_pointlio_and_wheels_and_alone_drives_the_planners() -> None:
    assert topic_users("out", "slam_sources") == {"R1ProLio", "R1ProConnection"}
    assert topic_users("in", "slam_sources") == {DimSlam.__name__}
    assert topic_users("out", "chassis_odom") <= {"PoseFromOdometry", "R1ProConnection"}
    # Off tf, so dimSLAM's odom -> base_link is base_link's only parent.
    assert "R1ProPointLio" not in topic_users("out", "tf")
