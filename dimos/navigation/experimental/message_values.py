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

"""Wire adapters for navigation algorithms that retain numerical value objects."""

from dimos.message_utils import occupancy, time as message_time
from dimos.messages.geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from dimos.messages.nav_msgs.msg import OccupancyGrid
from dimos.messages.std_msgs.msg import Header
from dimos.msgs.geometry_msgs.Pose import Pose as NumericalPose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped as NumericalPoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid as NumericalGrid


def pose_value(message: PoseStamped) -> NumericalPoseStamped:
    p, q = message.pose.position, message.pose.orientation
    return NumericalPoseStamped(
        position=(p.x, p.y, p.z),
        orientation=(q.x, q.y, q.z, q.w),
        frame_id=message.header.frame_id,
        ts=message_time.to_seconds(message.header.stamp),
    )


def pose_message(value: NumericalPoseStamped) -> PoseStamped:
    p, q = value.position, value.orientation
    return PoseStamped(
        header=Header(message_time.time_from_seconds(value.ts), value.frame_id),
        pose=Pose(Point(p.x, p.y, p.z), Quaternion(q.x, q.y, q.z, q.w)),
    )


def occupancy_value(message: OccupancyGrid) -> NumericalGrid:
    p, q = message.info.origin.position, message.info.origin.orientation
    return NumericalGrid(
        grid=occupancy.view(message).copy(),
        resolution=message.info.resolution,
        origin=NumericalPose(p.x, p.y, p.z, q.x, q.y, q.z, q.w),
        frame_id=message.header.frame_id,
        ts=message_time.to_seconds(message.header.stamp),
    )
