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

import asyncio
from contextlib import suppress
from copy import deepcopy

import numpy as np
import pytest

from dimos.message_utils import occupancy
from dimos.messages.builtin_interfaces.msg import Time
from dimos.messages.geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from dimos.messages.std_msgs.msg import Header
from dimos.navigation.experimental.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.experimental.message_values import occupancy_value, pose_message, pose_value
from dimos.navigation.experimental.patrolling.module import PatrollingModule


def test_navigation_values_preserve_pose_and_map_coordinates():
    header = Header(Time(17, 125000000), "map")
    pose = PoseStamped(header, Pose(Point(2.0, -3.0, 0.5), Quaternion(0.0, 0.0, 0.0, 1.0)))
    assert pose_message(pose_value(pose)) == pose
    cells = np.array([[-1, 0, 100], [20, 30, 40]], dtype=np.int8)
    grid = occupancy.from_array(cells, header=deepcopy(header), resolution=0.25, origin=pose.pose)
    values = occupancy_value(grid)
    np.testing.assert_array_equal(values.grid, cells)
    point = values.grid_to_world((2, 1, 0))
    expected = occupancy.grid_to_world(grid, (2, 1))
    assert (point.x, point.y) == (expected.x, expected.y)
    assert values.origin.position.z == 0.5
    assert values.ts == 17.125
    assert values.frame_id == "map"
    values.grid[0, 0] = 100
    assert occupancy.view(grid)[0, 0] == -1


def test_frontier_stop_publishes_generated_pose():
    explorer = WavefrontFrontierExplorer()
    received = []
    unsubscribe = explorer.goal_request.subscribe(received.append)
    try:
        pose = PoseStamped(
            Header(Time(17, 125000000), "map"),
            Pose(Point(2.0, -3.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
        )
        explorer._on_odometry(pose)
        explorer.exploration_active = True
        assert explorer.stop_exploration()
        assert received == [PoseStamped(Header(Time(17, 125000000), "world"), pose.pose)]
    finally:
        unsubscribe()
        explorer.stop()


@pytest.mark.asyncio
async def test_patrol_consumes_generated_map_and_emits_generated_goal(mocker):
    module = PatrollingModule()
    received = asyncio.Event()
    goals = []

    def capture(goal):
        goals.append(goal)
        received.set()

    unsubscribe = module.goal_request.subscribe(capture)
    task = None
    try:
        pose = PoseStamped(
            Header(Time(17, 125000000), "map"),
            Pose(Point(2.0, 3.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
        )
        grid = occupancy.from_array(
            np.zeros((20, 20), dtype=np.int8), header=deepcopy(pose.header), resolution=0.5
        )
        await module.handle_odom(pose)
        await module.handle_global_costmap(grid)
        mocker.patch.object(module._router, "next_goal", return_value=pose_value(pose))
        task = asyncio.create_task(module._patrol_loop())
        await asyncio.wait_for(received.wait(), timeout=2)
        assert goals == [pose]
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        unsubscribe()
        module.stop()
