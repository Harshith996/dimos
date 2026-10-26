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

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path
from dimos.navigation.go2.replanning_a_star.path_clearance import PathClearance


def _grid(*, origin=None, resolution=0.05, frame_id="odom", shape=(80, 80)):
    return OccupancyGrid(
        grid=np.zeros(shape, dtype=np.int8),
        resolution=resolution,
        origin=Pose() if origin is None else origin,
        frame_id=frame_id,
    )


@pytest.fixture
def path():
    return Path(frame_id="odom", poses=[PoseStamped(1, 1, 0), PoseStamped(2, 1, 0)])


def test_shifted_map_origin_does_not_turn_off_path_obstacle_into_collision(path):
    first = _grid()
    first.grid[28, 30] = 100
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(first)
    assert clearance.is_obstacle_ahead() is False

    # The same obstacle at world (1.5, 1.4) stays clear of the path at y=1.
    shifted = _grid(origin=Pose(0, 0.4, 0))
    shifted.grid[20, 30] = 100
    clearance.update_costmap(shifted)

    assert clearance.is_obstacle_ahead() is False
    assert clearance.mask[12, 30]
    assert not clearance.mask[20, 30]


@pytest.mark.parametrize(
    "metadata,expected_cell",
    [
        ({"origin": Pose(0.4, 0, 0)}, (20, 22)),
        ({"origin": Pose(0, 0.4, 0)}, (12, 30)),
        ({"origin": Pose(0, 0, 0.4)}, (20, 30)),
        ({"origin": Pose(0, 0, 0, 0, 0, 0.6, 0.8)}, (20, 30)),
        ({"resolution": 0.1}, (10, 15)),
        ({"frame_id": "map"}, (20, 30)),
        ({"shape": (90, 80)}, (20, 30)),
    ],
    ids=["origin-x", "origin-y", "origin-z", "origin-rotation", "resolution", "frame", "shape"],
)
def test_map_spatial_metadata_invalidates_cached_mask(path, metadata, expected_cell):
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(_grid())
    previous_mask = clearance.mask

    clearance.update_costmap(_grid(**metadata))
    updated_mask = clearance.mask

    assert updated_mask is not previous_mask
    assert updated_mask[expected_cell]


def test_origin_changed_in_place_invalidates_cached_mask(path):
    grid = _grid()
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(grid)
    previous_mask = clearance.mask

    grid.origin.position.y = 0.4

    assert clearance.mask is not previous_mask
    assert clearance.mask[12, 30]
    assert not clearance.mask[20, 30]


def test_same_geometry_reuses_mask_but_still_detects_new_real_obstacle(path):
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(_grid())
    previous_mask = clearance.mask
    assert clearance.is_obstacle_ahead() is False

    updated = _grid()
    updated.grid[20, 30] = 100
    clearance.update_costmap(updated)

    assert clearance.is_obstacle_ahead() is True
    assert clearance.mask is previous_mask


def test_shifted_map_with_blocked_path_still_rejects_unsafe_mask(path):
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(_grid())
    assert clearance.is_obstacle_ahead() is False

    blocked = _grid(origin=Pose(0, 0.4, 0))
    blocked.grid[12, 20:41] = 100
    clearance.update_costmap(blocked)

    with pytest.raises(ValueError, match="More than 5% of path points are occupied"):
        clearance.is_obstacle_ahead()


def test_changed_path_positions_invalidate_cached_mask(path):
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(_grid())
    previous_mask = clearance.mask

    for pose in path.poses:
        pose.position.y = 2

    assert clearance.mask is not previous_mask
    assert clearance.mask[40, 30]
    assert not clearance.mask[20, 30]


def test_progress_keeps_existing_distance_cache_and_discards_traversed_path():
    path = Path(
        frame_id="odom",
        poses=[PoseStamped(x, 1, 0) for x in (1, 1.5, 2.5, 3)],
    )
    clearance = PathClearance(0.2, path)
    clearance.update_costmap(_grid())
    previous_mask = clearance.mask
    assert previous_mask[20, 20]

    clearance.update_pose_index(1)
    assert clearance.mask is previous_mask

    clearance.update_pose_index(2)
    assert clearance.mask is not previous_mask
    assert clearance.mask[20, 50]
    assert not clearance.mask[20, 20]


def test_missing_map_remains_blocked(path):
    clearance = PathClearance(0.2, path)
    assert clearance.is_obstacle_ahead() is True
    clearance.update_costmap(_grid())
    assert clearance.is_obstacle_ahead() is False
