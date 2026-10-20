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

import numpy as np
import pytest

from dimos.simulation.alfred_sim.world import ALFRED
from dimos.simulation.go2_sim.world import GO2
from dimos.simulation.scenes.procedural import BLOCK_SIZE, block, office
from dimos.simulation.sim_world.premap import VOXEL_M, premap_cloud, without_objects

pytestmark = pytest.mark.self_hosted


def test_the_premap_is_what_the_lidar_sees_along_the_route() -> None:
    scene = office(1)
    x, y, z0 = scene.start
    route = np.array([[x, y, z0], [x + 2.0, y, z0]])
    cloud = premap_cloud(scene, GO2, route)
    lo, hi = scene.bounds()
    assert len(cloud) > 10_000 and np.all(cloud >= lo - VOXEL_M) and np.all(cloud <= hi + VOXEL_M)
    assert np.mean(np.abs(cloud[:, 2] - z0) < 0.05) > 0.1 and np.any(cloud[:, 2] > z0 + 2.0)


def test_a_premap_without_objects_shows_the_floor_where_the_block_stands() -> None:
    scene = block(1, block_height=0.4)
    bare = without_objects(scene)
    assert [box for box in scene.boxes if box.kind != "clutter"] == bare.boxes
    assert len(scene.boxes) == len(bare.boxes) + 1
    x, y, z0 = (scene.params[k] for k in ("block_x", "block_y", "z0"))
    walk = np.array([[x - 2.0, y, z0], [x - 1.0, y, z0]])

    def on_the_block(cloud: np.ndarray) -> int:
        over = np.abs(cloud[:, :2] - (x, y)).max(axis=1) < BLOCK_SIZE / 2
        return int((over & (cloud[:, 2] > z0 + 0.1)).sum())

    assert on_the_block(premap_cloud(scene, GO2, walk)) > 20
    assert on_the_block(premap_cloud(bare, GO2, walk)) == 0


def test_the_premap_keeps_the_blind_range_clear_of_the_sensor() -> None:
    scene = office(1)
    x, y, z0 = scene.start
    cloud = premap_cloud(scene, ALFRED, np.array([[x, y, z0], [x + 0.2, y, z0]]))
    sensor = np.array([x, y, z0]) + ALFRED.mount_xyz
    assert np.linalg.norm(cloud - sensor, axis=1).min() >= ALFRED.lio_blind_range - 0.2
