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
from scipy.spatial.transform import Rotation

from dimos.robot.galaxea.r1pro.head_depth import HeadFloorPriorConfig, grow_floor, onto_floor

CAMERA_HEIGHT_M = 1.3


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """Rays from a level camera (optical frame, y down) over a floor, a 0.3 m box face 2 m ahead in the middle columns;
    floor depth read 10% short (so it floats over the floor), the box read right."""
    u, v = np.meshgrid(np.linspace(-0.4, 0.4, 17), np.linspace(0.3, 0.9, 13))
    rays = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    floor_depth = CAMERA_HEIGHT_M / rays[:, 1]
    box = (np.abs(rays[:, 0]) < 0.1) & (floor_depth > 2.0)
    depth = np.where(box, 2.0, 0.9 * floor_depth)
    return rays * depth[:, None], box


def _grow(points: np.ndarray, seed_rows: int) -> np.ndarray:
    height = CAMERA_HEIGHT_M - points[:, 1]
    floor_depth = CAMERA_HEIGHT_M / (points[:, 1] / points[:, 2])
    log_scale = np.log(floor_depth / points[:, 2])
    # Only the nearest rows sit at or under the floor; the rest float above it.
    height[np.argsort(-points[:, 1] / points[:, 2])[: 17 * seed_rows]] = 0.0
    return grow_floor(points, log_scale, height, tolerance_m=0.05, smoothness=0.04)


def test_floor_grows_from_its_seeds_but_not_up_an_obstacle() -> None:
    points, box = _scene()
    floor_like = _grow(points, seed_rows=2)
    assert floor_like[~box].all()
    # Where the face meets the floor it reads like floor; above that it must not.
    face_up = box & (CAMERA_HEIGHT_M - points[:, 1] > 0.1)
    assert face_up.sum() > 10
    assert not floor_like[face_up].any()


def test_nothing_is_floor_without_a_seed() -> None:
    points, _ = _scene()
    assert not _grow(points, seed_rows=0).any()


def _pose(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose


# Optical frame (z forward, y down) to a base whose x is forward and z up.
OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@pytest.mark.parametrize(
    ("camera_height_m", "pitch_deg"),
    [(1.3, 20.0), (0.9, 45.0), (1.1, 5.0)],
    ids=["up", "bent", "level"],
)
def test_floor_lands_on_the_floor_whatever_the_torso_and_base_pose(
    camera_height_m: float, pitch_deg: float
) -> None:
    """The head rides an articulated torso on a base that drives: every pose comes in as a transform."""
    floor_from_camera = _pose(
        Rotation.from_euler("y", pitch_deg, degrees=True).as_matrix() @ OPTICAL,
        [0.1, 0.0, camera_height_m],
    )
    world_from_floor = _pose(
        Rotation.from_euler("z", 130, degrees=True).as_matrix(), [4.0, -2.0, 0.3]
    )
    world_from_camera = world_from_floor @ floor_from_camera
    level = 0.02
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.2, 0.8, 21))
    rays = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    down = (floor_from_camera[:3, :3] @ rays.T)[2]
    rays = rays[down < -0.05]
    floor_depth = (level - camera_height_m) / (floor_from_camera[:3, :3] @ rays.T)[2]
    rays, floor_depth = rays[floor_depth < 4.0], floor_depth[floor_depth < 4.0]
    # Read 15% too far at the bottom of the image (under the floor), drifting to 10% short at the top (over it).
    row = (rays[:, 1].max() - rays[:, 1]) / np.ptp(rays[:, 1])
    points = rays * (floor_depth * (1.15 - 0.25 * row))[:, None]
    # The lidar saw floor all around, so floor_reach_m leaves nothing out here.
    x, y = np.meshgrid(np.arange(0.0, 5.0, 0.5), np.arange(-3.0, 3.0, 0.5))
    lidar_floor = np.stack([x.ravel(), y.ravel(), np.full(x.size, level), np.ones(x.size)])
    seen = (world_from_floor @ lidar_floor)[:2].T

    on_floor = onto_floor(
        points, floor_from_camera, world_from_camera, level, seen, HeadFloorPriorConfig()
    )

    heights = (on_floor @ floor_from_camera[:3, :3].T + floor_from_camera[:3, 3])[:, 2]
    np.testing.assert_allclose(heights, level, atol=1e-6)
