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

from dimos.robot.galaxea.r1pro.head_depth import (
    HeadFloorPriorConfig,
    fit_depth_to_floor,
    footprint_floor,
    onto_floor,
)

LEVEL = 0.02
# A box on the floor ahead of the robot, in floor_frame: x, y extents and its height above the floor.
BOX_X, BOX_Y, BOX_HEIGHT = (1.2, 1.6), (-0.2, 0.2), 0.25
# Optical frame (z forward, y down) to a base whose x is forward and z up.
OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _pose(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose


def _in_floor(floor_from_camera: np.ndarray, points: np.ndarray) -> np.ndarray:
    moved: np.ndarray = points @ floor_from_camera[:3, :3].T + floor_from_camera[:3, 3]
    return moved


def _scene(floor_from_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points where each ray first meets the floor or the box (true depths), and which rays hit the box."""
    u, v = np.meshgrid(np.linspace(-0.6, 0.6, 41), np.linspace(-0.3, 0.9, 41))
    rays = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    origin, direction = floor_from_camera[:3, 3], rays @ floor_from_camera[:3, :3].T
    rays, direction = rays[direction[:, 2] < -0.05], direction[direction[:, 2] < -0.05]
    t_floor = (LEVEL - origin[2]) / direction[:, 2]
    # Ray against the box (slab method), whose top is BOX_HEIGHT over the floor.
    lo = np.array([BOX_X[0], BOX_Y[0], LEVEL])
    hi = np.array([BOX_X[1], BOX_Y[1], LEVEL + BOX_HEIGHT])
    with np.errstate(divide="ignore"):
        t1, t2 = (lo - origin) / direction, (hi - origin) / direction
    t_near, t_far = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
    box = (t_near <= t_far) & (t_near > 0) & (t_near < t_floor)
    t = np.where(box, t_near, t_floor)
    keep = t < 6.0
    return rays[keep] * t[keep, None], box[keep]


def _misread(points: np.ndarray) -> np.ndarray:
    """Depth Anything's kind of error, log-linear in depth: read 25% too far at 1 m and less so farther out."""
    depth = np.linalg.norm(points, axis=1)
    misread: np.ndarray = points * (1.25 * depth**-0.15)[:, None]
    return misread


@pytest.mark.parametrize(
    ("camera_height_m", "pitch_deg"),
    [(1.3, 20.0), (0.9, 45.0), (1.1, 30.0)],
    ids=["up", "bent", "level"],
)
def test_the_frame_fit_puts_the_floor_on_the_floor_and_keeps_the_box(
    camera_height_m: float, pitch_deg: float
) -> None:
    """The head rides an articulated torso: whatever its height and pitch, the corrected floor lands on the lidar's floor
    and the box keeps its height, since the whole frame shares one correction."""
    floor_from_camera = _pose(
        Rotation.from_euler("y", pitch_deg, degrees=True).as_matrix() @ OPTICAL,
        [0.1, 0.0, camera_height_m],
    )
    truth, box = _scene(floor_from_camera)
    assert box.sum() > 10

    corrected = fit_depth_to_floor(
        _misread(truth), floor_from_camera, LEVEL, HeadFloorPriorConfig()
    )

    np.testing.assert_allclose(corrected, truth, rtol=1e-3)
    assert _in_floor(floor_from_camera, corrected[box])[:, 2].max() == pytest.approx(
        LEVEL + BOX_HEIGHT, abs=1e-3
    )


def test_too_few_floor_rays_leave_the_frame_alone() -> None:
    floor_from_camera = _pose(
        Rotation.from_euler("y", 30, degrees=True).as_matrix() @ OPTICAL, [0.1, 0.0, 1.1]
    )
    points = _misread(_scene(floor_from_camera)[0])[:5]
    np.testing.assert_array_equal(
        fit_depth_to_floor(points, floor_from_camera, LEVEL, HeadFloorPriorConfig()), points
    )


def test_points_just_off_the_floor_slide_onto_it_where_the_lidar_saw_floor() -> None:
    """Seen from a base that has driven and turned in the world; a point far from any lidar floor is left alone."""
    floor_from_camera = _pose(
        Rotation.from_euler("y", 30, degrees=True).as_matrix() @ OPTICAL, [0.1, 0.0, 1.1]
    )
    world_from_floor = _pose(
        Rotation.from_euler("z", 130, degrees=True).as_matrix(), [4.0, -2.0, 0.3]
    )
    truth, box = _scene(floor_from_camera)
    floor = truth[~box]
    # 3 cm over the floor (inside floor_tolerance_m): each ray stops short of the floor by that much height.
    raised = floor * ((LEVEL + 0.03 - 1.1) / (LEVEL - 1.1))
    # The lidar saw floor only around 1 m ahead, so floor_reach_m covers up to 2.5 m.
    seen = (world_from_floor @ np.array([[1.0, 0.0, LEVEL, 1.0]]).T)[:2].T
    reached = np.hypot(*(_in_floor(floor_from_camera, floor)[:, :2] - [1.0, 0.0]).T) < 1.4

    slid = onto_floor(
        raised,
        floor_from_camera,
        world_from_floor @ floor_from_camera,
        LEVEL,
        seen,
        HeadFloorPriorConfig(),
    )

    heights = _in_floor(floor_from_camera, slid)[:, 2]
    np.testing.assert_allclose(heights[reached], LEVEL, atol=1e-6)
    far = np.hypot(*(_in_floor(floor_from_camera, floor)[:, :2] - [1.0, 0.0]).T) > 1.6
    np.testing.assert_allclose(heights[far], LEVEL + 0.03, atol=1e-6)


def test_the_footprint_is_a_disc_of_floor() -> None:
    under = footprint_floor(0.325, LEVEL, 0.05)
    assert np.hypot(under[:, 0], under[:, 1]).max() <= 0.325
    assert (under[:, 2] == LEVEL).all()
    assert len(under) == pytest.approx(np.pi * 0.325**2 / 0.05**2, rel=0.15)
