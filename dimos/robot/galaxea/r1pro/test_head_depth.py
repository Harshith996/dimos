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

from dimos.robot.galaxea.r1pro.head_depth import grow_floor

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
