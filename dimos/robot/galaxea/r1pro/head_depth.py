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

"""The R1 Pro's head depth: Depth Anything on the left head camera, calibrated to the chassis lidar, with a floor prior."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.perception.depth2depth_cloud.module import Depth2DepthCloud
from dimos.robot.galaxea.r1pro.config import R1PRO_CHASSIS_WIDTH_M
from dimos.robot.galaxea.r1pro.lio import R1ProLioConfig

BASE_FRAME = R1ProLioConfig.model_fields["base_frame"].default


class HeadFloorPriorConfig(ModuleConfig):
    floor_frame: str = BASE_FRAME
    world_frame: str = "odom"
    tf_tolerance_s: float = 0.1
    # Lidar returns within this of floor_frame's z = 0 are floor.
    floor_band_m: float = 0.1
    # Floor counts as seen within this of a lidar floor return from the last lidar_history_s (or of the camera).
    floor_reach_m: float = 1.5
    lidar_history_s: float = 6.0
    # Rays whose head point lands this far below to this far above the lidar's floor fit the frame's depth correction:
    # the floor Depth Anything misplaces sits anywhere in that band, an obstacle is a minority of it.
    fit_min_m: float = -0.3
    fit_max_m: float = 0.9
    fit_min_rays: int = 30
    # A corrected head point no higher than this above the lidar's floor is floor.
    floor_tolerance_m: float = 0.05
    # Kept from the floor to just above the lidar's plane, what the lidar sees worst.
    min_height_m: float = -0.15
    max_height_m: float = 0.35
    # The floor under the robot, which it stands on and no sensor sees, as points this far apart; 0 radius adds none.
    footprint_radius_m: float = 0.0
    footprint_spacing_m: float = 0.05


class HeadFloorPrior(Module):
    """Head depth onto the lidar's floor. Each frame's depth is corrected by how far its near-floor rays miss the floor
    (fit_depth_to_floor: Depth Anything's error grows with distance, and the whole frame shares it, obstacles included),
    then a point at or under the floor, where the lidar saw floor within floor_reach_m (or the camera stands over it),
    slides along its ray onto the floor, and the floor under the robot's own footprint is added. Every pose comes from
    tf at the cloud's stamp, so the torso and base may move."""

    config: HeadFloorPriorConfig

    depth_cloud: In[PointCloud2]
    lidar: In[PointCloud2]
    tf: In[TFMessage]
    head_cloud: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # (stamp, floor returns' world xy, their median height in floor_frame) per lidar scan.
        self._floor_scans: deque[tuple[float, np.ndarray, float]] = deque()
        self._head_frame = ""

    @rpc
    def start(self) -> None:
        self.tfbuffer  # noqa: B018 - Initialize TF before binding input handlers.
        super().start()

    async def handle_lidar(self, cloud: PointCloud2) -> None:
        # The head's own clouds share the lidar topic in r1pro-nav.
        if cloud.ts is None or cloud.frame_id == self._head_frame:
            return
        floor_from_lidar = self._lookup(self.config.floor_frame, cloud.frame_id, cloud.ts)
        world_from_lidar = self._lookup(self.config.world_frame, cloud.frame_id, cloud.ts)
        if floor_from_lidar is None or world_from_lidar is None:
            return
        points = cloud.points_f32().astype(np.float64)
        in_floor = _apply(floor_from_lidar, points)
        on_floor = np.abs(in_floor[:, 2]) < self.config.floor_band_m
        if on_floor.sum() == 0:
            return
        world_xy = _apply(world_from_lidar, points[on_floor])[:, :2]
        self._floor_scans.append((cloud.ts, world_xy, float(np.median(in_floor[on_floor, 2]))))
        while (
            self._floor_scans and self._floor_scans[0][0] < cloud.ts - self.config.lidar_history_s
        ):
            self._floor_scans.popleft()

    async def handle_depth_cloud(self, cloud: PointCloud2) -> None:
        self._head_frame = cloud.frame_id
        floor_from_camera = self._lookup(self.config.floor_frame, cloud.frame_id, cloud.ts)
        world_from_camera = self._lookup(self.config.world_frame, cloud.frame_id, cloud.ts)
        if floor_from_camera is None or world_from_camera is None:
            return
        points = cloud.points_f32().astype(np.float64)
        scans = [
            scan for scan in self._floor_scans if scan[0] >= cloud.ts - self.config.lidar_history_s
        ]
        if scans:
            level = float(np.median([scan[2] for scan in scans]))
            points = fit_depth_to_floor(points, floor_from_camera, level, self.config)
            seen = np.concatenate([scan[1] for scan in scans])
            points = onto_floor(
                points, floor_from_camera, world_from_camera, level, seen, self.config
            )
        in_floor = _apply(floor_from_camera, points)
        keep = (in_floor[:, 2] >= self.config.min_height_m) & (
            in_floor[:, 2] <= self.config.max_height_m
        )
        points = points[keep]
        if scans and self.config.footprint_radius_m > 0:
            under = footprint_floor(
                self.config.footprint_radius_m, level, self.config.footprint_spacing_m
            )
            points = np.concatenate([points, _apply(np.linalg.inv(floor_from_camera), under)])
        self.head_cloud.publish(
            PointCloud2.from_numpy(
                points.astype(np.float32), frame_id=cloud.frame_id, timestamp=cloud.ts
            )
        )

    def _lookup(self, target: str, source: str, stamp: float) -> np.ndarray | None:
        transform = self.tfbuffer.get(
            target,
            source,
            time_point=stamp,
            time_tolerance=self.config.tf_tolerance_s,
            forward_tolerance=self.config.tf_tolerance_s,
            warn=False,
        )
        return None if transform is None else transform.to_matrix()


def fit_depth_to_floor(
    points: np.ndarray, floor_from_camera: np.ndarray, level: float, config: HeadFloorPriorConfig
) -> np.ndarray:
    """Camera-frame points rescaled along their rays by log(true / predicted depth) = a + b * log(predicted depth), fit
    robustly over the downward rays that land near the floor at height `level` in floor_frame; unchanged when too few do."""
    in_floor = _apply(floor_from_camera, points)
    camera_z = floor_from_camera[2, 3]
    height = in_floor[:, 2] - level
    near = (
        (in_floor[:, 2] < camera_z - 1e-3)
        & (height > config.fit_min_m)
        & (height < config.fit_max_m)
    )
    if near.sum() < config.fit_min_rays:
        return points
    depth = np.linalg.norm(points, axis=1)
    x = np.log(depth[near])
    y = np.log((level - camera_z) / (in_floor[near, 2] - camera_z))
    inlier = np.ones(len(x), bool)
    for _ in range(3):
        slope, intercept = np.polyfit(x[inlier], y[inlier], 1)
        residual = np.abs(y - (intercept + slope * x))
        # Rays off the line by more than 2.5 typical residuals (and at least 5%) are obstacles, not the floor.
        inlier = residual < max(0.05, 2.5 * float(np.median(residual[inlier])))
    corrected: np.ndarray = points * np.exp(intercept + slope * np.log(depth))[:, None]
    return corrected


def onto_floor(
    points: np.ndarray,
    floor_from_camera: np.ndarray,
    world_from_camera: np.ndarray,
    level: float,
    seen: np.ndarray,
    config: HeadFloorPriorConfig,
) -> np.ndarray:
    """Camera-frame points at most floor_tolerance_m above the floor at height `level` in floor_frame (or under it) slid
    along their rays onto it, where that lands within floor_reach_m of `seen` (world xy of lidar floor) or of the camera."""
    points = points.copy()
    in_floor = _apply(floor_from_camera, points)
    camera_z = floor_from_camera[2, 3]
    low = np.flatnonzero(
        (in_floor[:, 2] < camera_z - 1e-3) & (in_floor[:, 2] - level <= config.floor_tolerance_m)
    )
    hits = points[low] * ((level - camera_z) / (in_floor[low, 2] - camera_z))[:, None]
    distance, _ = cKDTree(np.concatenate([seen, world_from_camera[:2, 3][None]])).query(
        _apply(world_from_camera, hits)[:, :2], distance_upper_bound=config.floor_reach_m
    )
    near = np.isfinite(distance)
    points[low[near]] = hits[near]
    return points


def footprint_floor(radius: float, level: float, spacing: float) -> np.ndarray:
    """Floor points in floor_frame on a `spacing` grid over the disc of `radius` around its origin, at height `level`."""
    grid = np.arange(-radius, radius + 1e-9, spacing)
    xx, yy = np.meshgrid(grid, grid)
    inside = np.hypot(xx, yy) <= radius
    points: np.ndarray = np.c_[xx[inside], yy[inside], np.full(int(inside.sum()), level)]
    return points


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    moved: np.ndarray = points @ matrix[:3, :3].T + matrix[:3, 3]
    return moved


def r1pro_head_depth(tf_tolerance_s: float, **overrides: object) -> Blueprint:
    """Head depth anchored on Point-LIO's scans, tuned for the Orin, then put onto the floor by HeadFloorPrior;
    keywords override the Depth2DepthCloud config."""
    return autoconnect(
        Depth2DepthCloud.blueprint(
            **{
                "tf_tolerance_s": tf_tolerance_s,
                # Past this the per-pixel calibration has few lidar anchors to lean on.
                "max_range_m": 6.0,
                # The chassis lidar never sees the floor near the robot; floor it swept a few seconds ago anchors it.
                "lidar_history_s": 6.0,
                # Nothing far above the floor spends the point budget; HeadFloorPrior needs the misplaced floor to fit
                # each frame's depth correction, and cuts to its own height band after.
                "height_frame": BASE_FRAME,
                "max_height_m": HeadFloorPriorConfig.model_fields["fit_max_m"].default,
                # One point per 8x8 pixels, at most 3000: replaying r1_live_20260930_1332 on the R1, floor near the robot
                # rose from ~45% of the map at 11x11 / 1500 to 51%.
                "decimation": 8,
                "max_points": 3000,
                # JPEG decoded at 1/4 size: calibration then works on a quarter of the pixels (3 -> 9.5 fps on the Orin).
                "decode_scale": 4,
                # One lidar anchor per 4x4 px block: ~31k anchors per frame was most of the calibration time.
                "anchor_step": 4,
                **overrides,
            }
        ).remappings(
            [
                (Depth2DepthCloud, "image", "head_left_color"),
                (Depth2DepthCloud, "camera_info", "head_left_info"),
                (Depth2DepthCloud, "depth_cloud", "head_depth_raw"),
            ]
        ),
        HeadFloorPrior.blueprint(
            tf_tolerance_s=tf_tolerance_s, footprint_radius_m=R1PRO_CHASSIS_WIDTH_M / 2
        ).remappings([(HeadFloorPrior, "depth_cloud", "head_depth_raw")]),
    )
