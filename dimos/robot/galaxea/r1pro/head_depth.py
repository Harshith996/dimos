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
    # A head point no higher than this above the lidar's floor is floor.
    floor_tolerance_m: float = 0.05
    # ...and so is its neighbour in the image while log(floor depth / depth) changes less than this between them.
    floor_smoothness: float = 0.04
    # Kept from the floor to just above the lidar's plane, what the lidar sees worst.
    min_height_m: float = -0.15
    max_height_m: float = 0.35


class HeadFloorPrior(Module):
    """Head depth onto the lidar's floor: a point the calibration puts at or under the floor, where the lidar saw floor
    within floor_reach_m (or the camera stands over it), slides along its camera ray onto the floor, and so do its
    neighbours that keep the floor's shape (grow_floor). Depth Anything often reads the near floor too far, and the chassis
    lidar never sees the floor right in front of the robot. Every pose comes from tf at the cloud's stamp, so the torso
    and base may move."""

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
        in_floor = _apply(floor_from_camera, points)
        camera_z = floor_from_camera[:3, 3][2]
        scans = [
            scan for scan in self._floor_scans if scan[0] >= cloud.ts - self.config.lidar_history_s
        ]
        if scans:
            level = float(np.median([scan[2] for scan in scans]))
            seen = np.concatenate([scan[1] for scan in scans] + [world_from_camera[:2, 3][None]])
            # Rays heading down: how far each must stretch (or shrink) to end on the floor, and where it would.
            down = np.flatnonzero(in_floor[:, 2] < camera_z - 1e-3)
            scale = (level - camera_z) / (in_floor[down, 2] - camera_z)
            hits = points[down] * scale[:, None]
            distance, _ = cKDTree(seen).query(
                _apply(world_from_camera, hits)[:, :2],
                distance_upper_bound=self.config.floor_reach_m,
            )
            near = np.isfinite(distance)
            down, scale, hits = down[near], scale[near], hits[near]
            floor_like = grow_floor(
                points[down],
                np.log(scale),
                in_floor[down, 2] - level,
                self.config.floor_tolerance_m,
                self.config.floor_smoothness,
            )
            points[down[floor_like]] = hits[floor_like]
            in_floor[down[floor_like], 2] = level
        keep = (in_floor[:, 2] >= self.config.min_height_m) & (
            in_floor[:, 2] <= self.config.max_height_m
        )
        self.head_cloud.publish(
            PointCloud2.from_numpy(
                points[keep].astype(np.float32), frame_id=cloud.frame_id, timestamp=cloud.ts
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


def grow_floor(
    points: np.ndarray,
    log_scale: np.ndarray,
    height: np.ndarray,
    tolerance_m: float,
    smoothness: float,
) -> np.ndarray:
    """Which camera-frame points are floor: those at most tolerance_m over it, grown over neighbouring rays in the image
    while log(floor depth / depth) changes less than `smoothness` from one to the next. An obstacle's face breaks it."""
    floor_like = height <= tolerance_m
    if len(points) < 2 or not floor_like.any():
        return floor_like
    image = points[:, :2] / points[:, 2:3]
    tree = cKDTree(image)
    spacing = float(np.median(tree.query(image, k=2)[0][:, 1]))
    neighbours: list[list[int]] = [[] for _ in range(len(points))]
    for i, j in tree.query_pairs(1.5 * spacing):
        if abs(log_scale[i] - log_scale[j]) < smoothness:
            neighbours[i].append(j)
            neighbours[j].append(i)
    frontier = list(np.flatnonzero(floor_like))
    while frontier:
        for n in neighbours[frontier.pop()]:
            if not floor_like[n]:
                floor_like[n] = True
                frontier.append(n)
    return floor_like


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


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
                # Nothing above the lidar's plane spends the point budget; what is under the floor goes to HeadFloorPrior.
                "height_frame": BASE_FRAME,
                "max_height_m": HeadFloorPriorConfig.model_fields["max_height_m"].default,
                # The ray tracer shares the Orin with everything else; one point per 11x11 pixels, at most 1500, keeps
                # it near one Point-LIO's worth of work with every lidar ray traced.
                "decimation": 11,
                "max_points": 1500,
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
        HeadFloorPrior.blueprint(tf_tolerance_s=tf_tolerance_s).remappings(
            [(HeadFloorPrior, "depth_cloud", "head_depth_raw")]
        ),
    )
