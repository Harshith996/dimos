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

"""Alfred's navigation fragment: a cloud to map from through to wheel commands.

Carries no cloud or odometry source, so it is composed into a blueprint that supplies
both rather than run on its own.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.navigation.experimental.dannav.holonomic_tc.module import DanHolonomicTC
from dimos.navigation.experimental.dannav.local_planner.module import DanLocalPlanner
from dimos.navigation.global_planner.mls_planner.mls_planner_native import MLSPlannerNative
from dimos.navigation.global_planner.mls_planner.start_relay import StartRelay
from dimos.navigation.global_planner.mls_planner.viz import planner_visual_override
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.diy.alfred.config import ALFRED, ALFRED_URDF
from dimos.visualization.rerun.urdf_robot import UrdfRobotStaticRerunFactory

ALFRED_RERUN_ROOT = "world/alfred"

VOXEL_SIZE_METERS = 0.05
WALL_CLEARANCE_M = 0.2
PLANNER_VIZ_HZ = 2.0
MID360_MAX_RANGE_METERS = 15.0
"""The Mid-360 reaches far past this, but the voxel map raytraces every point from the
sensor origin, so the far returns cost the most and carve the least reliable space."""


def _path_colored(msg: Any, color: tuple[int, int, int]) -> Any:
    return msg.to_rerun(color=color)


def _alfred_urdf_static(rr: Any) -> list[tuple[str, Any]]:
    factory = UrdfRobotStaticRerunFactory(urdf_path=ALFRED_URDF, root_path=ALFRED_RERUN_ROOT)
    return [
        *factory(rr),
        (ALFRED_RERUN_ROOT, rr.Transform3D(parent_frame="tf#/base_link")),
    ]


def _alfred_rerun_config() -> dict[str, Any]:
    """The rerun bridge config every Alfred navigation blueprint shares."""
    return {
        "static": {ALFRED_RERUN_ROOT: _alfred_urdf_static},
        # Keyed by the topic's entity path, before any visual_override renames it.
        # Everything is held to 1 Hz: the viewer rides a wifi link, and the
        # uncapped clouds put it seconds behind live. tf is capped too, so the
        # whole scene ticks once a second rather than re-posing per odom update.
        "max_hz": {
            "world/tf": 1.0,
            "world/lidar": 1.0,
            "world/global_map": 1.0,
            "world/local_map": 1.0,
            "world/nodes": 1.0,
        },
        "visual_override": {
            # The MLS planner path; the local planner's path keeps the stock green.
            "world/planner_path": partial(_path_colored, color=(170, 60, 220)),
            # The planner publishes its surface and edges by cell.
            **planner_visual_override(PLANNER_VIZ_HZ, VOXEL_SIZE_METERS, WALL_CLEARANCE_M),
        },
    }


def _alfred_nav(*, map_cloud_topic: str, map_max_range: float) -> Blueprint:
    """Mapping, planning and following. map_cloud_topic is what the voxel map raytraces."""
    return autoconnect(
        RayTracingVoxelMap.blueprint(
            voxel_size=VOXEL_SIZE_METERS,
            max_range=map_max_range,
        ).remappings([(RayTracingVoxelMap, "lidar", map_cloud_topic)]),
        MLSPlannerNative.blueprint(
            voxel_size=VOXEL_SIZE_METERS,
            robot_height=ALFRED.body_height,
            wall_clearance_m=WALL_CLEARANCE_M,
            step_penalty_weight=1.0,
            viz_publish_hz=PLANNER_VIZ_HZ,
        ).remappings([(MLSPlannerNative, "path", "planner_path")]),
        # Solely the tf-driven start_pose source for the dannav odom remaps below.
        StartRelay.blueprint(),
        DanLocalPlanner.blueprint(resample_spacing_m=0.1).remappings(
            [(DanLocalPlanner, "odom", "start_pose")]
        ),
        DanHolonomicTC.blueprint().remappings([(DanHolonomicTC, "odom", "start_pose")]),
        MovementManager.blueprint(),
    )
