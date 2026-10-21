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

"""Shared G1 mapping and navigation, including the sensor-to-mapper wiring."""

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.hardware.sensors.lidar.pointlio.pointlio_blueprints import mid360_for_pointlio
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import HeightCostConfig
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.navigation.go2.replanning_a_star.module import ReplanningAStarPlanner
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.g1.config import G1


def g1_navigation(*, simulated: bool) -> Blueprint:
    """Use simulated world-frame clouds or the real Mid360/Point-LIO pipeline."""
    if simulated:
        mapping = VoxelGridMapper.blueprint(emit_every=1).remappings(
            [(VoxelGridMapper, "lidar", "pointcloud")]
        )
        resolution = 0.05
    else:
        # The Orin raytracer cannot sustain the simulation's 5 cm resolution.
        resolution = 0.08
        mapping = autoconnect(
            mid360_for_pointlio(),
            PointLio.blueprint(),
            RayTracingVoxelMap.blueprint(
                voxel_size=resolution,
                emit_every=0,
                global_emit_every=4,
                max_health=10,
                graze_cos=0.85,
            ),
        )
    return autoconnect(
        mapping,
        CostMapper.blueprint(
            config=HeightCostConfig(
                resolution=resolution,
                can_pass_under=G1.height_clearance + 0.2,
                can_climb=0.10,
            ),
            initial_safe_radius_meters=G1.width_clearance + 0.6,
        ),
        ReplanningAStarPlanner.blueprint(
            robot_width=G1.width_clearance,
            robot_rotation_diameter=0.8,
        ),
        MovementManager.blueprint(),
    )
