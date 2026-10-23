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

"""3d navigation on a Go2's onboard L1 lidar that follows GSC PGO's loop closures.

GSC corrects odometry into ``map`` (tf map -> world). The ray tracer and the MLS
planner both work in ``map`` and warp themselves through each loop_closure_event
with fast-reform, so the map, the planner graph and the goal stay corrected
without a rebuild. Click a goal in the viewer, or drive with the viewer's keys.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.ray_tracing.reform_module import ReformRayTracingVoxelMap
from dimos.navigation.global_planner.mls_planner.mls_reform_planner_native import (
    MLSReformPlannerNative,
)
from dimos.navigation.global_planner.viz import nav_static, nav_visual_override
from dimos.navigation.jnav.components.loop_closure.gsc_pgo.module import GscPGO
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.trajectory_follower.basic.module import BasicPathFollower
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import rerun_config
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.constants import ROBOT_HEIGHT, ROBOT_LENGTH, ROBOT_WIDTH
from dimos.robot.unitree.go2.nav_3d_config import (
    NAV_VIEWER_TOPICS,
    mls_planner_config,
    ray_tracing_config,
    voxel_size,
    wall_clearance_m,
)
from dimos.robot.unitree.go2.normalize_go2_lidar import NormalizeGo2Lidar
from dimos.visualization.vis_module import vis_module

planner_viz_hz = 10.0

_rerun_config = {
    **rerun_config,
    "topics": list(NAV_VIEWER_TOPICS),
    "memory_limit": "64MB",
    "static": nav_static(ROBOT_LENGTH, ROBOT_WIDTH, ROBOT_HEIGHT, wall_clearance_m),
    "visual_override": {
        **rerun_config["visual_override"],
        "world/camera_info": None,
        **nav_visual_override(planner_viz_hz, voxel_size, wall_clearance_m),
    },
}

unitree_go2_mls_reform = autoconnect(
    vis_module(viewer_backend=global_config.viewer, rerun_config=_rerun_config),
    GO2Connection.blueprint(),
    # The L1 cloud arrives accumulated in world; this re-expresses it in base_link.
    NormalizeGo2Lidar.blueprint(),
    GscPGO.blueprint(child_frame_id="world"),
    ReformRayTracingVoxelMap.blueprint(
        **ray_tracing_config.model_copy(
            # map -> world comes from GSC once per cloud it processes (~0.13 s apart, later when busy).
            update={"world_frame": "map", "tf_match_tolerance_s": 0.5, "tf_wait_timeout_s": 1.0}
        ).model_dump(exclude_unset=True)
    ).remappings(
        [
            (ReformRayTracingVoxelMap, "lidar", "cloud"),
        ]
    ),
    MLSReformPlannerNative.blueprint(
        **mls_planner_config.model_copy(
            # 8 workers: on each loop closure the planner rebuilds its whole graph.
            update={"world_frame": "map", "viz_publish_hz": planner_viz_hz, "worker_threads": 8}
        ).model_dump(exclude_unset=True)
    ).remappings(
        [
            (MLSReformPlannerNative, "global_map", "global_map_unused"),
        ]
    ),
    BasicPathFollower.blueprint(
        world_frame="map", heading_gain=1.0, lookahead_time_s=2.5, min_lookahead_m=1.2
    ),
    MovementManager.blueprint(),
).global_config(n_workers=10, robot_model="unitree_go2", obstacle_avoidance=False)
