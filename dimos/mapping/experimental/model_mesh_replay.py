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

"""ModelMesh on a recorded walk instead of the robot.

    dimos --replay-db <walk.db> run go2-nav-model-replay --map-file=<map dir>

The walk's pointlio_lidar and tf replay at recorded rate into the ray tracer, whose local map
ModelMesh relocalizes; the model lands on ``world/mesh`` once a fix does. No planner, no robot.
"""

import os
import sys
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.experimental.mesh import MESH_ENTITY, MeshColours
from dimos.mapping.experimental.model_mesh import ModelMesh
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.memory.replay_module import dataset_path, replay_module
from dimos.robot.unitree.go2.nav_3d_config import ray_tracing_config
from dimos.visualization.vis_module import vis_module

_DATASET = dataset_path(global_config.replay_db, explicit="go2-nav-model-replay" in sys.argv[1:])
if _DATASET:
    # workers import this with the default config; the environment crosses the fork
    os.environ["REPLAY_DB"] = _DATASET

Replay = replay_module(_DATASET, topics="pointlio_lidar,tf")


def _view() -> Any:
    """The 3D view with the ray tracer's whole-map output off; the recorded premap starts hidden."""
    import rerun.blueprint as rrb

    hidden = ("world/global_map", "world/loaded_map", "world/region_bounds")
    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            overrides={path: rrb.EntityBehavior(visible=False) for path in hidden},
        ),
    )


go2_nav_model_replay = autoconnect(
    Replay.blueprint(dataset=_DATASET),
    # the recorded map only places the model here; it does not seed the live map
    RayTracingVoxelMap.blueprint(
        **ray_tracing_config.model_copy(update={"world_frame": "odom"}).model_dump(
            exclude_unset=True
        )
    ).remappings(
        [
            (RayTracingVoxelMap, "lidar", "pointlio_lidar"),
            (RayTracingVoxelMap, "loaded_map", "loaded_map_unused"),
        ]
    ),
    ModelMesh.blueprint(world_frame="odom", republish_loaded_map=0.0),
    vis_module(
        global_config.viewer,
        rerun_config={"visual_override": {MESH_ENTITY: MeshColours(alpha=1.0)}, "blueprint": _view},
    ),
).global_config(n_workers=5)
