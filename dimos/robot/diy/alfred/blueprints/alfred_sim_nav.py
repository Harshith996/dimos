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

"""Alfred's lidar-only navigation driving the simulated Alfred to a clicked goal.

dimos run alfred-sim-nav
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.diy.alfred.blueprints.alfred_nav import (
    MID360_MAX_RANGE_METERS,
    _alfred_nav,
    _alfred_rerun_config,
)
from dimos.robot.diy.alfred.mount_tf import AlfredMountTf
from dimos.simulation.alfred_sim.world import SimAlfredWorld
from dimos.visualization.vis_module import vis_module

if TYPE_CHECKING:
    from rerun._baseclasses import Archetype

    from dimos.msgs.nav_msgs.LineSegments3D import LineSegments3D


def _scene_lines(scene: LineSegments3D) -> Archetype:
    return scene.to_rerun(radii=0.01)


_rerun_config = _alfred_rerun_config()

alfred_sim_nav = autoconnect(
    SimAlfredWorld.blueprint(),
    AlfredMountTf.blueprint(root_frame="mid360_link"),
    _alfred_nav(map_cloud_topic="lidar", map_max_range=MID360_MAX_RANGE_METERS),
    vis_module(
        global_config.viewer,
        rerun_config={
            **_rerun_config,
            "visual_override": {**_rerun_config["visual_override"], "world/scene": _scene_lines},
        },
    ),
    # zenoh's pending-connection bug keeps natives spawned together from linking with gossip on
).global_config(transport="zenoh", zenoh_gossip=False, n_workers=10, robot_model="alfred")
