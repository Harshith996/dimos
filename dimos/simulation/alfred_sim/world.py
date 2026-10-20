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

"""Alfred and its Mid-360 in the simulated world."""

from __future__ import annotations

from functools import partial

from dimos.hardware.sensors.lidar.pointlio.module import PointLioConfig
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevelConfig
from dimos.simulation.alfred_sim.body import BASE_BODY, PARTS, alfred_spec, mid360_mount
from dimos.simulation.scenes.procedural import Scene
from dimos.simulation.sensors.mid360.lidar import SimMid360
from dimos.simulation.sim_world.body import BodyShape, Rig
from dimos.simulation.sim_world.planar import PlanarBase, tune_velocity_gains
from dimos.simulation.sim_world.world import Sim, SimWorld

MOUNT_XYZ, MOUNT_R = mid360_mount()
HALF_WIDTH = 0.27
HEIGHT = 1.86
STEP = 0.02

SHAPE = BodyShape(
    half_width=HALF_WIDTH,
    headroom=HEIGHT,
    step=STEP,
    base_height=0.0,
    top_center=(0.0, 0.0, HEIGHT / 2),
    top_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    top_half_extents=(HALF_WIDTH, HALF_WIDTH, HEIGHT / 2),
)

ALFRED_BODY = partial(PlanarBase, base_body=BASE_BODY, parts=PARTS)

ALFRED = Rig(
    spec=alfred_spec,
    base_body=BASE_BODY,
    mount_xyz=MOUNT_XYZ,
    mount_r=MOUNT_R,
    lidar=SimMid360.fitted,
    command_timeout=AlfredHighLevelConfig.model_fields["cmd_vel_timeout"].default,
    shape=SHAPE,
    tune=tune_velocity_gains,
    lio_blind_range=PointLioConfig.model_fields["blind"].default,
)


class SimAlfredWorld(SimWorld):
    """A generated scene with Alfred and its simulated Mid-360, paced to the wall clock."""

    def _build(self, scene: Scene, seed: int) -> Sim:
        return Sim(scene, seed, ALFRED, ALFRED_BODY)
