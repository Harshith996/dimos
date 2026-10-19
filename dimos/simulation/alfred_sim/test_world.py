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

from __future__ import annotations

import math

import numpy as np
import pytest

from dimos.msgs.sim_msgs.Contacts import Contact
from dimos.simulation.alfred_sim.body import PlanarBase, alfred_spec
from dimos.simulation.alfred_sim.world import ALFRED, MOUNT_R, MOUNT_XYZ
from dimos.simulation.scenes.procedural import Scene, office
from dimos.simulation.sim_world.world import Sim

pytestmark = pytest.mark.self_hosted

STILL = np.zeros(3)


@pytest.fixture(scope="module")
def scene() -> Scene:
    return office(1)


@pytest.fixture
def sim(scene: Scene) -> Sim:
    sim = Sim(scene, 1, ALFRED, PlanarBase)
    sim.reset(*scene.start, 0.0)
    return sim


def test_a_body_frame_command_is_followed_after_turning(sim: Sim, scene: Scene) -> None:
    sim.reset(scene.start[0], scene.start[1], scene.start[2], math.pi / 2)
    before, _ = sim.base_pose()
    for _ in range(100):
        sim.tick(np.array([0.5, 0.0, 0.0]))
    after, rotation = sim.base_pose()
    assert after - before == pytest.approx([0.0, 0.975, 0.0], abs=0.01)
    assert math.atan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(math.pi / 2, abs=1e-6)
    assert sim.robot.base_velocity()[0] == pytest.approx([0.0, 0.5, 0.0], abs=1e-3)


def test_a_yaw_command_turns_the_base(sim: Sim) -> None:
    for _ in range(100):
        sim.tick(np.array([0.0, 0.0, 1.0]))
    _, rotation = sim.base_pose()
    assert math.atan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(1.95, abs=0.02)
    assert sim.robot.base_velocity()[1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-3)


def test_base_link_stays_on_the_floor_and_only_the_casters_touch_it(sim: Sim, scene: Scene) -> None:
    for _ in range(50):
        sim.tick(np.array([0.3, 0.2, 0.5]))
    base, _ = sim.base_pose()
    assert base[2] == pytest.approx(scene.params["z0"])
    assert sim.contacts() == []


def test_contacts_name_the_base_and_the_mast(sim: Sim) -> None:
    sim.reset(0.2, 3.0, sim.scene.start[2], math.pi)
    assert {Contact("base", "wall"), Contact("mast", "wall")} <= set(sim.contacts())


def test_sensor_sits_on_the_urdf_mount(sim: Sim) -> None:
    base, rotation = sim.base_pose()
    sensor, sensor_rotation = sim.sensor_pose()
    assert sensor - base == pytest.approx(rotation @ MOUNT_XYZ)
    assert sensor_rotation == pytest.approx(rotation @ MOUNT_R)
    assert sensor[2] - base[2] == pytest.approx(0.437)


def test_collision_boxes_clear_the_floor_and_reach_the_mast_top() -> None:
    model = alfred_spec().compile()
    base, mast = model.geom("base"), model.geom("mast")
    assert base.pos[2] - base.size[2] > 0.0
    assert mast.pos[2] + mast.size[2] == pytest.approx(2.011)
    assert model.body_mass[model.body("base_link").id] == pytest.approx(60.0)
