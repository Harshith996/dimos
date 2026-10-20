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
from pathlib import Path
import time

import mujoco
import numpy as np
import pytest

from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sim_msgs.Contacts import Contact
from dimos.simulation.alfred_sim.body import PARTS, alfred_spec
from dimos.simulation.alfred_sim.world import (
    ALFRED,
    ALFRED_BODY,
    MOUNT_R,
    MOUNT_XYZ,
    SimAlfredWorld,
)
from dimos.simulation.scenes.procedural import Scene, block, office
from dimos.simulation.sim_world.world import Sim

pytestmark = pytest.mark.self_hosted

STILL = np.zeros(3)


@pytest.fixture(scope="module")
def scene() -> Scene:
    return office(1)


@pytest.fixture
def sim(scene: Scene) -> Sim:
    sim = Sim(scene, 1, ALFRED, ALFRED_BODY)
    sim.reset(*scene.start, 0.0)
    return sim


def test_a_body_frame_command_is_followed_after_turning(sim: Sim, scene: Scene) -> None:
    sim.reset(scene.start[0], scene.start[1], scene.start[2], math.pi / 2)
    before, _ = sim.base_pose()
    for _ in range(100):
        sim.tick(np.array([0.5, 0.0, 0.0]))
    after, rotation = sim.base_pose()
    assert after - before == pytest.approx([0.0, 0.975, 0.0], abs=0.02)
    assert math.atan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(math.pi / 2, abs=0.02)
    assert sim.robot.base_velocity()[0] == pytest.approx([0.0, 0.5, 0.0], abs=0.01)


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


def test_contacts_name_the_robot_part(sim: Sim) -> None:
    sim.reset(0.2, 3.0, sim.scene.start[2], 0.0)
    assert Contact("base", "wall") in sim.contacts()


def test_sensor_sits_on_the_urdf_mount(sim: Sim) -> None:
    base, rotation = sim.base_pose()
    sensor, sensor_rotation = sim.sensor_pose()
    assert sensor - base == pytest.approx(rotation @ MOUNT_XYZ)
    assert sensor_rotation == pytest.approx(rotation @ MOUNT_R)
    assert sensor[2] - base[2] == pytest.approx(0.376, abs=1e-3)


def test_the_lidar_never_reports_the_robot_itself(sim: Sim) -> None:
    frame = None
    while frame is None:
        frame = sim.tick(STILL)
    assert len(frame.points) > 10_000
    assert np.linalg.norm(frame.points, axis=1).min() >= ALFRED.lio_blind_range


def test_the_model_clears_the_floor_and_stands_under_the_door_lintel() -> None:
    model = alfred_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottom, top = np.inf, -np.inf
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            reach = np.abs(data.geom_xmat[geom].reshape(3, 3))[2] @ model.geom_size[geom]
            bottom = min(bottom, data.geom_xpos[geom][2] - reach)
            top = max(top, data.geom_xpos[geom][2] + reach)
    assert bottom > 0.0
    assert top == pytest.approx(1.859, abs=1e-3)
    assert {model.geom(geom).name.rsplit("_", 1)[0] for geom in range(model.ngeom)} == set(PARTS)


def test_the_world_swaps_scenes_and_hands_out_its_body_premap_and_model(tmp_path: Path) -> None:
    world = SimAlfredWorld()
    world.cmd_vel.transport = LCMTransport("/test_alfred_sim_world/cmd_vel", Twist)
    poses: list[PoseStamped] = []
    maps: list[PointCloud2] = []
    world.ground_truth.subscribe(poses.append)
    world.loaded_map.subscribe(maps.append)
    world.start()
    try:
        scene = block(2)
        world.load_scene("block", 2, {})
        swapped = time.time()
        deadline = swapped + 10.0
        while not [p for p in poses if p.ts >= swapped] and time.time() < deadline:
            time.sleep(0.05)
        after = next(p for p in poses if p.ts >= swapped)
        assert (after.position.x, after.position.y) == pytest.approx(scene.start[:2], abs=0.05)
        assert world.body() == ALFRED.shape
        x, y, z0 = scene.start
        count = world.publish_premap(np.array([[x, y, z0], [x + 1.0, y, z0]]), True)
        assert count > 1_000 and len(maps) == 1 and len(maps[0].points_f32()) == count
        world.export_model(tmp_path / "robot.xml")
        model = mujoco.MjSpec.from_file(str(tmp_path / "robot.xml")).compile()
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.nu == 0
    finally:
        world.stop()
