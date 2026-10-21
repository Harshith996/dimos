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

from contextlib import ExitStack
import time
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.transport import ZenohTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.LineSegments3D import LineSegments3D
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sim_msgs.Contacts import Contact, Contacts
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.simulation.scene import Go2SimTelemetry, procedural_scene
from dimos.sim2.module import SimulationModule
from dimos.sim2.spec import WorldConfig
from dimos.sim2.world.scene import describe_scene
from dimos.simulation.scenes.procedural import generate


def test_procedural_world_contains_the_same_boxes_without_a_robot(tmp_path, monkeypatch):
    monkeypatch.setattr("dimos.robot.unitree.go2.simulation.scene.CACHE_DIR", tmp_path)
    path = procedural_scene(seed=7)
    scene = generate("office", 7)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nu == model.nq == 0
    assert model.ngeom == len(scene.boxes)
    assert describe_scene(path).spawns["default"].position.to_tuple() == scene.start
    ceiling = next(i for i, b in enumerate(scene.boxes) if b.kind == "ceiling")
    assert model.geom(f"ceiling_{ceiling}").group == 1
    assert procedural_scene(seed=8).read_text() != path.read_text()


@pytest.fixture
def world_stack(tmp_path, monkeypatch):
    monkeypatch.setattr("dimos.robot.unitree.go2.simulation.scene.CACHE_DIR", tmp_path)
    blueprint = autoconnect(
        SimulationModule.blueprint(
            world=WorldConfig(scene=procedural_scene(), robots={}, timestep=0.002)
        ),
        GO2Connection.blueprint(camera=False, lidar_model="mid360", publish_tf=False),
        Go2SimTelemetry.blueprint(robot_id=GO2Connection.name, rate_hz=50),
    ).remappings(
        [
            (GO2Connection.name + "_lidar", "pointcloud", "go2_scan"),
            (GO2Connection.name + "_lidar", "tf", "sim_lidar_tf"),
            (Go2SimTelemetry, "scan", "go2_scan"),
        ]
    )
    samples = {name: [] for name in ("lidar", "odometry", "contacts", "scene")}
    with ExitStack() as stack:
        transports = {}
        for name, message in (
            ("lidar", PointCloud2),
            ("odometry", Odometry),
            ("contacts", Contacts),
            ("scene", LineSegments3D),
        ):
            transport = ZenohTransport(f"go2-test/{uuid4().hex}/{name}", message)
            stack.callback(transport.stop)
            stack.callback(transport.subscribe(samples[name].append))
            transports[name, message] = transport
        blueprint = blueprint.transports(transports)
        app = ModuleCoordinator.build(
            blueprint.global_config(
                simulation="mujoco",
                transport="zenoh",
                viewer="none",
                n_workers=2,
            )
        )
        stack.callback(app.stop)
        yield app, samples


@pytest.mark.self_hosted
def test_procedural_go2_walks_with_rolling_lidar_and_existing_nav_contract(world_stack):
    app, samples = world_stack
    world = app.get_instance(SimulationModule)
    robot = app.get_instance(GO2Connection)
    start = world.status()
    positions = []
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        robot.move(Twist(Vector3(0.35, 0, 0)))
        positions.append(world.status()["robots"][GO2Connection.name])
        time.sleep(0.1)  # Command input rate.
    robot.stop_movement()
    assert world.status()["error"] is None
    assert world.status()["steps"] - start["steps"] > 3000
    assert np.asarray(positions)[:, 2].min() > 0.2
    assert positions[-1][0] - positions[0][0] > 0.8
    assert len(samples["lidar"]) > 20, (
        app.get_instance(Go2SimTelemetry).sensor_status(),
        app.get_instance(GO2Connection.name + "_lidar").sensor_status(),
        {key: getattr(t, "topic", None) for key, t in app.transports.items()},
    )
    assert {cloud.frame_id for cloud in samples["lidar"]} == {"mid360_link"}
    assert len(samples["odometry"]) > 100
    assert samples["odometry"][-1].child_frame_id == "mid360_link"
    assert any(Contact("foot", "floor") in msg.contacts for msg in samples["contacts"])
    assert len(samples["scene"][-1].segments) > 12
    assert app.get_instance(Go2SimTelemetry).sensor_status()["error"] is None
