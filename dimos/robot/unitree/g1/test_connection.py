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

"""Robot-boundary contracts, plus actual GR00T/SONIC sim2 control loops."""

from contextlib import ExitStack
from pathlib import Path
import runpy
import time

import numpy as np
import pytest

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import GlobalConfig, global_config
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.g1.connection import G1Config, G1Connection, G1LocomotionSpec
from dimos.robot.unitree.g1.groot_coordinator import G1GrootCoordinator
from dimos.robot.unitree.g1.sim2 import G1_GROOT
from dimos.robot.unitree.g1.sonic_policy import G1SonicPolicyModule
from dimos.sim2.module import SimulationModule
from dimos.sim2.scene import scene_path
from dimos.spec.utils import spec_annotation_compliance


@pytest.fixture
def robot(mocker):
    robot = G1Connection(controller="groot")
    mocker.patch.object(robot, "_locomotion", mocker.Mock(spec=G1LocomotionSpec))
    yield robot
    robot.stop()


def test_custom_controller_owns_motors_not_a_second_twist_controller(robot):
    contract = robot.describe_control()
    assert list(contract.command_ports) == ["motor_command"]
    assert list(contract.description.joint_names()) == [j.name for j in G1_GROOT.joints]
    assert "odom" not in contract.reading_ports.values()
    assert {r.name for r in contract.description.resources} == {
        *(j.name.split("/", 1)[1] for j in G1_GROOT.joints),
        "imu",
    }


def test_stock_connection_describes_only_available_stock_control():
    with ExitStack() as stack:
        robot = G1Connection(controller="stock")
        stack.callback(robot.stop)
        contract = robot.describe_control()
        assert list(contract.command_ports) == ["cmd_vel"]
        assert contract.reading_ports == {}
        assert [r.name for r in contract.description.resources] == ["base"]


def test_public_move_routes_intent_without_exposing_motor_io(robot):
    robot.move(Twist(Vector3(0.4, -0.1, 0), Vector3(0, 0, 0.2)))
    robot._locomotion.set_velocity_command.assert_called_once_with(0.4, -0.1, 0.2)


def test_stock_actions_are_explicitly_unavailable_under_custom_policy(robot):
    with pytest.raises(NotImplementedError, match="stock SDK actions"):
        robot.publish_request("sport", {"api_id": 1004})


def test_policy_pd_targets_are_not_kinematic_position_limits(robot):
    limit = robot.describe_control().description.limits["g1/left_ankle_pitch/position"]
    assert limit.lo is None and limit.hi is None
    assert robot.describe_control().description.limits["g1/left_ankle_pitch/effort"].hi > 0


def test_cpu_sonic_cannot_drive_hardware():
    with ExitStack() as stack:
        policy = G1SonicPolicyModule(cpu_simulation=True, g=GlobalConfig(simulation=""))
        stack.callback(policy.stop)
        with pytest.raises(ValueError, match="not hardware"):
            policy.start()


def test_estop_is_latched_at_robot_boundary(robot):
    robot.set_estop(True)
    with pytest.raises(RuntimeError, match="latched"):
        robot.move(Twist())
    with pytest.raises(RuntimeError, match="latched"):
        robot.set_estop(False)
    robot._locomotion.set_estop.assert_called_once_with(True)


@pytest.mark.parametrize("host", [G1GrootCoordinator, G1SonicPolicyModule])
def test_policy_hosts_have_same_intent_contract(host):
    assert spec_annotation_compliance(host, G1LocomotionSpec)


def test_spawn_support_is_not_applied_twice(tmp_path):
    scene = tmp_path / "floor.xml"
    scene.write_text("<mujoco/>")
    config = G1Config(controller="groot", g=GlobalConfig(simulation="mujoco"))
    instance = G1Connection.describe_simulated_robot(config, scene)
    assert instance.xyz == (0.0, 0.0, 0.793)


@pytest.fixture
def walking_stack(request, monkeypatch):
    controller = request.param
    monkeypatch.setattr(global_config, "simulation", "mujoco")
    monkeypatch.setattr(global_config, "viewer", "none")
    monkeypatch.setattr(global_config, "scene_package", None)
    module = Path(__file__).parent / "blueprints/basic" / f"unitree_g1_{controller}_wbc.py"
    blueprint = runpy.run_path(str(module))[f"_unitree_g1_{controller}_wbc_core"]
    parsed = BlueprintConfigParser(blueprint).parse(
        [],
        environ={},
        global_overrides={
            "simulation": "mujoco",
            "scene_package": str(scene_path(None, "logistics.xml")),
            "transport": "zenoh",
            "viewer": "none",
            "n_workers": 2,
        },
    )
    with ExitStack() as stack:
        app = ModuleCoordinator.build(blueprint, parsed)
        stack.callback(app.stop)
        yield app


@pytest.mark.self_hosted
@pytest.mark.parametrize("walking_stack", ["groot", "sonic"], indirect=True)
def test_real_policy_moves_robot_through_public_connection(walking_stack):
    app = walking_stack
    world = app.get_instance(SimulationModule)
    robot = app.get_instance(G1Connection)
    start = world.status()
    positions = []
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        robot.move(Twist(Vector3(0.35, 0, 0)))
        positions.append(world.status()["robots"][G1Connection.name])
        time.sleep(0.1)  # The commanded input rate, not a readiness wait.
    robot.move(Twist())
    end = world.status()
    samples = np.array(positions)
    assert end["error"] is None
    assert end["steps"] - start["steps"] > 1200
    assert np.isfinite(samples).all()
    assert samples[:, 2].min() > 0.5
    assert samples[-1, 0] - samples[0, 0] > 0.8
    assert robot.status().last_error is None
