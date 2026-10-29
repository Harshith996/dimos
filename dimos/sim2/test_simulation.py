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

from pathlib import Path
import time

import mujoco
import pytest

from dimos.core.module import Module, ModuleConfig
from dimos.sim2.module import Simulation
from dimos.sim2.spec import Mount, SimulatedRobot, SimulatedSensor
from dimos.sim2.state import SimClient
from dimos.sim2.world import compile_world


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text("""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="shelf" pos="2 0 1"/>
  </worldbody>
</mujoco>
""")
    return path


@pytest.fixture
def robot_model(tmp_path: Path) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text("""
<mujoco>
  <worldbody>
    <body name="base" pos="9 9 9">
      <freejoint/>
      <geom type="box" size=".2 .1 .1" mass="1"/>
      <body name="head" pos=".2 0 .1">
        <joint name="neck" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass=".1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="neck" joint="neck"/></actuator>
  <keyframe><key name="home" qpos="0 0 0 1 0 0 0 0"/></keyframe>
</mujoco>
""")
    return path


def test_robots_and_sensors_compile_into_one_world(scene, robot_model):
    model = compile_world(
        scene,
        [
            SimulatedRobot("left", robot_model, "base", xyz=(-1, 0, 0.3)),
            SimulatedRobot("right", robot_model, "base", xyz=(1, 0, 0.3)),
            SimulatedSensor("camera", "camera", Mount("head"), robot_id="left"),
            SimulatedSensor("lidar", "lidar", Mount("base", xyz=(0, 0, 0.2)), robot_id="right"),
            SimulatedSensor("overhead", "camera", Mount("world", xyz=(0, 0, 3))),
        ],
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.body("left/base").xpos.tolist() == pytest.approx([-1, 0, 0.3])
    assert data.body("right/base").xpos.tolist() == pytest.approx([1, 0, 0.3])
    assert model.joint("right/neck") is not None
    assert model.camera("left/camera") is not None
    assert model.camera("world/overhead") is not None
    assert data.site("right/lidar").xpos.tolist() == pytest.approx([1, 0, 0.5])


@pytest.mark.parametrize(
    "described, error",
    [
        ([SimulatedSensor("imu", "imu", Mount("base"), robot_id="ghost")], "unknown robots"),
        ([SimulatedSensor("imu", "imu", Mount("nowhere"))], "no body 'nowhere'"),
    ],
)
def test_invalid_descriptions_are_rejected(scene, described, error):
    with pytest.raises(ValueError, match=error):
        compile_world(scene, described)


def test_duplicate_robot_ids_are_rejected(scene, robot_model):
    robot = SimulatedRobot("dog", robot_model, "base")
    with pytest.raises(ValueError, match="duplicate"):
        compile_world(scene, [robot, robot])


class _SimRobotConfig(ModuleConfig):
    model: Path
    x: float = 0.0


class _SimRobot(Module):
    config: _SimRobotConfig

    @staticmethod
    def describe_simulated(config: _SimRobotConfig) -> SimulatedRobot:
        return SimulatedRobot("dog", config.model, "base", xyz=(config.x, 0, 0))


class _Bystander(Module):
    pass


def test_autoconnect_collects_descriptions_from_atoms(scene, robot_model):
    blueprint = Simulation.autoconnect(
        _SimRobot.blueprint(model=robot_model, x=2.5), _Bystander.blueprint(), scene=scene
    )
    world = next(a for a in blueprint.active_blueprints if a.module is Simulation)
    assert world.kwargs["simulated"] == {
        "_simrobot": SimulatedRobot("dog", robot_model, "base", xyz=(2.5, 0, 0))
    }


def test_simulation_builds_and_describes_its_world(scene, robot_model):
    simulation = Simulation(
        scene=scene,
        simulated={
            "dog": SimulatedRobot("dog", robot_model, "base"),
            "cam": SimulatedSensor("cam", "camera", Mount("head"), robot_id="dog"),
        },
    )
    try:
        simulation.build()
        description = simulation.describe()
        assert description["robots"] == ["dog"]
        assert description["sensors"] == ["dog/cam"]
        assert Path(description["model"]).is_file()
    finally:
        simulation.stop()
    assert not Path(description["model"]).exists()


def test_simulation_steps_physics_and_takes_commands(scene, robot_model):
    simulation = Simulation(
        scene=scene, simulated={"dog": SimulatedRobot("dog", robot_model, "base", xyz=(0, 0, 1))}
    )
    try:
        simulation.build()
        simulation.start()
        client = SimClient(simulation.describe())
        client.write(client.actuator_ids(["dog/neck"]), [1.0])
        time.sleep(0.5)
        data = client.read()
        assert data.time > 0.3
        assert data.body("dog/base").xpos[2] < 0.2
        assert data.joint("dog/neck").qvel[0] > 1.0
    finally:
        simulation.stop()
