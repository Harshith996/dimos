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

from dataclasses import replace
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.sim2.runtime import SimulationRuntime
from dimos.sim2.scene import describe_scene, load_scene, scene_robot
from dimos.sim2.scene_types import (
    SceneDescription,
    SceneEntity,
    SceneJoint,
    SceneRegion,
    SceneUpdate,
)
from dimos.sim2.sensors.lidar.models.fibonacci import Fibonacci
from dimos.sim2.sensors.spec import Camera, Imu, Lidar, Mount
from dimos.sim2.spec import ControlInterface, Joint, RobotConfig, RobotInstance, WorldConfig

pytestmark = pytest.mark.mujoco


@pytest.fixture
def world(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("""<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="box" pos=".3 0 .5"><freejoint/>
        <geom type="box" size=".05 .05 .05" mass=".1"/></body>
      <body name="cabinet" pos="1 0 0"><geom type="box" size=".1 .1 .1"/>
        <body name="door"><joint name="hinge" range="0 90"/>
          <geom type="box" size=".01 .1 .1" pos="0 .1 0"/></body></body>
    </worldbody></mujoco>""")
    robot = tmp_path / "robot.xml"
    robot.write_text("""<mujoco><worldbody><body name="base">
      <body name="tip"><joint name="j"/><geom type="sphere" size=".02" mass="1"/>
      </body></body></worldbody><actuator><position name="a" joint="j"/></actuator></mujoco>""")
    metadata = SceneDescription(
        id="test",
        entities={
            "box": SceneEntity(body="box", label="Box", kind="box", movable=True),
            "cabinet": SceneEntity(body="cabinet", label="Cabinet", kind="cabinet"),
        },
        joints={"door": SceneJoint(joint="hinge", entity="cabinet", closed=0, opened=1.5)},
        regions={
            "interior": SceneRegion(
                body="door", kind="containment", pose=Pose(0, 0.1, 0), size=(0.1, 0.1, 0.1)
            )
        },
        initial=SceneUpdate(poses={"box": Pose(0.4, 0, 0.3)}, joints={"door": 0.2}),
        spawns={"workbench": Pose(0, 0, 0.2)},
    )
    scene.with_suffix(".json").write_text(metadata.model_dump_json())
    config = RobotConfig(
        robot, "base", ControlInterface.MANIPULATOR, (Joint("j", "j", "a", mode="position"),)
    )
    runtime = SimulationRuntime(
        WorldConfig(scene, {"arm": RobotInstance(config, xyz=(0, 0, 0.2))}), uuid4().hex
    )
    try:
        yield runtime
    finally:
        runtime.close()


def test_scene_edit_and_reset_keep_model_and_restore_authored_baseline(world):
    model = world.model
    initial = world.scene_state()
    changed = world.set_scene_state(
        SceneUpdate(poses={"box": Pose(0.8, 0.1, 0.6)}, joints={"door": 1.0})
    )
    assert changed.entities["box"].pose.position.to_tuple() == pytest.approx((0.8, 0.1, 0.6))
    assert changed.joints["door"] == 1
    assert changed.generation == initial.generation + 1
    for _ in range(32):
        world.step()
        reset = world.reset()
    assert world.model is model
    assert reset.entities["box"].pose.position.to_tuple() == pytest.approx((0.4, 0, 0.3))
    assert reset.joints == {"door": 0.2}
    assert np.all(world.data.qvel == 0)
    assert reset.sim_time == 0
    assert world.snapshots.read_observation().metadata.episode_id == reset.generation


def test_invalid_edit_is_atomic(world):
    before = world.scene_state()
    with pytest.raises(ValueError, match="outside joint limits"):
        world.set_scene_state(SceneUpdate(poses={"box": Pose(2, 3, 4)}, joints={"door": 99}))
    assert world.scene_state().entities == before.entities
    assert world.episode == before.generation
    with pytest.raises(ValueError, match="fixed geometry"):
        world.set_scene_state(SceneUpdate(poses={"cabinet": Pose(0, 0, 1)}))


def test_reset_overrides_do_not_replace_baseline(world):
    override = world.reset(SceneUpdate(poses={"box": Pose(0.7, 0, 0.5)}))
    assert override.entities["box"].pose.position.x == 0.7
    assert world.reset().entities["box"].pose.position.x == 0.4


def test_region_tracks_joint_and_bounds_are_world_space(world):
    before = world.scene_state()
    after = world.set_scene_state(SceneUpdate(joints={"door": 1}))
    assert after.regions["interior"].pose.position != before.regions["interior"].pose.position
    assert before.entities["box"].bounds_min == pytest.approx((0.35, -0.05, 0.25))
    assert before.entities["box"].bounds_max == pytest.approx((0.45, 0.05, 0.35))


def test_pause_and_pose_wire_round_trip(world):
    world.set_paused(True)
    world.step()
    assert world.tick == 0
    encoded = SceneUpdate(poses={"arm": Pose(1, 2, 3)}).model_dump_json()
    world.set_scene_state(SceneUpdate.model_validate_json(encoded))
    assert world.scene_state().robots["arm"].position.to_tuple() == (1, 2, 3)
    world.set_paused(False)
    world.step()
    assert world.tick == 1


def test_missing_named_support_is_not_a_floor_default(world):
    config = world.config.robots["arm"].config
    with pytest.raises(ValueError, match="no authored 'other' support"):
        scene_robot(world.config.scene, config, "other", default=(0, 0, 0))
    assert describe_scene(world.config.scene).id == "test"


@pytest.mark.parametrize("height", [0.0, 0.6, 0.793])
def test_named_support_is_reusable_across_robot_heights(world, height):
    config = replace(world.config.robots["arm"].config, spawn_height=height)
    support = Pose(1, 2, -1.4, 0, 0, np.sin(0.4), np.cos(0.4))
    metadata = SceneDescription(id="test", spawns={"default": support})
    world.config.scene.with_suffix(".json").write_text(metadata.model_dump_json())

    robot = scene_robot(world.config.scene, config, default=(0, 0, 0))

    assert robot.xyz == pytest.approx((1, 2, -1.4 + height))
    assert robot.rpy == pytest.approx((0, 0, 0.8))
    assert describe_scene(world.config.scene).spawns["default"] == support


def test_robot_height_is_applied_to_explicit_support_default(world):
    config = replace(world.config.robots["arm"].config, spawn_height=0.6)
    world.config.scene.with_suffix(".json").unlink()

    robot = scene_robot(world.config.scene, config, default=(1, 2, -1.4))

    assert robot.xyz == pytest.approx((1, 2, -0.8))


def test_mounted_arm_root_remains_at_workbench(world):
    config = world.config.robots["arm"].config

    robot = scene_robot(world.config.scene, config, "workbench", default=(0, 0, 0))

    assert robot.xyz == pytest.approx((0, 0, 0.2))


@pytest.fixture
def sensor_world(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco><worldbody/></mujoco>")
    asset = tmp_path / "robot.xml"
    asset.write_text("""<mujoco><worldbody><body name="base">
      <geom type="sphere" size=".1"/>
      <camera name="front" pos=".1 .2 .3" fovy="42"/>
      <site name="scan" pos=".2 .3 .4"/>
      <site name="imu" pos=".3 .4 .5"/>
      <body name="arm"><joint name="joint"/>
        <geom type="sphere" size=".1" pos=".2 0 0"/></body>
    </body></worldbody><actuator><motor name="motor" joint="joint"/></actuator></mujoco>""")
    config = RobotConfig(
        model=asset,
        root_body="base",
        control=ControlInterface.WHOLE_BODY,
        joints=(Joint("joint", "joint", "motor"),),
        sensors=(
            Camera("rgb", camera="front"),
            Lidar("lidar", "scan", Fibonacci),
            Imu("imu", "imu"),
        ),
    )
    return WorldConfig(
        scene, {"left": RobotInstance(config), "right": RobotInstance(config, xyz=(1, 0, 0))}
    )


def test_named_sensor_bindings_preserve_assets_and_instance_placement(sensor_world):
    model = load_scene(sensor_world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.ncam == 2
    assert model.nsite == 4
    assert model.cam_fovy.tolist() == [42, 42]
    assert data.cam_xpos[model.camera("left/front").id] == pytest.approx((0.1, 0.2, 0.3))
    assert data.cam_xpos[model.camera("right/front").id] == pytest.approx((1.1, 0.2, 0.3))
    assert data.site_xpos[model.site("right/scan").id] == pytest.approx((1.2, 0.3, 0.4))
    for robot in ("left", "right"):
        assert model.sensor(f"{robot}/sensor/imu/gyro").objid[0] == model.site(f"{robot}/imu").id


@pytest.mark.parametrize("sensor", [Camera("rgb", "missing"), Lidar("lidar", "missing", Fibonacci)])
def test_unknown_named_sensor_never_creates_a_replacement(sensor_world, sensor):
    original = sensor_world.robots["left"].config
    robot = replace(original, sensors=(sensor, original.sensors[-1]))
    with pytest.raises(ValueError, match="unknown (camera|site) 'missing'"):
        load_scene(replace(sensor_world, robots={"left": RobotInstance(robot)}))


def test_explicit_added_camera_attaches_to_body_without_replacing_stock_camera(sensor_world):
    original = sensor_world.robots["left"].config
    robot = original.with_sensor(Camera("extra", Mount("arm", xyz=(0.2, 0, 0)), fovy=75))
    model = load_scene(replace(sensor_world, robots={"left": RobotInstance(robot)}))
    assert model.ncam == 2
    assert model.camera("left/front").fovy[0] == 42
    added = model.camera("left/sensor/extra")
    assert added.fovy[0] == 75
    assert added.bodyid[0] == model.body("left/arm").id
