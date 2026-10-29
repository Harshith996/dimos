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

"""The simulated G1 binds the coordinator's joints and starts in the GR00T home pose."""

import mujoco
import pytest

from dimos.control.components import make_humanoid_joints
from dimos.robot.unitree.g1.groot_config import G1_GROOT_HOME, G1_GROOT_KD, G1_GROOT_KP
from dimos.robot.unitree.g1.simulation import G1_SIM, g1_groot_instance
from dimos.sim2.spec import WorldConfig
from dimos.sim2.world.scene import load_scene


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text(
        '<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/></worldbody></mujoco>'
    )
    return path


def test_definition_binds_every_coordinator_joint_in_order():
    assert [joint.name for joint in G1_SIM.joints] == make_humanoid_joints("g1")
    for joint in G1_SIM.joints:
        assert joint.model_name == joint.actuator == joint.name.split("/", 1)[1] + "_joint"
        assert joint.mode == "effort"
    assert [sensor.name for sensor in G1_SIM.sensors] == ["imu", "lidar", "camera"]
    assert len(G1_GROOT_KP) == len(G1_GROOT_KD) == len(G1_GROOT_HOME) == 29


def test_groot_instance_holds_the_home_pose_until_the_policy_commands(scene):
    instance = g1_groot_instance(scene)
    names = [joint.name for joint in G1_SIM.joints]
    assert instance.initial_positions == dict(zip(names, G1_GROOT_HOME, strict=True))
    assert [instance.hold_gains[name] for name in names] == list(
        zip(G1_GROOT_KP, G1_GROOT_KD, strict=True)
    )
    assert instance.control_imu == "imu"
    assert instance.wait_for_first_command
    assert instance.xyz == pytest.approx((0, 0, 0.793))


@pytest.mark.self_hosted
def test_asset_carries_the_named_mounts_and_actuators(scene):
    model = load_scene(WorldConfig(scene, {"g1": g1_groot_instance(scene)}))
    assert model.nu == 29
    for kind, name in (
        (mujoco.mjtObj.mjOBJ_SITE, "g1/control_imu"),
        (mujoco.mjtObj.mjOBJ_SITE, "g1/mid360_link"),
        (mujoco.mjtObj.mjOBJ_CAMERA, "g1/front_camera"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "g1/left_hip_pitch_joint"),
    ):
        assert mujoco.mj_name2id(model, kind, name) >= 0, name
