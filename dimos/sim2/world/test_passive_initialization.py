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

"""Passive coordinates reset with the robot without becoming controllable joints."""

from collections.abc import Callable
from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dimos.sim2.spec import WorldConfig
from dimos.sim2.world.runtime import SimulationRuntime


@pytest.fixture
def passive_world_config(world_config: WorldConfig) -> WorldConfig:
    instance = world_config.robots["left"]
    tree = ET.parse(instance.config.model)
    root = tree.getroot()
    base = root.find("./worldbody/body")
    assert base is not None
    finger = ET.SubElement(base, "body", name="passive_finger_body", pos="0 0 .7")
    ET.SubElement(
        finger,
        "joint",
        name="passive_finger",
        type="slide",
        axis="1 0 0",
        range="-.04 0",
        limited="true",
    )
    ET.SubElement(finger, "geom", type="box", size=".02 .02 .04", mass=".1")
    hinge = ET.SubElement(base, "body", name="passive_hinge_body", pos="0 0 .9")
    ET.SubElement(hinge, "joint", name="passive_hinge", type="hinge", limited="false")
    ET.SubElement(hinge, "geom", type="sphere", size=".02", mass=".1")
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality, "joint", joint1="passive_finger", joint2="finger_joint", polycoef="0 -1 0 0 0"
    )
    tree.write(instance.config.model, encoding="unicode")
    return replace(
        world_config,
        robots={
            "left": replace(
                instance,
                initial_positions={"shoulder": 0.1, "gripper": 425},
                passive_joint_positions={"passive_finger": -0.02, "passive_hinge": 0.3},
            )
        },
    )


def test_passive_home_and_native_targets_survive_repeated_reset(
    worlds: Callable[..., SimulationRuntime],
    passive_world_config: WorldConfig,
) -> None:
    world = worlds(config=passive_world_config)
    names = ("left/finger_joint", "left/passive_finger", "left/passive_hinge")
    qpos = [int(world.model.joint(name).qposadr[0]) for name in names]
    dofs = [int(world.model.joint(name).dofadr[0]) for name in names]
    actuator = world.model.actuator("left/finger").id
    assert world.robots["left"].channel.descriptor.joint_names == ("shoulder", "gripper")
    np.testing.assert_array_equal(world.data.qpos[qpos], [0.02, -0.02, 0.3])
    assert world.data.ctrl[actuator] == pytest.approx(0.02)
    for _ in range(2):
        episode = world.episode
        world.data.qpos[qpos] = [0.01, -0.01, -0.2]
        world.data.qvel[dofs] = [1, -1, 2]
        world.data.ctrl[actuator] = 0.01
        world.reset()
        np.testing.assert_array_equal(world.data.qpos[qpos], [0.02, -0.02, 0.3])
        np.testing.assert_array_equal(world.data.qvel[dofs], [0, 0, 0])
        assert world.data.ctrl[actuator] == pytest.approx(0.02)
        assert world.episode == episode + 1


def test_passive_initialization_is_scoped_to_its_robot(
    worlds: Callable[..., SimulationRuntime],
    passive_world_config: WorldConfig,
) -> None:
    left = passive_world_config.robots["left"]
    config = replace(
        passive_world_config,
        robots={
            "left": left,
            "right": replace(
                left,
                xyz=(2, 0, 0.3),
                initial_positions={"shoulder": 0.1, "gripper": 637.5},
                passive_joint_positions={"passive_finger": -0.01, "passive_hinge": -0.4},
            ),
        },
    )
    world = worlds(config=config)
    left_qpos = int(world.model.joint("left/passive_finger").qposadr[0])
    right_qpos = int(world.model.joint("right/passive_finger").qposadr[0])
    np.testing.assert_array_equal(world.data.qpos[[left_qpos, right_qpos]], [-0.02, -0.01])
    world.data.qpos[[left_qpos, right_qpos]] = 0
    world.reset()
    np.testing.assert_array_equal(world.data.qpos[[left_qpos, right_qpos]], [-0.02, -0.01])


@pytest.mark.parametrize(
    "name", ["", " ", " shoulder", "shoulder", "hinge", "gripper", "finger_joint"]
)
def test_passive_keys_reject_empty_or_controlled_names(
    world_config: WorldConfig, name: str
) -> None:
    with pytest.raises(ValueError, match="nonempty robot-local|overlaps a controlled"):
        replace(world_config.robots["left"], passive_joint_positions={name: 0.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_passive_initial_positions_require_finite_values(
    world_config: WorldConfig, value: float
) -> None:
    with pytest.raises(ValueError, match="finite initial position"):
        replace(world_config.robots["left"], passive_joint_positions={"passive_finger": value})


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("missing", 0, "unknown passive joint"),
        ("right/passive_finger", 0, "unknown passive joint"),
        ("passive_finger", -0.05, "outside model joint limits"),
        ("passive_finger", 0.01, "outside model joint limits"),
    ],
)
def test_passive_model_name_and_range_are_validated(
    worlds: Callable[..., SimulationRuntime],
    passive_world_config: WorldConfig,
    name: str,
    value: float,
    error: str,
) -> None:
    instance = replace(passive_world_config.robots["left"], passive_joint_positions={name: value})
    with pytest.raises(ValueError, match=error):
        worlds(config=replace(passive_world_config, robots={"left": instance}))


def test_unbound_joint_actuator_cannot_be_initialized_as_passive(
    worlds: Callable[..., SimulationRuntime],
    passive_world_config: WorldConfig,
) -> None:
    instance = passive_world_config.robots["left"]
    tree = ET.parse(instance.config.model)
    actuators = tree.getroot().find("actuator")
    assert actuators is not None
    ET.SubElement(actuators, "motor", name="unbound_motor", joint="passive_finger")
    tree.write(instance.config.model, encoding="unicode")
    with pytest.raises(ValueError, match="has a joint actuator"):
        worlds(config=passive_world_config)


def test_ball_joint_cannot_take_a_scalar_passive_initial_position(
    worlds: Callable[..., SimulationRuntime],
    passive_world_config: WorldConfig,
) -> None:
    instance = passive_world_config.robots["left"]
    tree = ET.parse(instance.config.model)
    joint = tree.getroot().find(".//joint[@name='passive_hinge']")
    assert joint is not None
    joint.set("type", "ball")
    tree.write(instance.config.model, encoding="unicode")
    with pytest.raises(ValueError, match="must be scalar"):
        worlds(config=passive_world_config)


def test_free_root_cannot_take_a_scalar_passive_initial_position(
    worlds: Callable[..., SimulationRuntime],
    passive_world_config: WorldConfig,
) -> None:
    instance = passive_world_config.robots["left"]
    tree = ET.parse(instance.config.model)
    base = tree.getroot().find("./worldbody/body")
    assert base is not None
    ET.SubElement(base, "freejoint", name="root_free")
    tree.write(instance.config.model, encoding="unicode")
    instance = replace(
        instance,
        config=replace(instance.config, floating=True),
        passive_joint_positions={"root_free": 0.0},
    )
    with pytest.raises(ValueError, match="must be scalar"):
        worlds(config=replace(passive_world_config, robots={"left": instance}))
