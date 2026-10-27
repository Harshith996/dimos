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

"""The simulated xArm blueprints size their MujocoSimModule by arm joints only."""

import json
import os
import platform
import subprocess
import sys

import pytest

# The tests extra excludes yourdfpy on Linux ARM because embreex has no wheel.
if sys.platform == "linux" and platform.machine() == "aarch64":
    pytest.importorskip(
        "yourdfpy", reason="yourdfpy is unavailable in the Linux ARM test environment"
    )

# Blueprints compose at import time under the global config, so the simulated
# variants only exist in a process started with SIMULATION set.
_PROBE = """
import json

from dimos.control.coordinator import ControlCoordinator
from dimos.robot.manipulators.xarm.blueprints.basic import coordinator_xarm6, coordinator_xarm7
from dimos.robot.manipulators.xarm.blueprints.grasp import xarm6_grasp, xarm_grasp
from dimos.robot.manipulators.xarm.blueprints.teleop import (
    coordinator_teleop_xarm6,
    coordinator_teleop_xarm7,
)
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule

blueprints = {
    "coordinator_xarm6": coordinator_xarm6,
    "coordinator_xarm7": coordinator_xarm7,
    "coordinator_teleop_xarm6": coordinator_teleop_xarm6,
    "coordinator_teleop_xarm7": coordinator_teleop_xarm7,
    "xarm6_grasp": xarm6_grasp,
    "xarm_grasp": xarm_grasp,
}
report = {}
for name, blueprint in blueprints.items():
    sim = next(atom for atom in blueprint.blueprints if atom.module is MujocoSimModule)
    coordinator = next(
        atom for atom in blueprint.blueprints if issubclass(atom.module, ControlCoordinator)
    )
    gripper_joints = {
        joint
        for task in coordinator.kwargs["tasks"]
        if task.type == "gripper"
        for joint in task.joint_names
    }
    arm_joints = [
        joint
        for component in coordinator.kwargs["hardware"]
        for joint in component.joints
        if joint not in gripper_joints
    ]
    report[name] = {"sim_dof": sim.kwargs["dof"], "arm_joints": len(arm_joints)}
print(json.dumps(report))
"""


def test_sim_blueprints_count_arm_joints_without_the_gripper() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env={**os.environ, "SIMULATION": "mujoco"},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    # Imported modules may greet on stdout; the report is the last line.
    report = json.loads(completed.stdout.strip().splitlines()[-1])

    assert report == {
        "coordinator_xarm6": {"sim_dof": 6, "arm_joints": 6},
        "coordinator_xarm7": {"sim_dof": 7, "arm_joints": 7},
        "coordinator_teleop_xarm6": {"sim_dof": 6, "arm_joints": 6},
        "coordinator_teleop_xarm7": {"sim_dof": 7, "arm_joints": 7},
        "xarm6_grasp": {"sim_dof": 6, "arm_joints": 6},
        "xarm_grasp": {"sim_dof": 7, "arm_joints": 7},
    }
