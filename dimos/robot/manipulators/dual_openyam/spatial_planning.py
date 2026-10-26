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

"""Lazy station model shared by the ordinary planner and teleop IK.

This retains the artifact's frame-only URDF contract. It provides joint/pose
planning, not robot or workspace collision coverage. The spatial viewer renders
the measured arm geometry separately.
"""

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dimos.manipulation.planning.groups.models import PlanningGroupDefinition
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.robot.assets.git_cache import DEFAULT_ROBOT_ASSET_CACHE_ROOT
from dimos.robot.assets.model import RobotModel
from dimos.robot.manipulators.dual_openyam.config import (
    DUAL_OPENYAM_SIDES,
    dual_openyam_arm_joints,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_ARM_JOINTS
from dimos.robot.manipulators.dual_openyam.spatial_model import artifact_directory, write_ik_model
from dimos.robot.manipulators.dual_openyam.stations.anvil import ANVIL
from dimos.robot.manipulators.openyam.simulation import load_yam_artifact
from dimos.sim2.sensors.spec import Mount

# Named rest pose for the v1 spatial station, in its model joint convention.
# In this artifact joint2 and joint3 have a 0.01 rad lower bound: zero is invalid.
# This is a configured pose, independent of the first observed joint state.
_SPATIAL_ARM_HOME = (0.0, 1.047, 1.05, -0.9, 0.0, 0.0)


class SpatialModelPath(os.PathLike[str]):
    """Materialize only when RobotModel loads, never during blueprint import."""

    def __init__(
        self,
        artifact: Path,
        bases: dict[str, Mount],
        cache: Path = DEFAULT_ROBOT_ASSET_CACHE_ROOT / "derived" / "openyam",
    ) -> None:
        self._artifact = artifact
        self._bases = bases
        self._cache = cache

    def __fspath__(self) -> str:
        artifact = load_yam_artifact(self._artifact)
        self._cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self._cache) as directory:
            temporary = write_ik_model(artifact, Path(directory) / "robot.urdf", self._bases)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            destination = self._cache / f"{digest}.urdf"
            # Each process writes its own temporary file; replacement is atomic.
            temporary.replace(destination)
        return str(destination)


def spatial_model_config(
    *, artifact: Path | None = None, bases: dict[str, Mount] = ANVIL.bases
) -> RobotModelConfig:
    return RobotModelConfig(
        model=RobotModel.from_file(
            SpatialModelPath(artifact or artifact_directory(), bases)
        ).with_default_joint_acceleration_limit(1.0),
        joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
        base_link="world",
        planning_groups=[
            PlanningGroupDefinition(
                name=f"{side}_manipulator",
                joint_names=tuple(dual_openyam_arm_joints(side)),
                base_link="world",
                tip_link=f"{side}_grasp_frame",
            )
            for side in DUAL_OPENYAM_SIDES
        ],
        home_joints=list(_SPATIAL_ARM_HOME) * len(DUAL_OPENYAM_SIDES),
    )
