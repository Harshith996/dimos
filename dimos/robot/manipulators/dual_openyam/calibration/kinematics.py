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

"""Where each arm's fingertip is, from its joint angles.

The arm chain comes from the authoritative dual OpenYAM URDF; both sides carry
the identical i2rt arm. Each arm is solved in its own base frame and then
placed in the rig frame with the calibration's base spacing, so the URDF's own
spacing never enters a calibration number.

Rig frame: origin midway between the two arm bases at the height of the
mounting plates, x forward, y toward the left arm, z up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin

from dimos.robot.manipulators.dual_openyam.calibration.encoders import JOINT_COUNT, SIDES
from dimos.robot.manipulators.dual_openyam.model import DUAL_OPENYAM_MODEL_PATH

# The flat front face of the gripper housing, where the fingers come out, in
# the gripper link frame (gripper.stl's lowest Z). The fingers extend along -Z.
# Measuring finger length from this face needs no access to the link origin.
GRIPPER_HOUSING_FACE_Z = -0.0635


@dataclass(frozen=True)
class RigGeometry:
    """The hand-measured numbers the arm chain is placed with."""

    base_spacing_m: float = 0.430
    # Housing face to the end of the closed fingertips, calipers 2026-10-14
    # (82 to 83 mm on both arms; the URDF's stock finger is 83.3 mm).
    finger_length_m: Mapping[str, float] = field(
        default_factory=lambda: {"left": 0.0825, "right": 0.0825}
    )
    # Grasp point: middle of the 10 mm pinch surface at the fingertip.
    grasp_point_from_tip_m: float = 0.005
    # Base block bottom to table top: 30 mm on both arms, rule 2026-10-14 (the
    # workcell model has 42 mm by tape).
    table_z_m: float = -0.030

    def base_position(self, side: str) -> np.ndarray:
        sign = 1.0 if side == "left" else -1.0
        return np.array([0.0, sign * self.base_spacing_m / 2.0, 0.0])

    def fingertip_in_gripper(self, side: str) -> np.ndarray:
        return np.array([0.0, 0.0, GRIPPER_HOUSING_FACE_Z - self.finger_length_m[side]])

    def grasp_point_in_gripper(self, side: str) -> np.ndarray:
        return self.fingertip_in_gripper(side) + np.array([0.0, 0.0, self.grasp_point_from_tip_m])


class ArmKinematics:
    """Forward kinematics of one arm at a time, in its own base frame."""

    def __init__(self, model_path: Path = DUAL_OPENYAM_MODEL_PATH) -> None:
        self._model = pin.buildModelFromUrdf(str(model_path))
        self._data = self._model.createData()

    def gripper_in_base(self, side: str, joints_rad: Sequence[float]) -> pin.SE3:
        """The ``{side}_gripper`` link pose in the ``{side}_base`` frame."""
        if side not in SIDES:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if len(joints_rad) != JOINT_COUNT:
            raise ValueError(f"expected {JOINT_COUNT} joint angles, got {len(joints_rad)}")
        q = pin.neutral(self._model)
        for index, value in enumerate(joints_rad, start=1):
            joint_id = self._model.getJointId(f"{side}_joint{index}")
            q[self._model.joints[joint_id].idx_q] = value
        pin.framesForwardKinematics(self._model, self._data, q)
        base = self._data.oMf[self._model.getFrameId(f"{side}_base")]
        gripper = self._data.oMf[self._model.getFrameId(f"{side}_gripper")]
        return base.actInv(gripper)

    def point_in_rig(
        self,
        side: str,
        joints_rad: Sequence[float],
        point_in_gripper: np.ndarray,
        geometry: RigGeometry,
    ) -> np.ndarray:
        gripper = self.gripper_in_base(side, joints_rad)
        return geometry.base_position(side) + gripper.act(point_in_gripper)

    def fingertip_in_rig(
        self, side: str, joints_rad: Sequence[float], geometry: RigGeometry
    ) -> np.ndarray:
        return self.point_in_rig(side, joints_rad, geometry.fingertip_in_gripper(side), geometry)
