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

"""Bimanual sim2 deployment using native servos and the shared teleop coordinator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinatorConfig, TaskConfig
from dimos.control.teleop_coordinator import TeleopControlCoordinator
from dimos.core.core import rpc
from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.spec import (
    IMUState,
    MotorCommand,
    MotorState,
    WholeBodyAdapter,
    WholeBodyConfig,
)
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.robot.assets.model import RobotModel
from dimos.robot.manipulators.common.blueprints import teleop_ik_task
from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_SIDES, dual_openyam_arm_joints
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_JOINTS,
)
from dimos.robot.manipulators.dual_openyam.spatial_model import (
    BASE_POSITIONS,
    artifact_directory,
    write_ik_model,
)
from dimos.robot.manipulators.dual_openyam.teleop_ik import DualOpenYamPinkPoseTargetSolver
from dimos.robot.manipulators.openyam.simulation import (
    PASSIVE_JOINT,
    TARGET_LIMITS,
    YamArtifact,
    load_yam_artifact,
)
from dimos.sim2.client import SimRobotClient
from dimos.sim2.messages import StateFrame, StreamDescriptor
from dimos.sim2.module import SimulationModule, SimulationModuleConfig, SimulationSpec
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.spec import WorldConfig

SCENE = Path(__file__).parent / "scenes" / "spatial.xml"
WORKSPACE_CAMERA = Camera(
    "workspace",
    "workspace",
    width=480,
    height=360,
    rate_hz=10,
    depth=True,
    pointcloud=True,
    shadows=False,
)
WRIST_CAMERA = Camera(
    "wrist", "camera_d405", width=320, height=240, rate_hz=15, depth=False, shadows=False
)


def _station_robot_physics(model: mujoco.MjModel, prefix: str) -> None:
    """Ideal firmware gravity compensation; hide collision proxies from cameras.

    The physical YAM driver enables gravity compensation. This station models
    it on robot links only, retaining the imported native servo gains. It is
    not a calibrated hardware model or the unmodified RL artifact deployment.
    """
    for body in range(model.nbody):
        if model.body(body).name.startswith(prefix):
            model.body_gravcomp[body] = 1.0
    # Recompute the compiled gravity-compensation count used by mj_passive.
    mujoco.mj_setConst(model, mujoco.MjData(model))  # type: ignore[attr-defined]
    for geom in range(model.ngeom):
        if model.body(model.geom_bodyid[geom]).name.startswith(prefix) and model.geom_group[
            geom
        ] in (3, 4):
            model.geom_rgba[geom, 3] = 0


def spatial_world(artifact: YamArtifact) -> WorldConfig:
    robot = artifact.robot_instance()
    robot = replace(
        robot,
        # Native YAM coordinates are closure; zero opens both coupled fingers.
        initial_positions={**robot.initial_positions, "arm/gripper": 0.0},
        passive_joint_positions={**robot.passive_joint_positions, PASSIVE_JOINT: 0.0},
        controller={**robot.controller, "gravity_compensation": "ideal_robot_links"},
        config=replace(
            robot.config, sensors=(WRIST_CAMERA,), configure_physics=_station_robot_physics
        ),
    )
    return WorldConfig(
        scene=SCENE,
        robots={
            side: replace(robot, xyz=BASE_POSITIONS[side], wait_for_first_command=True)
            for side in DUAL_OPENYAM_SIDES
        },
        sensors=(WORKSPACE_CAMERA,),
        timestep=artifact.contract.physics_dt,
    )


class SpatialSimulationConfig(SimulationModuleConfig):
    artifact_directory: Path = Field(default_factory=artifact_directory)
    world: WorldConfig = WorldConfig(scene=SCENE, robots={})


class OpenYamSpatialSimulation(SimulationModule):
    config: SpatialSimulationConfig

    @rpc
    def build(self) -> None:
        if self._runtime is None:
            self.config.world = spatial_world(load_yam_artifact(self.config.artifact_directory))
        super().build()


class DualYamSimAdapter:
    """Two sim2 device clients, one coordinator owner; public grippers are opening.

    Native position servos retain the artifact's gains. Resets invalidate this
    connection: restarting the blueprint is required before commanding again.
    """

    def __init__(self, descriptions: dict[str, Any]) -> None:
        self.clients = {
            side: SimRobotClient(StreamDescriptor.from_dict(descriptions[side]), writable=True)
            for side in DUAL_OPENYAM_SIDES
        }
        self.frames: dict[str, StateFrame] = {}
        self.generations: dict[str, int] = {}
        self.connected = False

    def connect(self) -> bool:
        try:
            for client in self.clients.values():
                client.connect()
            self.connected = True
            self.read_motor_states()
            # Complete sim2's startup handshake at the measured pose. Idle
            # TeleopIKTasks emit no targets until grabbed; physics must already
            # run for the camera observations that permit that first grab.
            for side, client in self.clients.items():
                frame = self.frames[side]
                client.write_positions(frame.values["position"].tolist(), observed=frame)
            return True
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        self.connected = False
        for client in self.clients.values():
            client.close()
        self.frames.clear()

    def is_connected(self) -> bool:
        return self.connected

    def has_motor_states(self) -> bool:
        try:
            self.read_motor_states()
            return True
        except (RuntimeError, TimeoutError):
            return False

    def read_motor_states(self) -> list[MotorState]:
        arms: list[MotorState] = []
        grippers: list[MotorState] = []
        for side, client in self.clients.items():
            frame = client.read()
            generation = frame.metadata.episode_id
            if self.generations.setdefault(side, generation) != generation:
                self.connected = False
                self.frames.clear()
                raise RuntimeError("Simulation reset; restart the spatial blueprint")
            self.frames[side] = frame
            values = frame.values
            arms.extend(
                MotorState(float(q), float(dq), float(tau))
                for q, dq, tau in zip(
                    values["position"][:6],
                    values["velocity"][:6],
                    values["effort"][:6],
                    strict=True,
                )
            )
            grippers.append(
                MotorState(
                    1 - float(values["position"][6]),
                    -float(values["velocity"][6]),
                    -float(values["effort"][6]),
                )
            )
        return arms + grippers

    def read_imu(self) -> IMUState:
        # Fixed-base arms have no IMU in this deployment; coordinator does not consume it.
        return IMUState()

    def get_limits(self) -> JointLimits:
        lower, upper = zip(*TARGET_LIMITS[:6], strict=True)
        return JointLimits(
            position_lower=[*lower, *lower, 0.0, 0.0],
            position_upper=[*upper, *upper, 1.0, 1.0],
            velocity_max=[2.0] * 12 + [1.0, 1.0],
        )

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        if not self.connected or len(commands) != 14 or len(self.frames) != 2:
            return False
        targets = np.array([c.q for c in commands])
        limits = self.get_limits()
        if (
            not np.isfinite(targets).all()
            or np.any(targets < limits.position_lower)
            or np.any(targets > limits.position_upper)
        ):
            return False
        if any(c.dq != 0 or c.kp != 0 or c.kd != 0 or c.tau != 0 for c in commands):
            return False
        for i, (side, client) in enumerate(self.clients.items()):
            values = [*targets[i * 6 : i * 6 + 6], 1.0 - targets[12 + i]]
            client.write_positions(values, observed=self.frames[side])
        return True


class SpatialCoordinatorConfig(ControlCoordinatorConfig):
    artifact_directory: Path = Field(default_factory=artifact_directory)


class OpenYamSpatialCoordinator(TeleopControlCoordinator):
    config: SpatialCoordinatorConfig
    _simulation: SimulationSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_directory: TemporaryDirectory[str] | None = None

    @rpc
    def build(self) -> None:
        if self._model_directory is not None:
            return
        artifact = load_yam_artifact(self.config.artifact_directory)
        self._model_directory = TemporaryDirectory(prefix="dimos-yam-ik-")
        path = write_ik_model(artifact, Path(self._model_directory.name) / "robot.urdf")
        model = RobotModelConfig(
            model=RobotModel.from_file(path).with_default_joint_acceleration_limit(1.0),
            joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
            base_link="world",
        )
        hardware = HardwareComponent(
            hardware_id="dual_openyam",
            hardware_type=HardwareType.WHOLE_BODY,
            joints=list(DUAL_OPENYAM_JOINTS),
            wb_config=WholeBodyConfig(kp=(0.0,) * 14, kd=(0.0,) * 14),
        )
        self.config.hardware = [hardware]
        self.config.tasks = [
            TaskConfig(
                name="joint_trajectory",
                type="trajectory",
                joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
                params={"hold_final_position": True},
            )
        ]
        for side in DUAL_OPENYAM_SIDES:
            self.config.tasks.extend(
                [
                    teleop_ik_task(
                        hardware,
                        robot_model=model,
                        name=f"spatial_{side}",
                        joint_names=dual_openyam_arm_joints(side),
                        priority=20,
                        solver_type=DualOpenYamPinkPoseTargetSolver,
                        bindings=[{"hand": side, "target_frame": f"{side}_grasp_frame"}],
                        params={
                            "pink": PinkKinematicsConfig(dt=0.01, posture_cost=0.01),
                            "timeout": 0.3,
                            "max_joint_velocity_rad_s": 1.0,
                            "feedback_limit_tolerance": 0.03,
                            "joint_command_filter_cutoff_hz": 15.0,
                        },
                    ),
                    TaskConfig(
                        name=f"{side}_gripper",
                        type="gripper",
                        priority=20,
                        joint_names=[f"{side}_arm/gripper"],
                        stream_bind={"gripper_command": f"{side}_gripper_command"},
                    ),
                ]
            )
        try:
            super().build()
        except BaseException:
            self._model_directory.cleanup()
            raise

    def _create_whole_body_adapter(self, component: HardwareComponent) -> WholeBodyAdapter:
        return DualYamSimAdapter(self._simulation.describe()["robots"])

    @rpc
    def stop(self) -> None:
        super().stop()
        if self._model_directory is not None:
            self._model_directory.cleanup()
            self._model_directory = None
