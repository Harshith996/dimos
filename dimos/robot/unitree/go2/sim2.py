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

"""A simulated Go2 with GO2Connection's cmd_vel/odom/tf: the walking policy driving sim2 servos."""

import asyncio
from collections.abc import AsyncIterator
import math
from pathlib import Path
import time

import numpy as np

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.constants import CMD_VEL_TIMEOUT
from dimos.sim2.module import SimulationSpec
from dimos.sim2.spec import Servo, SimulatedRobot
from dimos.sim2.state import SimClient
from dimos.simulation.go2_legged.policy import Proprioception, load_policy
from dimos.simulation.go2_legged.robot import COMMAND_SLEW, CONTROL_DT, projected_gravity, yaw_of
from dimos.utils.data import get_data

_POLL_SECONDS = 0.001


class Go2SimConfig(ModuleConfig):
    robot_id: str = "go2"
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.35)
    policy: Path | None = None
    odom_frame_id: str = "world"
    # Yaw rate commanded per radian of heading error while no turn is commanded.
    heading_gain: float = 2.0


class Go2Sim(Module):
    config: Go2SimConfig
    cmd_vel: In[Twist]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]
    _simulation: SimulationSpec

    @staticmethod
    def describe_simulated(config: Go2SimConfig) -> SimulatedRobot:
        policy = load_policy(config.policy)
        servos = zip(policy.joint_names, policy.kp, policy.kd, policy.default_pose, strict=True)
        return SimulatedRobot(
            config.robot_id,
            get_data("go2_sim") / "unitree_go2" / "go2.xml",
            "base",
            xyz=config.xyz,
            servos=tuple(
                Servo(joint, float(kp), float(kd), float(q)) for joint, kp, kd, q in servos
            ),
        )

    async def main(self) -> AsyncIterator[None]:
        self._command = np.zeros(3)
        self._command_expires = 0.0
        self._walking = asyncio.create_task(self._walk(SimClient(self._simulation.describe())))
        yield

    async def handle_cmd_vel(self, twist: Twist) -> None:
        self._command = np.array([twist.linear.x, twist.linear.y, twist.angular.z])
        self._command_expires = time.monotonic() + CMD_VEL_TIMEOUT

    async def _walk(self, client: SimClient) -> None:
        policy = load_policy(self.config.policy)
        model, prefix = client.model, f"{self.config.robot_id}/"
        joints = [model.joint(prefix + name) for name in policy.joint_names]
        qpos_adr = np.array([joint.qposadr[0] for joint in joints])
        dof_adr = np.array([joint.dofadr[0] for joint in joints])
        actuators = [
            int(np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)[0]) for joint in joints
        ]
        root = model.joint(model.body(prefix + "base").jntadr[0])
        base_qpos, base_dof = root.qposadr[0], root.dofadr[0]
        command = np.zeros(3)
        heading: float | None = None
        next_tick = 0.0
        while True:
            data = client.read()
            # The policy acts every CONTROL_DT of sim time, however fast the sim runs.
            if data.time < next_tick:
                await asyncio.sleep(_POLL_SECONDS)
                continue
            next_tick = data.time + CONTROL_DT
            quat = data.qpos[base_qpos + 3 : base_qpos + 7]
            desired = self._command if time.monotonic() < self._command_expires else np.zeros(3)
            command += np.clip(desired - command, -COMMAND_SLEW, COMMAND_SLEW)
            # The bundled policy drifts in yaw even standing still; with no turn commanded, hold heading.
            if desired[2] == 0:
                heading = yaw_of(quat) if heading is None else heading
                turn = self.config.heading_gain * math.remainder(heading - yaw_of(quat), math.tau)
            else:
                heading, turn = None, 0.0
            observation = Proprioception(
                angular_velocity=data.qvel[base_dof + 3 : base_dof + 6].copy(),
                gravity=projected_gravity(quat),
                joint_position=data.qpos[qpos_adr].copy(),
                joint_velocity=data.qvel[dof_adr].copy(),
            )
            client.write(actuators, policy.act(observation, command + np.array([0.0, 0.0, turn])))
            self._publish_pose(data.qpos[base_qpos : base_qpos + 3], quat)

    def _publish_pose(self, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
        w, x, y, z = quat_wxyz
        odom = PoseStamped(
            ts=time.time(),
            frame_id=self.config.odom_frame_id,
            position=position.tolist(),
            orientation=[x, y, z, w],
        )
        self.tf.publish(TFMessage(*GO2Connection._odom_to_tf(odom)))
        self.odom.publish(odom)
