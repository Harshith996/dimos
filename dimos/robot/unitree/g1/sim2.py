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

"""A simulated G1 with G1SimConnection's cmd_vel/odom/tf: main's MuJoCo Playground policy on sim2."""

import asyncio
from collections.abc import AsyncIterator
import math
import time

import mujoco
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.unitree.go2.constants import CMD_VEL_TIMEOUT
from dimos.sim2.module import SimulationSpec
from dimos.sim2.spec import Servo, SimulatedRobot
from dimos.sim2.state import SimClient
from dimos.simulation.mujoco.menagerie import menagerie_path
from dimos.utils.data import get_data

CONTROL_DT = 0.02
ACTION_SCALE = 0.5
GAIT_HZ = 1.5
# The command scaling and drift compensation main's G1OnnxController applies.
COMMAND_SCALE = np.array([2.0, 2.0, 1.0])
DRIFT_COMPENSATION = np.array([-0.18, 0.0, -0.09])


_POLL_SECONDS = 0.001


class G1SimConfig(ModuleConfig):
    robot_id: str = "g1"
    xy: tuple[float, float] = (0.0, 0.0)


class G1Sim(Module):
    config: G1SimConfig
    cmd_vel: In[Twist]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]
    _simulation: SimulationSpec

    @staticmethod
    def describe_simulated(config: G1SimConfig) -> SimulatedRobot:
        meshes = menagerie_path() / "unitree_g1" / "assets"
        model = get_data("mujoco_sim") / "unitree_g1.xml"
        assets = {path.name: path.read_bytes() for path in meshes.iterdir()}
        spec = mujoco.MjSpec.from_file(str(model), assets=assets)
        home = np.array(spec.key("home").qpos)
        # qpos is the floating base's 7 values, then one per hinge in model order.
        hinges = [joint.name for joint in spec.joints][1:]
        # Its actuators are already position servos; only the standing pose comes from here.
        servos = tuple(
            Servo(actuator.target, initial=float(home[7 + hinges.index(actuator.target)]))
            for actuator in spec.actuators
        )
        x, y = config.xy
        return SimulatedRobot(
            config.robot_id,
            model,
            "pelvis",
            xyz=(x, y, float(home[2])),
            servos=servos,
            assets=meshes,
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
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        policy = ort.InferenceSession(
            str(get_data("mujoco_sim") / "unitree_g1_policy.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        model, prefix = client.model, f"{self.config.robot_id}/"
        root = model.joint(model.body(prefix + "pelvis").jntadr[0])
        joints = np.flatnonzero(
            [model.joint(j).name.startswith(prefix) and j != root.id for j in range(model.njnt)]
        )
        qpos_adr, dof_adr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        actuators = [int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in joints]
        default = client.read().qpos[qpos_adr].copy()
        last_action = np.zeros(len(joints), np.float32)
        phase = np.array([0.0, math.pi])
        next_tick = 0.0
        while True:
            data = client.read()
            # The policy acts every CONTROL_DT of sim time, however fast the sim runs.
            if data.time < next_tick:
                await asyncio.sleep(_POLL_SECONDS)
                continue
            next_tick = data.time + CONTROL_DT
            desired = self._command if time.monotonic() < self._command_expires else np.zeros(3)
            command = desired * COMMAND_SCALE + DRIFT_COMPENSATION
            pelvis = data.site(prefix + "imu_in_pelvis").xmat.reshape(3, 3)
            observation = np.hstack(
                [
                    data.sensor(prefix + "local_linvel_pelvis").data,
                    data.sensor(prefix + "gyro_pelvis").data,
                    pelvis.T @ [0.0, 0.0, -1.0],
                    command,
                    data.qpos[qpos_adr] - default,
                    data.qvel[dof_adr],
                    last_action,
                    np.cos(phase),
                    np.sin(phase),
                ]
            ).astype(np.float32)
            last_action = policy.run(["continuous_actions"], {"obs": observation[None]})[0][0]
            client.write(actuators, last_action * ACTION_SCALE + default)
            phase = (
                np.fmod(phase + 2 * math.pi * GAIT_HZ * CONTROL_DT + math.pi, 2 * math.pi) - math.pi
            )
            self._publish_pose(data.qpos[root.qposadr[0] : root.qposadr[0] + 7])

    def _publish_pose(self, qpos: np.ndarray) -> None:
        w, x, y, z = qpos[3:7]
        odom = PoseStamped(
            ts=time.time(), frame_id="world", position=qpos[:3].tolist(), orientation=[x, y, z, w]
        )
        self.tf.publish(TFMessage(Transform.from_pose("base_link", odom)))
        self.odom.publish(odom)
