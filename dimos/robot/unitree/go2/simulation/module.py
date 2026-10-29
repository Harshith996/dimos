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

"""A simulated Go2 with GO2Connection's streams and RPCs: the walking policy on a sim2 robot."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.unitree.go2.constants import CMD_VEL_TIMEOUT
from dimos.robot.unitree.go2.simulation.model import CONTROL_IMU, go2_definition
from dimos.sim2.client import SimRobotClient
from dimos.sim2.module import SimulationSpec
from dimos.sim2.pacing import wait_until
from dimos.sim2.spec import RobotInstance, SimulatedRobot
from dimos.simulation.go2_legged.policy import Go2Policy, Proprioception, load_policy
from dimos.simulation.go2_legged.robot import COMMAND_SLEW, CONTROL_DT, projected_gravity
from dimos.utils.data import LfsPath
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class Go2SimConfig(ModuleConfig):
    robot_id: str = "go2"
    policy: Path | None = None
    spawn: str = "default"
    odom_frame_id: str = "world"
    publish_tf: bool = True


class Go2Sim(Module):
    dedicated_worker = True
    config: Go2SimConfig
    _simulation: SimulationSpec

    cmd_vel: In[Twist]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._policy: Go2Policy = load_policy(self.config.policy)
        self._client: SimRobotClient | None = None
        self._stop_event = Event()
        self._command_lock = Lock()
        self._command = np.zeros(3)
        self._command_expires = 0.0
        self._thread: Thread | None = None
        self._error: str | None = None

    @staticmethod
    def describe_simulated(config: Go2SimConfig) -> SimulatedRobot:
        policy = load_policy(config.policy)
        robot = RobotInstance(
            go2_definition(),
            initial_positions=dict(
                zip(policy.joint_names, map(float, policy.default_pose), strict=True)
            ),
            hold_gains={
                name: (float(p), float(d))
                for name, p, d in zip(policy.joint_names, policy.kp, policy.kd, strict=True)
            },
            control_imu=CONTROL_IMU,
            wait_for_first_command=True,
            controller={
                "implementation": type(policy).__name__,
                "weights": str(config.policy or LfsPath("go2_sim/go2_policy/policy.onnx")),
            },
        )
        return SimulatedRobot(config.robot_id, robot, spawn=config.spawn)

    @rpc
    def start(self) -> None:
        super().start()
        self._client = SimRobotClient(world=self._simulation, robot_id=self.config.robot_id)
        self._client.connect()
        names = self._client.joint_names
        if set(names) != set(self._policy.joint_names):
            raise ValueError("Go2 policy joints do not match the simulated robot")
        self._policy_order = np.array([names.index(name) for name in self._policy.joint_names])
        self._motor_order = np.argsort(self._policy_order)
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="go2-sim-policy", daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._client is not None:
            self._client.close()
            self._client = None
        super().stop()

    def _run(self) -> None:
        # ONNX inference blocks, so the 50 Hz policy loop keeps its own thread.
        assert self._client is not None
        client, policy = self._client, self._policy
        command = np.zeros(3)
        episode = -1
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                frame = client.read()
                if frame.imu is None:
                    raise RuntimeError("Go2 locomotion requires IMU feedback")
                if frame.metadata.episode_id != episode:
                    policy.reset()
                    command[:] = 0
                    if episode != -1:
                        self.stop_movement()
                    episode = frame.metadata.episode_id
                with self._command_lock:
                    desired = (
                        self._command if time.monotonic() < self._command_expires else np.zeros(3)
                    )
                command += np.clip(desired - command, -COMMAND_SLEW, COMMAND_SLEW)
                x, y, z, w = frame.imu.orientation.to_tuple()
                target = policy.act(
                    Proprioception(
                        angular_velocity=np.array(frame.imu.angular_velocity.to_tuple()),
                        gravity=projected_gravity(np.array([w, x, y, z])),
                        joint_position=np.asarray(frame.joints.position)[self._policy_order],
                        joint_velocity=np.asarray(frame.joints.velocity)[self._policy_order],
                    ),
                    command,
                )
                try:
                    client.write_motors(
                        MotorCommandArray(
                            q=target[self._motor_order].tolist(),
                            kp=policy.kp[self._motor_order].tolist(),
                            kd=policy.kd[self._motor_order].tolist(),
                            timestamp=frame.joints.ts,
                        ),
                        observed=frame,
                    )
                except TimeoutError:
                    # Physics stalled past the feedback timeout; act again on the next sample.
                    logger.warning("Go2 sim skipped a stale policy step")
                    wait_until(self._stop_event, time.monotonic() + CONTROL_DT)
                    continue
                except RuntimeError:
                    # A reset during inference invalidates this result, not the policy loop.
                    if client.read().metadata.episode_id != episode:
                        continue
                    raise
                self._publish_odom(frame.odom)
                deadline += CONTROL_DT
                wait_until(self._stop_event, deadline)
                if time.monotonic() - deadline > CONTROL_DT:
                    deadline = time.monotonic()
        except Exception as error:
            self._error = str(error)
            logger.exception("Go2 sim policy stopped; the physics command watchdog will expire")

    async def handle_cmd_vel(self, twist: Twist) -> None:
        self.move(twist)

    def _publish_odom(self, odom: PoseStamped) -> None:
        odom.frame_id = self.config.odom_frame_id
        if self.config.publish_tf:
            base_link = (
                f"{self.config.frame_id_prefix}/base_link"
                if self.config.frame_id_prefix
                else "base_link"
            )
            self.tf.publish(TFMessage(Transform.from_pose(base_link, odom)))
        self.odom.publish(odom)

    def _check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Go2 sim policy stopped: {self._error}")
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Go2 sim policy is not running")

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        command = np.array([twist.linear.x, twist.linear.y, twist.angular.z])
        if duration < 0 or not np.isfinite([*command, duration]).all():
            raise ValueError("Go2 movement must be finite with a nonnegative duration")
        self._check_health()
        with self._command_lock:
            self._command = command
            self._command_expires = time.monotonic() + (duration or CMD_VEL_TIMEOUT)
        return True

    @rpc
    def stop_movement(self) -> None:
        with self._command_lock:
            self._command = np.zeros(3)
            self._command_expires = 0.0

    @rpc
    def standup(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @rpc
    def balance_stand(self) -> bool:
        self.stop_movement()
        return self.standup()

    @rpc
    def liedown(self) -> bool:
        raise NotImplementedError("the Go2 walking policy does not implement lie-down")

    @rpc
    def sport_command(self, api_id: int) -> bool:
        raise NotImplementedError("vendor sport actions are not simulated")

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        raise NotImplementedError("vendor requests are not simulated")

    @rpc
    def set_rage_mode(self, enable: bool) -> bool:
        return False

    @rpc
    def set_light(self, level: int) -> bool:
        return False

    @rpc
    def set_obstacle_avoidance(self, enabled: bool = True) -> bool:
        return False

    @rpc
    def switch_joystick(self, enable: bool = True) -> bool:
        return False

    @rpc
    def battery_soc(self) -> int | None:
        return None
