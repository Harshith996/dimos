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

"""Publish one simulated robot's odometry, TF, IMU and motor feedback as streams.

The coordinator's sim2 adapter consumes the robot but is not a Module, so it
cannot publish. This module attaches a read-only client to the same robot and
publishes what a hardware connection would: ground-truth odometry with its TF
edge, the control IMU and the motor states.
"""

from __future__ import annotations

from copy import copy
import threading
import time
from typing import Any

from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.sim2.client import SimRobotClient, connect_with_retry
from dimos.sim2.messages import RobotFeedback
from dimos.sim2.world.spec import SimulationSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class RobotStateConfig(ModuleConfig):
    """Which robot to observe and how to label it.

    Attributes:
        robot_id: The robot's id in the world.
        frame_id: Child frame of the published ``world`` to robot TF edge, also
            stamped on the IMU and motor states.
        rate_hz: Publication rate; the world's physics rate bounds it.
        connect_timeout: Seconds to wait at start for the world to publish the robot.
    """

    robot_id: str
    frame_id: str
    rate_hz: float = Field(default=50.0, gt=0)
    connect_timeout: float = Field(default=30.0, ge=0)


class RobotStateModule(Module):
    config: RobotStateConfig
    _simulation: SimulationSpec

    odom: Out[PoseStamped]
    tf: Out[TFMessage]
    imu: Out[Imu]
    motor_states: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: SimRobotClient | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._published = 0
        self._last_publication: float | None = None
        self._stale = False
        self._last_error: str | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._client = SimRobotClient(
            world=self._simulation, robot_id=self.config.robot_id, writable=False
        )
        connect_with_retry(self._client, self.config.connect_timeout)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim2-robot-state", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._client is not None
        period = 1.0 / self.config.rate_hz
        sequence = -1
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                sample: RobotFeedback | None = self._client.read()
            except (RuntimeError, TimeoutError) as error:
                # A sample can age past the client's timeout while another module in
                # this worker holds the interpreter, or while the world resets. Keep
                # reading; the next live sample resumes publication.
                sample = None
                with self._lock:
                    self._last_error = str(error)
                    if not self._stale:
                        self._stale = True
                        logger.warning(
                            "sim2 robot state unavailable",
                            robot=self.config.robot_id,
                            error=str(error),
                        )
            if sample is not None:
                with self._lock:
                    if self._stale:
                        logger.info("sim2 robot state restored", robot=self.config.robot_id)
                    self._stale = False
                if sample.metadata.sequence != sequence:
                    sequence = sample.metadata.sequence
                    self._publish(sample)
            deadline = max(deadline + period, time.monotonic())
            self._stop.wait(deadline - time.monotonic())

    def _publish(self, sample: RobotFeedback) -> None:
        joints = copy(sample.joints)
        joints.frame_id = self.config.frame_id
        self.motor_states.publish(joints)
        if sample.imu is not None:
            imu = copy(sample.imu)
            imu.frame_id = self.config.frame_id
            self.imu.publish(imu)
        pose = sample.odom
        self.odom.publish(pose)
        self.tf.publish(
            TFMessage(
                Transform(
                    translation=pose.position,
                    rotation=pose.orientation,
                    frame_id=pose.frame_id,
                    child_frame_id=self.config.frame_id,
                    ts=pose.ts,
                )
            )
        )
        with self._lock:
            self._published += 1
            self._last_publication = pose.ts

    @rpc
    def status(self) -> dict[str, Any]:
        """Publication count and time, whether feedback is currently stale, and the last error."""
        with self._lock:
            return {
                "robot_id": self.config.robot_id,
                "published": self._published,
                "last_publication": self._last_publication,
                "stale": self._stale,
                "last_error": self._last_error,
            }

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None
        super().stop()
