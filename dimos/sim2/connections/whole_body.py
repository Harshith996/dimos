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

"""Whole-body feedback and optional policy command streams for sim2."""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.whole_body.spec import MotorCommand
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.std_msgs.String import String
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.sim2.control.adapters import WholeBodyAdapter
from dimos.sim2.spec import RobotConfig
from dimos.utils.logging_config import setup_logger
from dimos.utils.realtime import make_current_thread_periodic

logger = setup_logger()


class WholeBodyConnectionConfig(ModuleConfig):
    definition: RobotConfig
    address: str
    robot_id: str
    rate_hz: float = 50.0
    command_source: Literal["coordinator", "stream"] = "coordinator"
    command_timeout_seconds: float = Field(default=0.15, gt=0.0, allow_inf_nan=False)
    feedback_timeout_seconds: float = Field(default=0.15, gt=0.0, allow_inf_nan=False)


class WholeBodyConnection(Module):
    config: WholeBodyConnectionConfig
    # the policy's feedback (200 Hz): sharing a process with planners/mappers lets their GIL time stall it past the
    # policy's 150 ms watchdog
    dedicated_worker = True
    motor_command: In[MotorCommandArray]
    controller_fault: In[String]
    fault: Out[String]
    motor_states: Out[JointState]
    imu: Out[Imu]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._device = WholeBodyAdapter(
            address=self.config.address,
            dof=len(self.config.definition.joints),
            definition=self.config.definition,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._command_lock = threading.RLock()
        self._last_command_at: float | None = None
        self._fault_reason: str | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._device.connect()
        if self.config.command_source == "stream":
            self.register_disposable(Disposable(self.motor_command.subscribe(self._command)))
            self.register_disposable(
                Disposable(self.controller_fault.subscribe(lambda msg: self._latch_fault(msg.data)))
            )
        self._stop.clear()
        self._thread = threading.Thread(target=self._publish, daemon=True)
        self._thread.start()

    def _command(self, msg: MotorCommandArray) -> None:
        with self._command_lock:
            if self._fault_reason is not None:
                return
            try:
                accepted = self._device.write_motor_commands(
                    [
                        MotorCommand(*row)
                        for row in zip(msg.q, msg.dq, msg.kp, msg.kd, msg.tau, strict=True)
                    ]
                )
                if not accepted:
                    raise RuntimeError("sim2 rejected motor commands")
                self._last_command_at = time.monotonic()
            except Exception as exc:
                self._latch_fault(str(exc))

    def _latch_fault(self, reason: str) -> None:
        with self._command_lock:
            if self._fault_reason is not None:
                return
            self._fault_reason = (reason or "sim2 controller fault")[:256]
            self._device.write_motor_commands(
                [
                    MotorCommand(q=0.0, dq=0.0, kd=joint.kd)
                    for joint in self.config.definition.joints
                ]
            )
            self.fault.publish(String(self._fault_reason))
            logger.error("sim2 damping stop latched; restart required", reason=self._fault_reason)

    def _check_command_timeout(self, now: float) -> None:
        with self._command_lock:
            if (
                self.config.command_source == "stream"
                and self._last_command_at is not None
                and now - self._last_command_at >= self.config.command_timeout_seconds
            ):
                self._latch_fault("sim2 motor command timeout")

    def _publish(self) -> None:
        make_current_thread_periodic(1.0 / self.config.rate_hz)
        try:
            while not self._stop.is_set():
                before = time.monotonic()
                self._check_command_timeout(before)
                try:
                    frame = self._device.sample()
                except BlockingIOError:
                    # A missed read is not new feedback. Retry next period;
                    # the policy watchdog still detects sustained starvation.
                    self._stop.wait(1.0 / self.config.rate_hz)
                    continue
                v = frame.values
                ts = float(v["wall_time"][0])
                if (
                    self.config.command_source == "stream"
                    and time.time() - ts >= self.config.feedback_timeout_seconds
                ):
                    self._latch_fault("sim2 physics feedback timeout")
                    self._stop.wait(1.0 / self.config.rate_hz)
                    continue
                self.motor_states.publish(
                    JointState(
                        name=[j.name for j in self.config.definition.joints],
                        position=v["position"].tolist(),
                        velocity=v["velocity"].tolist(),
                        effort=v["effort"].tolist(),
                        ts=ts,
                    )
                )
                iq = v["imu_quaternion"]
                self.imu.publish(
                    Imu(
                        orientation=Quaternion(
                            float(iq[1]), float(iq[2]), float(iq[3]), float(iq[0])
                        ),
                        angular_velocity=Vector3(*v["imu_gyroscope"]),
                        linear_acceleration=Vector3(*v["imu_accelerometer"]),
                        frame_id=f"{self.config.robot_id}/imu",
                        ts=ts,
                    )
                )
                pos = Vector3(*v["root_position"])
                q = v["root_quaternion"]
                rot = Quaternion(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
                self.odom.publish(
                    PoseStamped(position=pos, orientation=rot, frame_id="world", ts=ts)
                )
                self.tf.publish(
                    TFMessage(
                        Transform(
                            translation=pos,
                            rotation=rot,
                            frame_id="world",
                            child_frame_id=f"{self.config.robot_id}/{self.config.definition.root_body}",
                            ts=ts,
                        )
                    )
                )
                self._stop.wait(max(0, 1.0 / self.config.rate_hz - (time.monotonic() - before)))
        except Exception:
            if not self._stop.is_set():
                logger.exception("sim2 whole-body connection stopped")

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._device.disconnect()
        super().stop()
