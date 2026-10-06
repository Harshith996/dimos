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

"""Coordinate SONIC references without taking ownership of its motor loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import inspect
import math
import threading
import time
from typing import Any

from dimos.control.components import HardwareComponent, JointState as JointReading
from dimos.control.coordinator import ControlCoordinator, ControlCoordinatorConfig
from dimos.control.hardware_interface import ConnectedHardware
from dimos.control.tasks.trajectory_task.trajectory_task import JointTrajectoryTask
from dimos.control.tasks.velocity_task.velocity_task import JointVelocityTask
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.hardware.manipulators.spec import ControlMode
from dimos.hardware.spec import JointLimits
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.sonic_connection import G1SonicConnection


class _SonicArmReferences(ConnectedHardware):
    """Expose measured arms and send only the references selected by the coordinator.

    This wraps a connection module, not a motor adapter. In particular it must
    not fill unclaimed joints with measured angles or send PD/torque commands.
    """

    def __init__(
        self,
        component: HardwareComponent,
        feedback: In[JointState],
        command: Out[JointState],
    ) -> None:
        self._component = component
        self._joint_names = list(component.joints)
        self._command = command
        self._state: dict[str, JointReading] = {}
        self._received_at = -math.inf
        self._lock = threading.Lock()
        self._unsubscribe = feedback.subscribe(self._on_state)

    @property
    def adapter(self) -> _SonicArmReferences:  # type: ignore[override]
        # The endpoint is local. Coordinator teardown must not query another
        # module over RPC after that module's service has already stopped.
        return self

    def _on_state(self, msg: JointState) -> None:
        readings = {
            name: JointReading(q, dq, effort)
            for name, q, dq, effort in zip(
                msg.name, msg.position, msg.velocity, msg.effort, strict=True
            )
            if name in self._joint_names
        }
        if len(readings) == self.dof:
            with self._lock:
                self._state = readings
                self._received_at = time.perf_counter()

    def ready_for_control(self) -> bool:
        with self._lock:
            return time.perf_counter() - self._received_at < 0.1

    def read_state(self) -> dict[str, JointReading]:
        with self._lock:
            return self._state.copy()

    def get_limits(self) -> JointLimits | None:
        return self.component.limits

    def write_command(self, commands: dict[str, float], mode: ControlMode) -> bool:
        if mode not in (ControlMode.POSITION, ControlMode.SERVO_POSITION):
            raise ValueError("SONIC arms accept position references only")
        self._command.publish(JointState(name=list(commands), position=list(commands.values())))
        return True

    def disconnect(self) -> None:
        self._unsubscribe()
        with self._lock:
            self._state.clear()
            self._received_at = -math.inf


class SonicCoordinatorConfig(ControlCoordinatorConfig):
    auto_activated: bool = False


class SonicCoordinator(ControlCoordinator):
    """Arbitrate walking/arm tasks and forward lifecycle commands to SONIC.

    The standard connection also works without this module. In this blueprint
    navigation and joint targets enter the coordinator; only its selected
    references reach the connection. SONIC continuously owns all 29 motors.
    """

    config: SonicCoordinatorConfig
    _sonic: G1SonicConnection
    sonic_joint_state: In[JointState]
    position_command: Out[JointState]

    # Preserve the former sonic_wbc task's shell commands. These configure the
    # policy; velocity and arm targets instead enter the actual coordinator tasks.
    _policy_commands = frozenset(
        {
            "set_locomotion_mode",
            "list_locomotion_modes",
            "set_base_height",
            "play_motion_clip",
            "stop_motion_clip",
            "list_motion_clips",
            "set_vr_3point",
            "clear_vr_3point",
            "state_snapshot",
            "status",
        }
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._commands_enabled = self.config.auto_activated
        self._command_lock = threading.RLock()

    def _setup_hardware(self, component: HardwareComponent) -> None:
        if component.adapter_type != "sonic_arm_references":
            super()._setup_hardware(component)
            return
        with self._hardware_lock:
            if component.hardware_id in self._hardware:
                return
            self._hardware[component.hardware_id] = _SonicArmReferences(
                component, self.sonic_joint_state, self.position_command
            )
            self._joint_to_hardware.update(dict.fromkeys(component.joints, component.hardware_id))

    def _dispatch(self, stream: str, msg: Any) -> None:
        with self._command_lock:
            if self._commands_enabled:
                super()._dispatch(stream, msg)

    @contextmanager
    def _clear_commands(self, arms_only: bool = False) -> Iterator[None]:
        """Drain an in-flight reference before resetting/arming the connection."""
        with self._command_lock:
            running = self._tick_loop is not None and self._tick_loop.is_running
            if running:
                assert self._tick_loop is not None
                self._tick_loop.stop()
            try:
                with self._task_lock:
                    for task in self._tasks.values():
                        if isinstance(task, JointVelocityTask) and not arms_only:
                            task.clear()
                        elif isinstance(task, JointTrajectoryTask):
                            task.cancel()
                if not arms_only:
                    self._sonic.set_velocity_command(0.0, 0.0, 0.0)
                yield
            finally:
                if running:
                    assert self._tick_loop is not None
                    self._tick_loop.start()

    @rpc
    def set_activated(self, engaged: bool) -> None:
        """Arm/disarm SONIC and discard pending walking and arm commands."""
        if engaged:
            self.arm()
        else:
            self.disarm()

    @rpc
    def arm(self, ramp_seconds: float | None = None) -> bool:
        with self._clear_commands():
            self._commands_enabled = False
            result = self._sonic.arm(ramp_seconds)
            self._commands_enabled = bool(self._sonic.status()["active"])
            return result

    @rpc
    def disarm(self) -> bool:
        with self._clear_commands():
            self._commands_enabled = False
            return self._sonic.disarm()

    @rpc
    def set_dry_run(self, enabled: bool) -> None:
        self._sonic.set_dry_run(enabled)

    @rpc
    def set_estop(self, estopped: bool) -> bool:
        # Reach the motor-owner latch before waiting for task/command locks.
        self._sonic.set_estop(estopped)
        if estopped:
            with self._clear_commands():
                self._commands_enabled = False
        return True

    @rpc
    def reset_runtime_state(self, reactivate: bool | None = None) -> dict[str, bool]:
        with self._clear_commands():
            result = self._sonic.reset_runtime_state(reactivate)
            self._commands_enabled = result and (
                self._commands_enabled if reactivate is None else reactivate
            )
            return {"sonic_wbc": result}

    @rpc
    def set_upper_body(self, positions: list[float]) -> bool:
        """Submit a complete arm reference to the coordinator's trajectory task."""
        if len(positions) != 14 or not all(math.isfinite(q) for q in positions):
            raise ValueError("Upper-body reference requires 14 finite positions")
        with self._command_lock, self._task_lock:
            if not self._commands_enabled:
                raise RuntimeError("Activate SONIC before sending motion commands")
            if self._trajectory_task is None:
                return False
            return self._trajectory_task.on_joint_command(
                JointState(name=self._hardware["g1_arms"].joint_names, position=positions),
                time.perf_counter(),
            )

    @rpc
    def clear_upper_body(self) -> None:
        with self._clear_commands(arms_only=True):
            self._sonic.clear_upper_body()

    @rpc
    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float, t_now: float | None = None
    ) -> None:
        """Submit walking intent to the velocity task, subject to arbitration."""
        if not all(math.isfinite(v) for v in (vx, vy, yaw_rate)):
            raise ValueError("Velocity commands must be finite")
        with self._command_lock, self._task_lock:
            if not self._commands_enabled:
                raise RuntimeError("Activate SONIC before sending motion commands")
            task = self._tasks["sonic_wbc"]
            if not isinstance(task, JointVelocityTask):
                raise TypeError("sonic_wbc must be a velocity task")
            task.set_velocities([vx, vy, yaw_rate], time.perf_counter() if t_now is None else t_now)

    def _legacy_commands(self) -> dict[str, Callable[..., Any]]:
        return {
            **{name: getattr(self._sonic, name) for name in self._policy_commands},
            "set_velocity_command": self.set_velocity_command,
            "set_upper_body": self.set_upper_body,
            "clear_upper_body": self.clear_upper_body,
            "arm": self.arm,
            "disarm": self.disarm,
            "set_dry_run": self.set_dry_run,
            "set_estop": self.set_estop,
            "reset_runtime_state": self.reset_runtime_state,
        }

    @rpc
    def task_invoke(self, task_name: str, method: str, kwargs: dict[str, Any] | None = None) -> Any:
        if task_name == "sonic_wbc":
            handler = self._legacy_commands().get(method)
            if handler is not None:
                result = handler(**(kwargs or {}))
                return result["sonic_wbc"] if method == "reset_runtime_state" else result
        with self._command_lock:
            return super().task_invoke(task_name, method, kwargs)

    @rpc
    def describe_task(self, task_name: str) -> dict[str, Any] | None:
        description = super().describe_task(task_name)
        if task_name == "sonic_wbc" and description is not None:
            for name in self._legacy_commands():
                # Read signatures locally; the injected object may be an RPC proxy.
                method = getattr(G1SonicConnection, name)
                sig = inspect.signature(method)
                sig = sig.replace(parameters=[p for n, p in sig.parameters.items() if n != "self"])
                description["commands"][name] = {
                    "signature": str(sig),
                    "params": list(sig.parameters),
                }
        return description

    @rpc
    def stop(self) -> None:
        # Stop references locally. SONIC's input timeout keeps it balancing;
        # its own stop() latches damping when the whole stack shuts down.
        self._commands_enabled = False
        super().stop()
