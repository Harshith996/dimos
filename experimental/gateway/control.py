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

"""Operator lifecycle API. SONIC is the first controller adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from dimos.core.run_registry import list_runs
from dimos.utils.logging_config import setup_logger

logger = setup_logger()
RPC_TIMEOUT_S = 2.0
Action = Literal["activate", "deactivate", "stop-motion", "damping-stop"]
Phase = Literal[
    "unavailable", "pose-hold", "activating", "balancing", "dry-run", "fault", "stopped"
]


class ControlError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    controller_id: str
    action: Action


class ControllerInfo(BaseModel):
    id: str
    name: str
    robot: str
    model: str | None = None
    teleop_topic: str = "/tele_cmd_vel"
    capabilities: list[str] = ["base_velocity", "arm_references", "motion_clips", "vr_references"]


class ControlStatus(BaseModel):
    available: bool = False
    run_id: str | None = None
    controller: ControllerInfo | None = None
    phase: Phase = "unavailable"
    dry_run: bool = True
    fault: str | None = None
    reason: str | None = None
    can_activate: bool = False
    can_deactivate: bool = False
    command_source: str = "unknown"
    feedback_age_ms: float | None = None
    command_age_ms: float | None = None
    motor_rate_hz: float | None = None
    actions: list[Action] = []


class ControlBus(Protocol):
    def call(self, name: str, **kwargs: Any) -> Any: ...
    def close(self) -> None: ...


class ZenohControlBus:
    """Own a lazily opened RPC client; no robot modules or model weights are imported."""

    def __init__(self) -> None:
        self._rpc: Any = None
        self._lock = threading.Lock()

    def call(self, name: str, **kwargs: Any) -> Any:
        # Optional transport dependency; opening the UI must not initialize a session.
        from dimos.protocol.rpc.zenohrpc import ZenohRPC

        with self._lock:
            if self._rpc is None:
                self._rpc = ZenohRPC()
                self._rpc.start()
            rpc = self._rpc
        value, _ = rpc.call_sync(name, ([], kwargs), rpc_timeout=RPC_TIMEOUT_S)
        return value

    def close(self) -> None:
        with self._lock:
            if self._rpc is not None:
                self._rpc.stop()
                self._rpc = None


@dataclass(frozen=True)
class Target:
    run_id: str
    policy: str
    movement: str | None
    hardware: str | None
    ready: bool


class Controls:
    def __init__(self, bus: ControlBus | None = None) -> None:
        self.bus = bus or ZenohControlBus()
        self._target: Target | None = None
        self._discovery_lock = threading.Lock()
        self._action_lock = threading.Lock()

    def close(self) -> None:
        self.bus.close()

    def target(self, expected_run: str | None = None) -> Target | None:
        runs = list_runs(alive_only=True)
        if len(runs) != 1:
            if expected_run is not None or len(runs) > 1:
                raise ControlError(
                    409, "Select one running robot stack before using policy controls"
                )
            return None
        run = runs[0]
        if expected_run is not None and run.run_id != expected_run:
            raise ControlError(409, "The robot stack changed. Refresh its state before acting")
        # This adapter's supported blueprint explicitly requires Zenoh.
        if "unitree-g1-sonic-wbc" not in (run.cli_args or [run.blueprint]):
            return None
        with self._discovery_lock:
            if self._target is not None and self._target.run_id == run.run_id:
                return self._target
            modules = self.bus.call("Coordinator/list_modules")
            policies = [m for m in modules if m.class_name == "G1SonicPolicyModule"]
            if len(policies) != 1:
                raise ControlError(409, "Expected one SONIC policy module in this robot stack")
            policy = policies[0]
            movements = [m for m in modules if m.class_name == "MovementManager"]
            hardware = [m for m in modules if m.class_name == "G1WholeBodyConnection"]
            if len(movements) != 1 or len(hardware) > 1:
                raise ControlError(409, "The stack has ambiguous movement or motor ownership")
            movement = movements[0]
            self._target = Target(
                run.run_id,
                policy.rpc_name or policy.class_name,
                movement.rpc_name or movement.class_name,
                (hardware[0].rpc_name or hardware[0].class_name) if hardware else None,
                "activate" in policy.rpc_names
                and {"stop_motion", "movement_status"}.issubset(movement.rpc_names),
            )
            return self._target

    def status(self) -> ControlStatus:
        target = self.target()
        if target is None:
            return ControlStatus(reason="No supported motion controller is running")
        raw = self.bus.call(f"{target.policy}/status")
        motor = self.bus.call(f"{target.hardware}/command_stream_status") if target.hardware else {}
        movement = self.bus.call(f"{target.movement}/movement_status") if target.ready else {}
        fault = raw.get("fault_reason") or motor.get("fault_reason")
        phases: dict[str, Phase] = {
            "unarmed": "pose-hold",
            "initializing": "activating",
            "control": "balancing",
            "fault": "fault",
            "stopped": "stopped",
        }
        phase = phases.get(raw["control_state"], "unavailable")
        if raw.get("arm_pending"):
            phase = "activating"
        if fault:
            phase = "fault"
        elif phase == "balancing" and raw["dry_run"]:
            phase = "dry-run"
        reason = None
        if not target.ready:
            reason = "Restart the stack with the policy-control update to enable these actions"
        elif fault:
            reason = "Damping stop is latched. Restart the stack to recover"
        elif not raw.get("feedback_received"):
            reason = "Waiting for robot feedback"
        elif not raw.get("simulation"):
            ages = [raw.get("feedback_age_ms"), raw.get("imu_age_ms"), motor.get("feedback_age_ms")]
            limit = raw.get("feedback_timeout_ms", 0)
            if any(age is None or not math.isfinite(age) or age >= limit for age in ages):
                reason = "Robot feedback is stale"
        self.target(target.run_id)
        return ControlStatus(
            available=True,
            run_id=target.run_id,
            controller=ControllerInfo(
                id=target.policy, name="SONIC", robot="Unitree G1", model=raw.get("model")
            ),
            phase=phase,
            dry_run=raw["dry_run"],
            fault=fault,
            reason=reason,
            can_activate=reason is None and phase == "pose-hold" and raw.get("active", False),
            can_deactivate=phase in ("activating", "balancing", "dry-run"),
            command_source=movement.get("source", "unknown"),
            feedback_age_ms=motor.get("feedback_age_ms", raw.get("feedback_age_ms")),
            command_age_ms=motor.get("latest_command_age_ms"),
            motor_rate_hz=motor.get("configured_rate_hz"),
            actions=["activate", "deactivate", "stop-motion", "damping-stop"]
            if target.ready
            else ["damping-stop"],
        )

    def action(self, request: ControlRequest) -> ControlStatus:
        target = self.target(request.run_id)
        if target is None or request.controller_id != target.policy:
            raise ControlError(409, "The selected controller is no longer running")
        logger.info(
            "Operator policy action",
            action=request.action,
            run_id=request.run_id,
            controller=request.controller_id,
        )
        # A damping stop must not wait behind a blocked policy/status request.
        if request.action == "damping-stop":
            if target.hardware:
                self.bus.call(f"{target.hardware}/set_estop", estopped=True)
            else:
                self.bus.call(f"{target.policy}/halt_robot")
            return self.status()
        with self._action_lock:
            self.target(request.run_id)
            if not target.ready:
                raise ControlError(409, "Restart the stack with the policy-control update first")
            if request.action == "activate":
                before = self.status()
                if not before.can_activate:
                    raise ControlError(
                        409, before.reason or "Deactivate the policy before activating it"
                    )
                self.bus.call(f"{target.movement}/stop_motion")
                if not self.bus.call(f"{target.policy}/activate"):
                    raise ControlError(409, "The policy did not accept activation")
            elif request.action == "deactivate":
                self.bus.call(f"{target.policy}/disarm")
                self.bus.call(f"{target.movement}/stop_motion")
            elif request.action == "stop-motion":
                self.bus.call(f"{target.movement}/stop_motion")
            return self.status()
