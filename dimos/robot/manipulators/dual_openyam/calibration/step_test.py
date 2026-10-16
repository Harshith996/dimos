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

"""Joint step-response test for the Dual OpenYAM (END-310).

The module owns the whole-body adapter directly: every tick it sends one MIT
command per joint (target angle, kp, kd, zero feed-forward; the adapter adds
gravity compensation itself) and publishes the measured and commanded joint
states. RPCs move one joint's target by a small step so the response can be
recorded with ``dimos run dual_openyam_step_test`` and analysed offline by
``step_report``; ``go_to`` ramps to any pose and ``mark`` drops a labelled
marker (``gravity_events``) so ``gravity_fit`` can group steps by pose.
Without CAN ports the mock whole-body adapter is used.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

from pydantic import Field

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.hardware.whole_body.registry import whole_body_adapter_registry
from dimos.hardware.whole_body.spec import MotorCommand, MotorState, WholeBodyAdapter
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.dual_openyam.calibration.step_analysis import StepEvent
from dimos.robot.manipulators.dual_openyam.config import (
    _ARM_KD,
    _ARM_KP,
    DUAL_OPENYAM_HOME_JOINTS,
    dual_openyam_hardware,
)
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_GRIPPER_JOINTS,
    DUAL_OPENYAM_JOINTS,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# The control coordinator ticks the same hardware at 100 Hz (ControlCoordinatorConfig).
DEFAULT_LOOP_HZ = 100.0
MAX_STEP_DEG = 5.0
# The Damiao MIT frame packs kp into 0..500 and kd into 0..5; the codec refuses anything else.
MAX_KP = 500.0
MAX_KD = 5.0
MAX_POSE_ERROR_DEG = 10.0
MAX_GO_TO_SPEED_DEG_S = 30.0
# The URDF zero is the folded rest pose on the supports.
REST_SPEED_DEG_S = 10.0
# `dimos stop` escalates to SIGKILL after 5 s, so stop() parks on a budget and
# then disables wherever the ramp got to; call park() first for a full landing.
STOP_PARK_BUDGET_S = 4.0
DEFAULT_ARM_KP = (*_ARM_KP, *_ARM_KP)
DEFAULT_ARM_KD = (*_ARM_KD, *_ARM_KD)


class StepRefusedError(ValueError):
    """A step or move was refused by the safety checks."""


class JointStepTestConfig(ModuleConfig):
    left_can_port: str | None = None
    right_can_port: str | None = None
    loop_hz: float = Field(default=DEFAULT_LOOP_HZ, gt=0.0)
    # Per arm joint, in DUAL_OPENYAM_ARM_JOINTS order (left 1-6, right 1-6).
    kp: list[float] = Field(default_factory=lambda: list(DEFAULT_ARM_KP))
    kd: list[float] = Field(default_factory=lambda: list(DEFAULT_ARM_KD))
    # Radians, per arm joint; where go_to_test_pose ramps to.
    test_pose: list[float] = Field(default_factory=lambda: list(DUAL_OPENYAM_HOME_JOINTS))
    max_step_deg: float = Field(default=MAX_STEP_DEG, gt=0.0)
    # Steps are refused while any joint is further than this from its own target.
    max_pose_error_deg: float = Field(default=MAX_POSE_ERROR_DEG, gt=0.0)
    # Path to a gravity_fit JSON (per-link mass and COM) applied to the adapter's
    # gravity model; ignored by the mock adapter.
    inertial_overrides: str | None = None


def check_step(
    joint: str,
    delta_deg: float,
    measured_rad: dict[str, float],
    target_rad: dict[str, float],
    *,
    max_step_deg: float = MAX_STEP_DEG,
    max_pose_error_deg: float = MAX_POSE_ERROR_DEG,
) -> None:
    """Raise ``StepRefusedError`` unless the step is small and every joint tracks its target.

    ``target_rad`` is the current hold target (where the last ramp ended), so
    steps are allowed at any pose ``go_to`` reached; a joint far from its own
    target means the arm is stalled, sagging or blocked.
    """
    if joint not in target_rad:
        raise StepRefusedError(f"unknown arm joint {joint!r}; one of {sorted(target_rad)}")
    if not math.isfinite(delta_deg) or delta_deg == 0.0:
        raise StepRefusedError(f"step must be a finite non-zero angle, got {delta_deg}")
    if abs(delta_deg) > max_step_deg:
        raise StepRefusedError(
            f"step of {delta_deg:+.2f} deg exceeds the {max_step_deg:g} deg limit"
        )
    far = {
        name: math.degrees(measured_rad[name] - target)
        for name, target in target_rad.items()
        if abs(math.degrees(measured_rad[name] - target)) > max_pose_error_deg
    }
    if far:
        worst = ", ".join(f"{name} {error:+.1f} deg" for name, error in sorted(far.items()))
        raise StepRefusedError(
            f"arm is more than {max_pose_error_deg:g} deg from its target ({worst}); "
            "wait for the ramp, check the arm, or go_to_test_pose() first"
        )


class JointStepTestModule(Module):
    """Hold both arms with MIT gains and step one joint at a time."""

    config: JointStepTestConfig

    joint_state: Out[JointState]
    joint_command: Out[JointState]
    joint_gains: Out[JointState]
    step_events: Out[String]
    gravity_events: Out[String]

    def __init__(self, *, adapter: WholeBodyAdapter | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if len(self.config.kp) != len(DUAL_OPENYAM_ARM_JOINTS) or len(self.config.kd) != len(
            DUAL_OPENYAM_ARM_JOINTS
        ):
            raise ValueError(
                f"kp and kd need one value per arm joint ({len(DUAL_OPENYAM_ARM_JOINTS)})"
            )
        if len(self.config.test_pose) != len(DUAL_OPENYAM_ARM_JOINTS):
            raise ValueError("test_pose needs one angle per arm joint")
        self._adapter_override = adapter
        self._adapter: WholeBodyAdapter | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._arm_index = {name: i for i, name in enumerate(DUAL_OPENYAM_ARM_JOINTS)}
        self._target = [0.0] * len(DUAL_OPENYAM_ARM_JOINTS)
        self._kp = list(self.config.kp)
        self._kd = list(self.config.kd)
        self._accepted_kp = list(self.config.kp)
        self._accepted_kd = list(self.config.kd)
        self._gripper_hold = [0.0] * len(DUAL_OPENYAM_GRIPPER_JOINTS)
        self._measured: list[MotorState] = []
        self._ramp_goal: list[float] | None = None
        self._ramp_speed_rad_s = 0.0
        self._ramp_done = threading.Event()
        self._tick_count = 0
        self._tick_window: tuple[float, int] = (time.monotonic(), 0)
        self._measured_hz = 0.0

    # Lifecycle

    @rpc
    def start(self) -> None:
        super().start()
        adapter = self._adapter_override or self._make_adapter()
        if not adapter.connect():
            raise RuntimeError("step test: whole-body adapter failed to connect")
        activate = getattr(adapter, "activate", None)
        if callable(activate) and not activate():
            adapter.disconnect()
            raise RuntimeError("step test: whole-body adapter failed to activate")
        states = adapter.read_motor_states()
        if len(states) != len(DUAL_OPENYAM_JOINTS):
            adapter.disconnect()
            raise RuntimeError(f"expected {len(DUAL_OPENYAM_JOINTS)} joints, got {len(states)}")
        arm_count = len(DUAL_OPENYAM_ARM_JOINTS)
        with self._lock:
            self._adapter = adapter
            self._measured = states
            # Never jump: the target starts at the measured pose.
            self._target = [state.q for state in states[:arm_count]]
            self._gripper_hold = [min(max(state.q, 0.0), 1.0) for state in states[arm_count:]]
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._control_loop, name="step-test-loop", daemon=True
        )
        self._thread.start()
        logger.info(
            "Joint step test holding %d joints at %.0f Hz",
            len(DUAL_OPENYAM_JOINTS),
            self.config.loop_hz,
        )

    @rpc
    def park(self, speed_deg_s: float = REST_SPEED_DEG_S) -> str:
        """Ramp to the test pose, then down onto the rest supports (the URDF zero), and wait."""
        first = self.go_to(self._test_pose_rad(), speed_deg_s)
        second = self.go_to(self._rest_pose_rad(), speed_deg_s)
        return f"parked on the supports ({first}; {second})"

    def _rest_pose_rad(self) -> dict[str, float]:
        return dict.fromkeys(self._arm_index, 0.0)

    def _park_before_disable(self) -> None:
        """Best-effort landing inside the shutdown budget: ramp toward rest, then disable."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive() or self._adapter is None:
                return
            rest = [0.0] * len(self._target)
            travel = max(abs(t) for t in self._target)
            if travel < math.radians(1.0):
                return
            self._ramp_goal = rest
            self._ramp_speed_rad_s = math.radians(MAX_GO_TO_SPEED_DEG_S)
            self._ramp_done.clear()
        needed = travel / self._ramp_speed_rad_s
        logger.warning(
            "step test: parking before disable, %.0f deg to go at %.0f deg/s (%.1f s, budget %.0f s);"
            " call park() first next time for a gentle landing",
            math.degrees(travel),
            MAX_GO_TO_SPEED_DEG_S,
            needed,
            STOP_PARK_BUDGET_S,
        )
        if not self._ramp_done.wait(min(needed + 0.5, STOP_PARK_BUDGET_S)):
            logger.warning("step test: parking did not finish inside the budget; disabling here")

    @rpc
    def stop(self) -> None:
        self._park_before_disable()
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        adapter = self._adapter
        self._adapter = None
        if adapter is not None:
            deactivate = getattr(adapter, "deactivate", None)
            if callable(deactivate):
                try:
                    deactivate()
                except Exception:
                    logger.exception("step test: deactivate failed")
            try:
                adapter.disconnect()
            except Exception:
                logger.exception("step test: disconnect failed")
        super().stop()

    def _make_adapter(self) -> WholeBodyAdapter:
        component = dual_openyam_hardware(
            left_can_port=self.config.left_can_port,
            right_can_port=self.config.right_can_port,
            inertial_overrides=self.config.inertial_overrides,
        )
        logger.info("Joint step test using %s adapter", component.adapter_type)
        return whole_body_adapter_registry.create(
            component.adapter_type,
            dof=len(component.joints),
            hardware_id=component.hardware_id,
            address=component.address,
            domain_id=component.domain_id,
            **component.adapter_kwargs,
        )

    # Control loop

    def _control_loop(self) -> None:
        period_ns = round(1_000_000_000 / self.config.loop_hz)
        next_tick = time.perf_counter_ns()
        last = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            try:
                self._tick(now - last)
            except Exception:
                logger.exception("step test tick failed")
            last = now
            next_tick += period_ns
            sleep_ns = next_tick - time.perf_counter_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000)
            else:
                next_tick = time.perf_counter_ns()

    def _tick(self, dt: float) -> None:
        adapter = self._adapter
        if adapter is None:
            return
        states = adapter.read_motor_states()
        stamp = time.time()
        with self._lock:
            self._measured = states
            self._advance_ramp(dt)
            targets = list(self._target)
            kp = list(self._kp)
            kd = list(self._kd)
            grippers = list(self._gripper_hold)
        commands = [
            MotorCommand(q=q, dq=0.0, kp=p, kd=d, tau=0.0)
            for q, p, d in zip(targets, kp, kd, strict=True)
        ] + [MotorCommand(q=opening, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for opening in grippers]
        if not adapter.write_motor_commands(commands):
            # Revert to the last gains the motors accepted so the loop does not
            # keep sending a refused frame while the arm sits uncommanded.
            with self._lock:
                self._kp = list(self._accepted_kp)
                self._kd = list(self._accepted_kd)
            logger.warning("step test: motor command rejected; gains reverted to the last accepted")
        else:
            self._accepted_kp = kp
            self._accepted_kd = kd
        self._count_tick()
        self.joint_state.publish(
            JointState(
                ts=stamp,
                frame_id=self.frame_id,
                name=list(DUAL_OPENYAM_JOINTS),
                position=[s.q for s in states],
                velocity=[s.dq for s in states],
                effort=[s.tau for s in states],
            )
        )
        self.joint_command.publish(
            JointState(
                ts=stamp,
                frame_id=self.frame_id,
                name=list(DUAL_OPENYAM_JOINTS),
                position=[*targets, *grippers],
                velocity=[0.0] * len(DUAL_OPENYAM_JOINTS),
                effort=[0.0] * len(DUAL_OPENYAM_JOINTS),
            )
        )
        self.joint_gains.publish(
            JointState(
                ts=stamp,
                frame_id=self.frame_id,
                name=list(DUAL_OPENYAM_ARM_JOINTS),
                position=kp,
                velocity=kd,
                effort=[0.0] * len(DUAL_OPENYAM_ARM_JOINTS),
            )
        )

    def _advance_ramp(self, dt: float) -> None:
        """Move targets toward the go_to goal at the ramp speed. Caller holds the lock."""
        if self._ramp_goal is None:
            return
        max_move = self._ramp_speed_rad_s * max(dt, 0.0)
        remaining = 0.0
        for i, goal in enumerate(self._ramp_goal):
            error = goal - self._target[i]
            self._target[i] += max(-max_move, min(max_move, error))
            remaining = max(remaining, abs(goal - self._target[i]))
        if remaining < 1e-9:
            self._target = list(self._ramp_goal)
            self._ramp_goal = None
            self._ramp_done.set()

    def _count_tick(self) -> None:
        self._tick_count += 1
        t0, n0 = self._tick_window
        now = time.monotonic()
        if now - t0 >= 1.0:
            self._measured_hz = (self._tick_count - n0) / (now - t0)
            self._tick_window = (now, self._tick_count)

    # RPC surface

    @rpc
    def set_gains(self, joint: str, kp: float, kd: float) -> str:
        """Set the MIT gains of one arm joint; takes effect on the next tick."""
        index = self._require_arm_joint(joint)
        if not (math.isfinite(kp) and math.isfinite(kd)) or kp < 0.0 or kd < 0.0:
            raise StepRefusedError(f"gains must be finite and non-negative, got kp={kp} kd={kd}")
        if kp > MAX_KP or kd > MAX_KD:
            raise StepRefusedError(
                f"the Damiao MIT frame allows kp 0..{MAX_KP:g} and kd 0..{MAX_KD:g}, got kp={kp} kd={kd}"
            )
        with self._lock:
            self._kp[index] = float(kp)
            self._kd[index] = float(kd)
        return f"{joint}: kp={kp:g} kd={kd:g}"

    @rpc
    def get_gains(self) -> dict[str, tuple[float, float]]:
        """Current (kp, kd) per arm joint."""
        with self._lock:
            return {name: (self._kp[i], self._kd[i]) for name, i in self._arm_index.items()}

    @rpc
    def step(self, joint: str, delta_deg: float, hold_s: float = 2.0) -> str:
        """Move one joint's target by ``delta_deg``, hold, and report the final error.

        Refused when the step exceeds the limit, a ramp is running, or any arm
        joint is further than ``max_pose_error_deg`` from its current target.
        """
        if not math.isfinite(hold_s) or hold_s <= 0.0:
            raise StepRefusedError(f"hold_s must be positive, got {hold_s}")
        index = self._require_arm_joint(joint)
        with self._lock:
            self._require_running()
            if self._ramp_goal is not None:
                raise StepRefusedError("a go_to ramp is still running")
            check_step(
                joint,
                delta_deg,
                self._measured_arm_rad(),
                self._target_rad(),
                max_step_deg=self.config.max_step_deg,
                max_pose_error_deg=self.config.max_pose_error_deg,
            )
            before = self._target[index]
            after = before + math.radians(delta_deg)
            event = StepEvent(
                joint=joint,
                delta_deg=float(delta_deg),
                kp=self._kp[index],
                kd=self._kd[index],
                t0=time.time(),
                hold_s=float(hold_s),
                target_before_deg=math.degrees(before),
                target_after_deg=math.degrees(after),
            )
            self._target[index] = after
        self.step_events.publish(String(event.to_json()))
        logger.info("step %s %+.2f deg (kp=%g kd=%g)", joint, delta_deg, event.kp, event.kd)
        self._stop_event.wait(hold_s)
        with self._lock:
            error_deg = math.degrees(self._measured[index].q - after)
        return (
            f"{joint} {delta_deg:+.2f} deg at kp={event.kp:g} kd={event.kd:g}: "
            f"final error {error_deg:+.3f} deg"
        )

    @rpc
    def go_to(self, joints: dict[str, float], speed_deg_s: float = 10.0) -> str:
        """Ramp the named arm joints' targets linearly to ``joints`` (radians) and wait."""
        if not joints:
            raise StepRefusedError("go_to needs at least one joint")
        if not math.isfinite(speed_deg_s) or not 0.0 < speed_deg_s <= MAX_GO_TO_SPEED_DEG_S:
            raise StepRefusedError(f"speed must be in (0, {MAX_GO_TO_SPEED_DEG_S:g}] deg/s")
        for name, value in joints.items():
            self._require_arm_joint(name)
            if not math.isfinite(value):
                raise StepRefusedError(f"{name}: target {value} is not finite")
        with self._lock:
            self._require_running()
            if self._ramp_goal is not None:
                raise StepRefusedError("a go_to ramp is still running")
            goal = list(self._target)
            for name, value in joints.items():
                goal[self._arm_index[name]] = float(value)
            travel = max(abs(g - t) for g, t in zip(goal, self._target, strict=True))
            self._ramp_speed_rad_s = math.radians(speed_deg_s)
            self._ramp_done.clear()
            self._ramp_goal = goal
        duration = travel / self._ramp_speed_rad_s
        if not self._ramp_done.wait(duration + 1.0) and not self._stop_event.is_set():
            raise RuntimeError("go_to ramp did not finish in time")
        return f"moved {len(joints)} joint(s) {math.degrees(travel):.1f} deg in {duration:.1f} s"

    @rpc
    def go_to_test_pose(self, speed_deg_s: float = 10.0) -> str:
        """Ramp every arm joint to the configured test pose."""
        return self.go_to(self._test_pose_rad(), speed_deg_s)

    @rpc
    def mark(self, label: str, data: dict[str, Any] | None = None) -> str:
        """Publish a labelled JSON marker on ``gravity_events`` (e.g. the start of a pose).

        The marker carries the current targets so an offline fit can group the
        following steps by pose without trusting the driver's own bookkeeping.
        """
        if not label:
            raise StepRefusedError("mark needs a non-empty label")
        with self._lock:
            targets_deg = {
                name: math.degrees(self._target[i]) for name, i in self._arm_index.items()
            }
        event = {
            "label": label,
            "t0": time.time(),
            "targets_deg": targets_deg,
            "data": dict(data or {}),
        }
        self.gravity_events.publish(String(json.dumps(event)))
        logger.info("mark %s", label)
        return f"marked {label}"

    @rpc
    def status(self) -> dict[str, Any]:
        """Measured and target angles (deg), gains, pose error and loop health."""
        with self._lock:
            measured = self._measured_arm_rad() if self._measured else {}
            pose = self._test_pose_rad()
            return {
                "running": self._adapter is not None and self._thread is not None,
                "adapter": type(self._adapter).__name__ if self._adapter else None,
                "loop_hz": self.config.loop_hz,
                "measured_hz": round(self._measured_hz, 1),
                "ticks": self._tick_count,
                "ramping": self._ramp_goal is not None,
                "joints": {
                    name: {
                        "measured_deg": math.degrees(measured[name]) if measured else None,
                        "target_deg": math.degrees(self._target[i]),
                        "test_pose_deg": math.degrees(pose[name]),
                        "pose_error_deg": math.degrees(measured[name] - pose[name])
                        if measured
                        else None,
                        "target_error_deg": math.degrees(measured[name] - self._target[i])
                        if measured
                        else None,
                        "kp": self._kp[i],
                        "kd": self._kd[i],
                    }
                    for name, i in self._arm_index.items()
                },
                "grippers": dict(zip(DUAL_OPENYAM_GRIPPER_JOINTS, self._gripper_hold, strict=True)),
                "max_step_deg": self.config.max_step_deg,
                "max_pose_error_deg": self.config.max_pose_error_deg,
            }

    # Helpers

    def _require_arm_joint(self, joint: str) -> int:
        try:
            return self._arm_index[joint]
        except KeyError:
            raise StepRefusedError(
                f"unknown arm joint {joint!r}; one of {DUAL_OPENYAM_ARM_JOINTS}"
            ) from None

    def _require_running(self) -> None:
        if self._adapter is None or not self._measured:
            raise StepRefusedError("step test is not running (start() first)")

    def _measured_arm_rad(self) -> dict[str, float]:
        return {name: self._measured[i].q for name, i in self._arm_index.items()}

    def _test_pose_rad(self) -> dict[str, float]:
        return dict(zip(DUAL_OPENYAM_ARM_JOINTS, self.config.test_pose, strict=True))

    def _target_rad(self) -> dict[str, float]:
        return dict(zip(DUAL_OPENYAM_ARM_JOINTS, self._target, strict=True))
