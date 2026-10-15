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

"""Read the dual OpenYAM joint encoders with the motors left off.

Every read is a Damiao state query (0xCC on 0x7FF), which makes a motor report
its position without enabling it. Nothing in this module enables, commands or
zeroes a motor, and dropping the robot handle sends nothing, so the arms stay
limp the whole time: they can be read while parked on their supports or held
by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time

import can_motor_control
import numpy as np

from dimos.hardware.whole_body.damiao.config import DamiaoRuntimeConfig
from dimos.hardware.whole_body.dual_openyam_damiao.adapter import DualOpenYamDamiaoAdapter

SIDES = ("left", "right")
DEFAULT_CAN_PORTS: dict[str, str] = {"left": "follower_l", "right": "follower_r"}
JOINT_COUNT = 6
# A motor that never answered keeps the library's initial state: position 0 and
# 0 degC on both sensors. A powered Damiao motor reports room temperature or
# more, so a zero temperature marks a reading that must not be trusted.
_MIN_PLAUSIBLE_TEMPERATURE_C = 1
# Joint spread across the samples of one reading above which the arm counts as
# moving. A still arm flickers by one or two encoder counts (0.022 deg each);
# 0.1 deg is half the 0.2 deg zero-hold pass mark.
_STILL_SPREAD_RAD = float(np.radians(0.1))
# State queries to wait for every motor's first reply, about half a second.
_WARMUP_QUERIES = 50


class EncoderReadError(RuntimeError):
    """The encoders could not be read, or the reading cannot be trusted."""


@dataclass(frozen=True)
class ArmReading:
    """One arm's raw encoder values: no limit clamping, no offsets."""

    joints_rad: tuple[float, ...]
    gripper_motor_rad: float
    temperatures_c: tuple[int, ...]


@dataclass(frozen=True)
class EncoderReading:
    """Both arms, averaged over ``samples`` state queries."""

    arms: Mapping[str, ArmReading]
    samples: int
    max_spread_rad: float


def bus_frame_count(port: str, *, sysfs: Path = Path("/sys/class/net")) -> int:
    """Frames the interface has received so far."""
    return int((sysfs / port / "statistics" / "rx_packets").read_text())


def require_quiet_buses(
    ports: Mapping[str, str],
    *,
    window_s: float = 0.5,
    frame_count: Callable[[str], int] = bus_frame_count,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Refuse to share a bus with a running controller.

    A controller streams commands and the motors answer each one, so its bus
    receives frames continuously. These checks only ever query, so any traffic
    before they connect belongs to someone else.
    """
    before = {port: frame_count(port) for port in ports.values()}
    sleep(window_s)
    after = {port: frame_count(port) for port in ports.values()}
    busy = [
        f"{port} (+{after[port] - before[port]} frames)"
        for port in ports.values()
        if after[port] != before[port]
    ]
    if busy:
        raise EncoderReadError(
            "another process is talking to the arms on "
            + ", ".join(busy)
            + "; stop it (dimos stop, or Ctrl-C the teleop) and try again"
        )


class DualOpenYamEncoders:
    """Both follower arms on their own CAN buses, read-only."""

    def __init__(
        self,
        can_ports: Mapping[str, str] = DEFAULT_CAN_PORTS,
        *,
        tick_deadline_us: int = 2_000,
    ) -> None:
        self._ports = dict(can_ports)
        self._tick_deadline_us = tick_deadline_us
        adapter = DualOpenYamDamiaoAdapter(
            runtime_config=DamiaoRuntimeConfig(bus_devices=self._ports, gravity_comp=False)
        )
        # The adapter clamps feedback to the model's joint limits, and the rest
        # pose sits on the lower limits of joints 2 and 3, so its reads would
        # hide exactly the offsets calibration looks for. Take its physical
        # topology only and read the motors directly.
        self._robot: can_motor_control.Robot = adapter._build_robot()
        self._connected = False

    def connect(self) -> None:
        require_quiet_buses(self._ports)
        self._robot.connect()
        self._connected = True

    def read(self, samples: int = 20, interval_s: float = 0.01) -> EncoderReading:
        """Average ``samples`` state queries; reject silent motors and a moving arm."""
        if not self._connected:
            raise EncoderReadError("connect() first")
        # Until a motor's first reply lands its cached position is the library's
        # initial 0, so sampling starts only once every motor has answered.
        for _ in range(_WARMUP_QUERIES):
            self._query()
            if not self._silent_motors():
                break
            time.sleep(interval_s)
        silent = self._silent_motors()
        if silent:
            raise EncoderReadError(
                "no reply from "
                + ", ".join(silent)
                + "; is the arm powered and its CAN plugged in?"
            )

        joints: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
        grippers: dict[str, list[float]] = {side: [] for side in SIDES}
        for _ in range(samples):
            self._query()
            for side in SIDES:
                joints[side].append(np.asarray(self._arm(side).positions(), dtype=np.float64))
                grippers[side].append(float(self._gripper(side).motor.position))
            time.sleep(interval_s)

        arms: dict[str, ArmReading] = {}
        max_spread = 0.0
        for side in SIDES:
            stacked = np.vstack(joints[side])
            spread = np.ptp(stacked, axis=0)
            max_spread = max(max_spread, float(spread.max()))
            arms[side] = ArmReading(
                joints_rad=tuple(float(v) for v in stacked.mean(axis=0)),
                gripper_motor_rad=float(np.mean(grippers[side])),
                temperatures_c=self._temperatures(side),
            )
        if max_spread > _STILL_SPREAD_RAD:
            raise EncoderReadError(
                f"the arm moved during the reading ({np.degrees(max_spread):.2f} deg); "
                "let it settle and read again"
            )
        return EncoderReading(arms=arms, samples=samples, max_spread_rad=max_spread)

    def _query(self) -> None:
        self._robot.refresh()
        self._robot.tick(self._tick_deadline_us)

    def _silent_motors(self) -> list[str]:
        return [
            f"{side}_joint{index + 1}" if index < JOINT_COUNT else f"{side}_gripper"
            for side in SIDES
            for index, temperature in enumerate(self._temperatures(side))
            if temperature < _MIN_PLAUSIBLE_TEMPERATURE_C
        ]

    def _arm(self, side: str) -> can_motor_control.Arm:
        group = self._robot[f"{side}_arm"]
        assert isinstance(group, can_motor_control.Arm)
        return group

    def _gripper(self, side: str) -> can_motor_control.Gripper:
        group = self._robot[f"{side}_gripper"]
        assert isinstance(group, can_motor_control.Gripper)
        return group

    def _temperatures(self, side: str) -> tuple[int, ...]:
        arm = self._arm(side)
        motors = [arm[f"{side}_joint{index}"] for index in range(1, JOINT_COUNT + 1)]
        motors.append(self._gripper(side).motor)
        return tuple(min(motor.temperature_mos, motor.temperature_rotor) for motor in motors)
