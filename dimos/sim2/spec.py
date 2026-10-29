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

"""What a module's static describe_simulated(config) returns."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class Mount:
    """A pose on a body: a robot body name for robot sensors, a scene body or "world" otherwise."""

    body: str
    xyz: Vector = (0.0, 0.0, 0.0)
    rpy: Vector = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Servo:
    """A joint that starts at, and is held at, initial; with gains its actuator becomes a PD servo."""

    joint: str
    kp: float | None = None
    kd: float = 0.0
    initial: float = 0.0


@dataclass(frozen=True)
class SimulatedRobot:
    robot_id: str
    model: Path
    root_body: str
    xyz: Vector = (0.0, 0.0, 0.0)
    rpy: Vector = (0.0, 0.0, 0.0)
    floating: bool = True
    servos: tuple[Servo, ...] = ()
    # A directory whose files the model may name by file name, like its meshes.
    assets: Path | None = None


@dataclass(frozen=True)
class SimulatedSensor:
    """A camera becomes an MJCF camera; any other kind becomes a site to measure from."""

    name: str
    kind: Literal["camera", "lidar", "imu"]
    mount: Mount
    robot_id: str | None = None
    fovy: float = 60.0


Simulated = SimulatedRobot | SimulatedSensor
