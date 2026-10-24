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

"""The single-arm learning example; the artifact remains the robot definition."""

from dataclasses import replace
from pathlib import Path

from dimos.robot.manipulators.openyam.simulation import YamArtifact
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.spec import WorldConfig

FPS = 20
WIDTH = 320
HEIGHT = 180
OBSERVATION_PROFILE = "openyam-artifact-d405-rgb-v3"


def learning_world(artifact: YamArtifact) -> WorldConfig:
    robot = artifact.robot_instance()
    wrist = Camera(
        "wrist",
        "camera_d405",
        width=WIDTH,
        height=HEIGHT,
        rate_hz=FPS,
        depth=False,
    )
    robot = replace(robot, config=replace(robot.config, sensors=(wrist,)))
    return WorldConfig(
        scene=Path(__file__).parent / "scenes" / "learning.xml",
        robots={"openyam": robot},
        sensors=(
            Camera("overview", "overview", width=WIDTH, height=HEIGHT, rate_hz=FPS, depth=False),
        ),
        timestep=artifact.contract.physics_dt,
    )
