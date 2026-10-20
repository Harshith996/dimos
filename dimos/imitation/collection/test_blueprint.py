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

from __future__ import annotations

import pytest

from dimos.core.coordination.blueprints import Blueprint
from dimos.imitation.collection.blueprint import (
    learning_collect_webxr_piper,
    learning_collect_webxr_xarm7,
)
from dimos.imitation.collection.episode_monitor import (
    EpisodeMonitorModule,
    EpisodeMonitorModuleConfig,
)
from dimos.imitation.collection.recorder import CollectionRecorder
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_teleop import unitree_g1_teleop


@pytest.mark.parametrize(
    "blueprint",
    [learning_collect_webxr_xarm7, learning_collect_webxr_piper, unitree_g1_teleop],
)
def test_shipped_collection_monitor_configuration_is_valid(blueprint: Blueprint) -> None:
    monitor = next(
        atom for atom in blueprint.active_blueprints if atom.module is EpisodeMonitorModule
    )
    config = EpisodeMonitorModuleConfig(**monitor.kwargs)

    assert config.task.strip()
    assert any(
        stream.name == "teleop_buttons" and stream.direction == "in" for stream in monitor.streams
    )


@pytest.mark.parametrize(
    "blueprint",
    [learning_collect_webxr_xarm7, learning_collect_webxr_piper],
)
def test_collection_streams_are_poseless(blueprint: Blueprint) -> None:
    recorder = next(atom for atom in blueprint.blueprints if atom.module is CollectionRecorder)

    assert recorder.kwargs["poseless_streams"] == [
        "color_image",
        "coordinator_joint_state",
        "status",
    ]
    assert recorder.kwargs["record_tf"] is False


@pytest.mark.parametrize(
    "blueprint",
    [learning_collect_webxr_xarm7, learning_collect_webxr_piper],
)
def test_collection_recorder_stops_after_producers(blueprint: Blueprint) -> None:
    assert blueprint.active_blueprints[0].module is CollectionRecorder


@pytest.mark.parametrize(
    "blueprint",
    [learning_collect_webxr_xarm7, learning_collect_webxr_piper],
)
def test_episode_monitor_stops_after_input_producers(blueprint: Blueprint) -> None:
    assert blueprint.active_blueprints[1].module is EpisodeMonitorModule
