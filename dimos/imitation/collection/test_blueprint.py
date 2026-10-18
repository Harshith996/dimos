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

from dimos.constants import DATASETS_DIR
from dimos.core.coordination.blueprints import Blueprint
from dimos.imitation.collection.blueprint import (
    learning_collect_webxr_dual_openyam,
    learning_collect_webxr_piper,
    learning_collect_webxr_xarm7,
)
from dimos.imitation.collection.episode_monitor import (
    EpisodeMonitorModule,
    EpisodeMonitorModuleConfig,
)
from dimos.imitation.collection.lerobot_export import LeRobotExportModule
from dimos.imitation.collection.recorder import CollectionRecorder
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
    EGOCENTRIC_CAMERA,
    LEFT_WRIST_CAMERA,
    RIGHT_WRIST_CAMERA,
)
from dimos.robot.manipulators.dual_openyam.collection import (
    DUAL_OPENYAM_CAMERA_STREAMS,
    LEROBOT_DATAPREP_CONFIG,
    DualOpenYamCollectionRecorder,
)
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_teleop import unitree_g1_teleop
from dimos.teleop.webxr.extensions import ArmTeleopModule

AGGREGATE = "coordinator_joint_state"


@pytest.mark.parametrize(
    "blueprint",
    [
        learning_collect_webxr_xarm7,
        learning_collect_webxr_piper,
        learning_collect_webxr_dual_openyam,
        unitree_g1_teleop,
    ],
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


@pytest.mark.parametrize(
    "blueprint",
    [
        learning_collect_webxr_xarm7,
        learning_collect_webxr_piper,
        learning_collect_webxr_dual_openyam,
    ],
)
def test_collection_status_is_wired_to_webxr_hud(blueprint: Blueprint) -> None:
    hud = next(atom for atom in blueprint.blueprints if issubclass(atom.module, ArmTeleopModule))
    status = next(stream for stream in hud.streams if stream.name == "status")

    assert status.direction == "in"
    assert status.type.__name__ == "String"


def _joint_streams(blueprint: Blueprint) -> dict[tuple[str, str], str]:
    """(instance, port) -> effective stream name, as ModuleCoordinator pairs streams."""
    return {
        (atom.name, stream.name): blueprint.remapping_map.get((atom.name, stream.name), stream.name)
        for atom in blueprint.active_blueprints
        for stream in atom.streams
        if stream.type is JointState
    }


@pytest.mark.parametrize("blueprint", [learning_collect_webxr_xarm7, learning_collect_webxr_piper])
def test_recorder_reads_aggregate_joint_state(blueprint: Blueprint) -> None:
    streams = _joint_streams(blueprint)

    # Plain name pairing on both ends, no remap in between. The coordinator
    # atom carries its explicit instance_name (the RPC lookup contract).
    assert streams[("collectionrecorder", AGGREGATE)] == AGGREGATE
    assert streams[("ControlCoordinator", AGGREGATE)] == AGGREGATE
    assert not [port for _instance, port in streams if port.endswith("_joints")]


def test_dual_openyam_recorder_captures_every_camera_panel() -> None:
    blueprint = learning_collect_webxr_dual_openyam
    recorder = blueprint.active_blueprints[1]
    assert recorder.module is DualOpenYamCollectionRecorder
    assert blueprint.active_blueprints[2].module is EpisodeMonitorModule
    assert recorder.kwargs["poseless_streams"] == [
        *DUAL_OPENYAM_CAMERA_STREAMS,
        AGGREGATE,
        "status",
    ]

    camera_streams = {
        blueprint.remapping_map[(camera, "color_image")]
        for camera in (EGOCENTRIC_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA)
    }
    recorder_ports = {stream.name for stream in recorder.streams if stream.direction == "in"}
    assert camera_streams == set(DUAL_OPENYAM_CAMERA_STREAMS)
    assert camera_streams <= recorder_ports

    streams = _joint_streams(blueprint)
    assert streams[(recorder.name, AGGREGATE)] == AGGREGATE
    assert streams[("ControlCoordinator", AGGREGATE)] == AGGREGATE


def test_dual_openyam_zero_reset_keeps_clear_of_the_episode_buttons() -> None:
    blueprint = learning_collect_webxr_dual_openyam
    teleop = next(
        atom for atom in blueprint.active_blueprints if issubclass(atom.module, ArmTeleopModule)
    )
    monitor = next(
        atom for atom in blueprint.active_blueprints if atom.module is EpisodeMonitorModule
    )
    episode_buttons = set(EpisodeMonitorModuleConfig(**monitor.kwargs).button_map.values())

    assert teleop.kwargs["zero_button"] == "A"
    assert teleop.kwargs["zero_button"] not in episode_buttons


def test_dual_openyam_exports_the_session_it_records() -> None:
    blueprint = learning_collect_webxr_dual_openyam
    exporter, recorder = blueprint.active_blueprints[:2]
    assert exporter.module is LeRobotExportModule
    assert exporter.kwargs["source"] == recorder.kwargs["db_path"]
    assert exporter.kwargs["dataprep_config"] == LEROBOT_DATAPREP_CONFIG
    assert exporter.kwargs["dataset_path"].parent == DATASETS_DIR
    assert exporter.kwargs["dataset_path"].name.startswith("dual_openyam_")

    teleop = next(
        atom for atom in blueprint.active_blueprints if issubclass(atom.module, ArmTeleopModule)
    )
    status = next(stream for stream in exporter.streams if stream.name == "status")
    export_status = next(stream for stream in teleop.streams if stream.name == "export_status")
    assert status.direction == "in"
    assert export_status.direction == "in"
    assert "B" in teleop.kwargs["hud_hint"] and "Y" in teleop.kwargs["hud_hint"]
