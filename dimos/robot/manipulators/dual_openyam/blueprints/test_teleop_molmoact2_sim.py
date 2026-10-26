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

from dimos.constants import DATASETS_DIR
from dimos.imitation.collection.episode_monitor import (
    EpisodeMonitorModule,
    EpisodeMonitorModuleConfig,
)
from dimos.imitation.collection.lerobot_export import LeRobotExportModule
from dimos.imitation.dataprep.schema import DataPrepConfig
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.blueprints.teleop_molmoact2_sim import (
    MOLMOACT2_SIM_DATAPREP_CONFIG,
    MOLMOACT2_SIM_HUD_HINT,
    learning_collect_webxr_dual_openyam_molmoact2_sim,
    teleop_webxr_dual_openyam_molmoact2_sim,
)
from dimos.robot.manipulators.dual_openyam.collection import (
    DUAL_OPENYAM_COLLECTION_STREAMS,
    DualOpenYamCollectionRecorder,
)
from dimos.robot.manipulators.dual_openyam.molmoact2_sim import (
    MolmoAct2SceneResetController,
    MolmoAct2YamSimModule,
)
from dimos.teleop.webxr.camera import TeleopCamera
from dimos.teleop.webxr.extensions import DualArmVideoTeleopModule


def test_molmoact2_blueprint_uses_simulator_cameras_and_transport_feedback() -> None:
    blueprint = teleop_webxr_dual_openyam_molmoact2_sim

    assert any(atom.module is MolmoAct2YamSimModule for atom in blueprint.blueprints)
    assert not any(atom.module is TeleopCamera for atom in blueprint.blueprints)
    [coordinator] = [atom for atom in blueprint.blueprints if atom.module is DualOpenYamCoordinator]
    assert coordinator.kwargs["use_transport_hardware"] is True
    assert ("motor_states", JointState) in blueprint.transport_map
    assert ("motor_command", MotorCommandArray) in blueprint.transport_map

    [sim] = [atom for atom in blueprint.blueprints if atom.module is MolmoAct2YamSimModule]
    sim_outputs = {stream.name for stream in sim.streams if stream.direction == "out"}
    [video] = [atom for atom in blueprint.blueprints if atom.module is DualArmVideoTeleopModule]
    assert video.kwargs["body_tracking_mode"] == "optional"
    video_inputs = {stream.name for stream in video.streams if stream.direction == "in"}
    assert {
        "egocentric_image",
        "left_wrist_image",
        "right_wrist_image",
    } <= sim_outputs & video_inputs


def test_collection_blueprint_maps_a_to_reset_and_b_to_recording() -> None:
    blueprint = learning_collect_webxr_dual_openyam_molmoact2_sim
    [teleop] = [
        atom for atom in blueprint.active_blueprints if atom.module is DualArmVideoTeleopModule
    ]
    [monitor] = [
        atom for atom in blueprint.active_blueprints if atom.module is EpisodeMonitorModule
    ]
    [reset] = [
        atom for atom in blueprint.active_blueprints if atom.module is MolmoAct2SceneResetController
    ]

    monitor_config = EpisodeMonitorModuleConfig(**monitor.kwargs)
    assert teleop.kwargs["zero_button"] == "A"
    assert monitor_config.button_map == {"toggle": "B", "discard": "Y"}
    assert teleop.kwargs["zero_button"] not in monitor_config.button_map.values()
    assert teleop.kwargs["hud_hint"] == MOLMOACT2_SIM_HUD_HINT
    assert reset.kwargs["reset_button"] == "X"
    assert "X reset scene" in teleop.kwargs["hud_hint"]
    [sim] = [atom for atom in blueprint.active_blueprints if atom.module is MolmoAct2YamSimModule]
    assert sim.kwargs["camera_hz"] == 10.0
    assert sim.kwargs["reset_on_termination"] is False


def test_collection_blueprint_records_and_exports_simulator_streams() -> None:
    blueprint = learning_collect_webxr_dual_openyam_molmoact2_sim
    [recorder] = [
        atom for atom in blueprint.active_blueprints if atom.module is DualOpenYamCollectionRecorder
    ]
    [exporter] = [
        atom for atom in blueprint.active_blueprints if atom.module is LeRobotExportModule
    ]

    assert recorder.kwargs["poseless_streams"] == list(DUAL_OPENYAM_COLLECTION_STREAMS)
    assert exporter.kwargs["source"] == recorder.kwargs["db_path"]
    assert exporter.kwargs["dataprep_config"] == MOLMOACT2_SIM_DATAPREP_CONFIG
    assert exporter.kwargs["dataset_path"].parent == DATASETS_DIR

    config = DataPrepConfig.model_validate_json(MOLMOACT2_SIM_DATAPREP_CONFIG.read_text())
    camera_shapes = {spec.shape for spec in config.observation.values() if spec.dtype == "video"}
    assert camera_shapes == {(360, 640, 3)}
    assert config.sync.rate_hz == 10.0
    assert config.quality.mode == "fill"
