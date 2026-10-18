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

"""Recording blueprints.

`CollectionRecorder` (a memory Recorder) captures the obs/action/status
streams to a SQLite session DB during the run and flushes it durably on
shutdown. DataPrep reads that DB afterwards.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dimos.constants import DATASETS_DIR, RECORDINGS_DIR
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.lerobot_export import LeRobotExportModule
from dimos.imitation.collection.recorder import CollectionRecorder
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import build_dual_openyam_webxr
from dimos.robot.manipulators.dual_openyam.collection import (
    DUAL_OPENYAM_COLLECTION_STREAMS,
    LEROBOT_DATAPREP_CONFIG,
    DualOpenYamCollectionRecorder,
)
from dimos.teleop.webxr.blueprints import (
    teleop_webxr_piper,
    teleop_webxr_xarm7,
)


def _session_db(robot: str) -> str:
    """Timestamped session DB path under RECORDINGS_DIR, namespaced by robot."""
    return str(RECORDINGS_DIR / f"session_{robot}_{datetime.now():%Y%m%d_%H%M%S}.db")


def _session_paths(robot: str) -> tuple[str, Path]:
    """One run's session DB and the LeRobot dataset directory built from it."""
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    return str(RECORDINGS_DIR / f"session_{robot}_{stamp}.db"), DATASETS_DIR / f"{robot}_{stamp}"


def _camera_if_real() -> tuple[Blueprint, ...]:
    """Real RealSense only off-sim. In `--simulation` the teleop coordinator's
    MujocoSimModule already publishes color_image on /camera/color_image, so a
    real camera would be redundant (and fail with no device connected)."""
    if global_config.simulation:
        return ()
    return (RealSenseCamera.blueprint(enable_pointcloud=False),)


# teleop_buttons / color_image / coordinator_joint_state / status are left to
# autoconnect — each name is unique across the composed blueprint, so it
# resolves to a stable /<name> topic shared by producer and recorder. The
# recorder captures whatever joints are present, so the coordinator's aggregate
# stream is its intended input (see dimos/control/README.md).
# These generic teleoperation labels should be customized for a specific
# training task when composing a collection blueprint.
learning_collect_webxr_xarm7 = autoconnect(
    CollectionRecorder.blueprint(
        db_path=_session_db("xarm7"),
        poseless_streams=["color_image", "coordinator_joint_state", "status"],
        record_tf=False,
    ),
    EpisodeMonitorModule.blueprint(task="Teleoperate the xArm"),
    teleop_webxr_xarm7,
    *_camera_if_real(),
)


learning_collect_webxr_piper = autoconnect(
    CollectionRecorder.blueprint(
        db_path=_session_db("piper"),
        poseless_streams=["color_image", "coordinator_joint_state", "status"],
        record_tf=False,
    ),
    EpisodeMonitorModule.blueprint(task="Teleoperate the Piper"),
    teleop_webxr_piper,
    *_camera_if_real(),
)


# Dual OpenYAM with its three headset cameras. B and Y start, save, and discard
# episodes, so the arms' zero-pose reset moves from B to A. Every saved episode
# rebuilds the LeRobot dataset from the session recording. The exporter is
# listed first so it stops last, after the recorder has flushed.
_dual_openyam_session_db, _dual_openyam_dataset = _session_paths("dual_openyam")
DUAL_OPENYAM_HUD_HINT = "B start/save · Y discard · A zero pose · grip to drive"

learning_collect_webxr_dual_openyam = autoconnect(
    LeRobotExportModule.blueprint(
        source=_dual_openyam_session_db,
        dataprep_config=LEROBOT_DATAPREP_CONFIG,
        dataset_path=_dual_openyam_dataset,
    ),
    DualOpenYamCollectionRecorder.blueprint(
        db_path=_dual_openyam_session_db,
        poseless_streams=list(DUAL_OPENYAM_COLLECTION_STREAMS),
    ),
    EpisodeMonitorModule.blueprint(task="Teleoperate the Dual OpenYAM"),
    build_dual_openyam_webxr(zero_button="A", hud_hint=DUAL_OPENYAM_HUD_HINT),
)
