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

"""Meta Quest WebXR teleoperation of the MolmoAct2 ManiSkill Bimanual YAM."""

from datetime import datetime
from pathlib import Path

from dimos.constants import DATASETS_DIR, RECORDINGS_DIR
from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, TransportSpec, autoconnect
from dimos.core.transport import ZenohTransport
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.lerobot_export import LeRobotExportModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.visualization.viser.config import ViserVisualizationConfig
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.protocol.pubsub.impl.zenohpubsub import Topic as ZenohTopic
from dimos.robot.manipulators.dual_openyam.blueprints.basic import (
    DualOpenYamCoordinator,
    dual_openyam_trajectory_task,
)
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
    _dual_openyam_webxr_model,
    _dual_openyam_webxr_pink,
    _dual_openyam_webxr_tasks,
)
from dimos.robot.manipulators.dual_openyam.collection import (
    DUAL_OPENYAM_COLLECTION_STREAMS,
    DualOpenYamCollectionRecorder,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_GRIPPER_JOINTS
from dimos.robot.manipulators.dual_openyam.molmoact2_sim import (
    MolmoAct2SceneResetController,
    MolmoAct2YamSimModule,
)
from dimos.teleop.webxr.extensions import DualArmVideoTeleopModule

MOLMOACT2_SIM_DATAPREP_CONFIG = Path(__file__).parents[1] / "lerobot_dataprep_molmoact2_sim.json"
MOLMOACT2_SIM_HUD_HINT = "B start/save · Y discard · A zero arms · X reset scene"


def _zenoh_transport(topic: str, msg_type: type) -> TransportSpec:
    return ZenohTransport.spec(ZenohTopic(f"dimos/{topic.lstrip('/')}", msg_type))


def build_dual_openyam_molmoact2_sim(
    *,
    zero_button: str | None = "B",
    hud_hint: str | None = None,
    camera_hz: float = 15.0,
    reset_on_termination: bool = False,
) -> Blueprint:
    """Compose Quest teleoperation, transport feedback, and the ManiSkill scene."""
    return (
        autoconnect(
            DualArmVideoTeleopModule.blueprint(
                zero_button=zero_button,
                hud_hint=hud_hint,
                body_tracking_mode="optional",
            ),
            MolmoAct2YamSimModule.blueprint(
                camera_hz=camera_hz,
                reset_on_termination=reset_on_termination,
            ),
            DualOpenYamCoordinator.blueprint(
                instance_name="ControlCoordinator",
                use_transport_hardware=True,
                tasks=[
                    *_dual_openyam_webxr_tasks.values(),
                    TaskConfig(
                        name="left_arm_gripper",
                        type="gripper",
                        joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[0]],
                        priority=20,
                        stream_bind={"gripper_command": "left_gripper_command"},
                    ),
                    TaskConfig(
                        name="right_arm_gripper",
                        type="gripper",
                        joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[1]],
                        priority=20,
                        stream_bind={"gripper_command": "right_gripper_command"},
                    ),
                    dual_openyam_trajectory_task(priority=10),
                ],
            ),
            ManipulationModule.blueprint(
                model=_dual_openyam_webxr_model,
                kinematics=_dual_openyam_webxr_pink,
                visualization=ViserVisualizationConfig(),
            ),
        )
        .remappings(
            [
                (DualArmVideoTeleopModule, "left_controller_output", "left_cartesian_command"),
                (DualArmVideoTeleopModule, "left_gripper_command", "left_gripper_command"),
                (DualArmVideoTeleopModule, "right_controller_output", "right_cartesian_command"),
                (DualArmVideoTeleopModule, "right_gripper_command", "right_gripper_command"),
            ]
        )
        .transports(
            {
                ("motor_states", JointState): _zenoh_transport(
                    "/dual_openyam/motor_states", JointState
                ),
                ("motor_command", MotorCommandArray): _zenoh_transport(
                    "/dual_openyam/motor_command", MotorCommandArray
                ),
            }
        )
    )


teleop_webxr_dual_openyam_molmoact2_sim = build_dual_openyam_molmoact2_sim()


def _collection_paths() -> tuple[str, Path]:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    return (
        str(RECORDINGS_DIR / f"session_dual_openyam_molmoact2_sim_{stamp}.db"),
        DATASETS_DIR / f"dual_openyam_molmoact2_sim_{stamp}",
    )


_sim_session_db, _sim_dataset = _collection_paths()
learning_collect_webxr_dual_openyam_molmoact2_sim = autoconnect(
    LeRobotExportModule.blueprint(
        source=_sim_session_db,
        dataprep_config=MOLMOACT2_SIM_DATAPREP_CONFIG,
        dataset_path=_sim_dataset,
    ),
    DualOpenYamCollectionRecorder.blueprint(
        db_path=_sim_session_db,
        poseless_streams=list(DUAL_OPENYAM_COLLECTION_STREAMS),
    ),
    EpisodeMonitorModule.blueprint(task="Put the red and blue cubes into the center tray"),
    MolmoAct2SceneResetController.blueprint(reset_button="X"),
    build_dual_openyam_molmoact2_sim(
        zero_button="A",
        hud_hint=MOLMOACT2_SIM_HUD_HINT,
        camera_hz=10.0,
        reset_on_termination=False,
    ),
)
