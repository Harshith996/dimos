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

"""The same sensor-derived WebXR view on sim2 or physical dual-YAM devices."""

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.manipulation.visualization.config import NoManipulationVisualizationConfig
from dimos.perception.detection.detectors.yoloe import YoloePromptMode
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.protocol.testing.link import LinkConditions
from dimos.protocol.tf.static_tf_publisher import StaticTfPublisher
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import build_dual_openyam_webxr
from dimos.robot.manipulators.dual_openyam.collection import DualOpenYamCollection
from dimos.robot.manipulators.dual_openyam.config import dual_openyam_model_config
from dimos.robot.manipulators.dual_openyam.spatial_cameras import OpenYamSpatialCameras
from dimos.robot.manipulators.dual_openyam.spatial_planning import spatial_model_config
from dimos.robot.manipulators.dual_openyam.spatial_simulation import (
    OpenYamSpatialCoordinator,
    OpenYamSpatialSimulation,
)
from dimos.robot.manipulators.dual_openyam.spatial_station import SpatialStation
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop
from dimos.robot.manipulators.dual_openyam.stations.anvil import ANVIL


def _object_perception() -> Blueprint:
    """Request-driven observations from the station's calibrated RGB-D stream."""
    return ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        prompt_mode=YoloePromptMode.PROMPT,
        detect_on_request=True,
        min_detections_for_permanent=1,
        distance_threshold=0.05,
        detector_confidence=0.25,
    )


def build_openyam_spatial(*, model: RobotModelConfig | None = None) -> Blueprint:
    """Physical camera/control wiring; configure CAN ports and measured mount frames.

    A custom calibrated robot model is shared by IK, joint limits and display.
    The default uses the existing physical dual-YAM model, not the sim artifact.
    """
    robot_model = model or dual_openyam_model_config()
    return (
        autoconnect(
            build_dual_openyam_webxr(
                visualization=NoManipulationVisualizationConfig(),
                teleop=OpenYamSpatialTeleop,
                teleop_kwargs={"robot_model": robot_model, "zero_button": None},
                robot_model=robot_model,
            ),
            RealSenseCamera.blueprint(instance_name="workspace_camera", align_depth_to_color=True),
            StaticTfPublisher.blueprint(instance_name="workspace_mount", publish_hz=20),
            _object_perception(),
        )
        .remappings(
            [
                (OpenYamSpatialTeleop, "left_wrist", "camera1"),
                (OpenYamSpatialTeleop, "right_wrist", "camera2"),
            ]
        )
        .global_config(viewer="none")
    )


openyam_spatial = autoconnect(build_openyam_spatial())


def build_openyam_spatial_sim(
    station: SpatialStation = ANVIL, *, link_conditions: LinkConditions | None = None
) -> Blueprint:
    """Bind one station description to physics, sensors, control and display."""
    wrist = station.wrists["left"]
    return (
        autoconnect(
            OpenYamSpatialSimulation.blueprint(sim_id="openyam-spatial", station=station),
            OpenYamSpatialCoordinator.blueprint(
                instance_name="SpatialControlCoordinator", base_poses=station.bases
            ),
            ManipulationModule.blueprint(model=spatial_model_config(bases=station.bases)),
            _object_perception(),
            OpenYamSpatialTeleop.blueprint(
                base_poses=station.bases, link_conditions=link_conditions
            ),
            DualOpenYamCollection.blueprint(
                camera_source="streams",
                camera_ids=("sim2:left/wrist", "sim2:right/wrist"),
                width=wrist.width,
                height=wrist.height,
                fps=int(wrist.rate_hz),
            ),
            OpenYamSpatialCameras.blueprint(
                instance_name="spatial_cameras",
                sensor=station.workspace,
                wrists=station.wrists,
                rate_hz=2 * max(station.workspace.rate_hz, wrist.rate_hz),
            ),
        )
        .global_config(zenoh_domain="openyam-spatial", viewer="none")
        .remappings(
            [
                (OpenYamSpatialTeleop, "left_controller_output", "left_cartesian_command"),
                (OpenYamSpatialTeleop, "right_controller_output", "right_cartesian_command"),
                (DualOpenYamCollection, "left_image", "left_wrist"),
                (DualOpenYamCollection, "right_image", "right_wrist"),
            ]
        )
    )


openyam_spatial_sim = autoconnect(build_openyam_spatial_sim())
