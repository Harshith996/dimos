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
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.manipulation.visualization.config import NoManipulationVisualizationConfig
from dimos.protocol.tf.static_tf_publisher import StaticTfPublisher
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import build_dual_openyam_webxr
from dimos.robot.manipulators.dual_openyam.config import dual_openyam_model_config
from dimos.robot.manipulators.dual_openyam.spatial_cameras import OpenYamSpatialCameras
from dimos.robot.manipulators.dual_openyam.spatial_simulation import (
    WORKSPACE_CAMERA,
    OpenYamSpatialCoordinator,
    OpenYamSpatialSimulation,
)
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop


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

openyam_spatial_sim = (
    autoconnect(
        OpenYamSpatialSimulation.blueprint(sim_id="openyam-spatial"),
        OpenYamSpatialCoordinator.blueprint(instance_name="SpatialControlCoordinator"),
        OpenYamSpatialTeleop.blueprint(),
        OpenYamSpatialCameras.blueprint(
            instance_name="spatial_cameras", sensor=WORKSPACE_CAMERA, rate_hz=30
        ),
    )
    .global_config(zenoh_domain="openyam-spatial", viewer="none")
    .remappings(
        [
            (OpenYamSpatialTeleop, "left_controller_output", "left_cartesian_command"),
            (OpenYamSpatialTeleop, "right_controller_output", "right_cartesian_command"),
        ]
    )
)
