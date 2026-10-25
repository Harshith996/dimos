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

"""One simulation blueprint, one sensor-derived WebXR viewer; no CAN devices."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.dual_openyam.spatial_cameras import OpenYamSpatialCameras
from dimos.robot.manipulators.dual_openyam.spatial_simulation import (
    WORKSPACE_CAMERA,
    OpenYamSpatialCoordinator,
    OpenYamSpatialSimulation,
)
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop

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
