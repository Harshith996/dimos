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

"""Joint step-response test blueprint (END-310).

    dimos run dual-openyam-step-test                                   # mock hardware
    dimos run dual-openyam-step-test --left-can-port can0 --right-can-port can1
    dimos run dual-openyam-step-test-camera --left-can-port can0 --right-can-port can1 \
        --listen-host 0.0.0.0 --rerun-open none --rerun-web          # plus the overhead D405

Recording is pinned on: every stream (joint_state, joint_command, joint_gains,
step_events) lands in ``recordings/<run-id>/memory.db`` through the standard
``--record`` tap, ready for ``calibration.step_report``.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.camera.realsense.blueprints import RealSenseMountTf
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.robot.manipulators.dual_openyam.calibration.step_test import JointStepTestModule
from dimos.visualization.vis_module import vis_module

dual_openyam_step_test = JointStepTestModule.blueprint().global_config(record="sqlite")


# Step test plus the overhead D405 for the camera-in-the-loop calibration
# drivers (hand-eye capture). The camera streams `color_image` at 1280x720 and
# the recording keeps it; `--listen-host 0.0.0.0 --rerun-open none --rerun-web`
# serves the viewer. Without the camera attached the camera module fails to
# start, so mock runs use the plain `dual-openyam-step-test` blueprint and the
# drivers' `--frames` option.
OVERHEAD_CAMERA_SERIAL = "230322272156"
# The RealSense driver already built in Krishna's checkout (its own nix flake).
REALSENSE_NATIVE = (
    "/home/anvil/krishna/dimos/dimos/hardware/sensors/camera/realsense/rust/target/release/"
    "realsense_native"
)

dual_openyam_step_test_camera = autoconnect(
    JointStepTestModule.blueprint(),
    RealSenseCamera.blueprint(
        serial_number=OVERHEAD_CAMERA_SERIAL,
        width=1280,
        height=720,
        fps=15,
        enable_depth=False,
        executable=REALSENSE_NATIVE,
        build_command=None,
        source_dir=None,
    ),
    RealSenseMountTf.blueprint(),
    vis_module(viewer_backend=global_config.viewer),
).global_config(record="sqlite")
