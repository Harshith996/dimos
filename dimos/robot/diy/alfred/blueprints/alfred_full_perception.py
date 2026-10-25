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

"""Alfred, the whole robot, plus what the mast D455 sees: an obstacle map for the arm planner and
2D detections for the Controller's camera panel.

    dimos run alfred-full-perception

Layered on alfred-full. The D455 streams colour for the camera panel and the detector, and a
decimated depth cloud at 5 Hz; the robot's own returns are self-filtered out, what is left is
voxelised into a second, short-range map and handed to the ManipulationModule as ``voxel_map``.
The venv's torch has no CUDA, so YOLO runs on the CPU here: one frame every four seconds.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.planning.utils.roboplan_point_cloud_self_filter import (
    RoboPlanPointCloudSelfFilter,
)
from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.perception.detection.detectors.yolo import Yolo2DDetector
from dimos.perception.detection.module2D import Detection2DModule
from dimos.robot.diy.alfred.alfred_model import ALFRED_SELF_FILTER_MODEL
from dimos.robot.diy.alfred.blueprints.alfred_full import BASE_FRAME, alfred_full
from dimos.robot.diy.alfred.config import ALFRED_HARDWARE

# librealsense leaks usbfs fds (~8/s) in its worker; keep socket-accepting modules out of
# its fd table so EMFILE cannot take down the rest (see alfred_hardware.py).
RealSenseCamera.dedicated_worker = True

MANIP_VOXEL_MAP = "ManipVoxelMap"
"""The second voxel map's instance name; its streams are renamed so the nav map's stay put."""
MANIP_VOXEL_M = 0.05  # ManipulationModule's voxel_map_resolution default, so the octrees line up
MANIP_MAP_RANGE_M = 2.5  # arm's reach, not a room
DEPTH_TRUNC_M = 3.0

MOUNT_TF_TOLERANCE_S = 0.3

D455_WIDTH, D455_HEIGHT, D455_FPS = 848, 480, 15
# Measured on the Orin NX with the venv's CPU-only torch: one yolo11n inference at imgsz 416
# takes 1.7 s on two threads, so a frame every four seconds is what fits beside navigation.
# The real fix is a CUDA torch for JetPack (yolo11n is ~15 ms on the Orin's GPU); with it,
# Yolo2DDetector picks "cuda" by itself and these caps can go.
DETECTION_HZ = 0.25
# torch and OpenCV otherwise spread one inference over every core (217% CPU measured);
# dimos's workers run with OMP_NUM_THREADS=4, so this matches the pool they already have.
DETECTION_CPU_THREADS = 4
DETECTION_IMGSZ = 416  # letterboxed from 848x480; ultralytics' 640 costs 2.4x the pixels


class _CpuYolo(Yolo2DDetector):
    """yolo11n on a bounded slice of the CPU."""

    def __init__(self) -> None:
        import cv2
        import torch

        torch.set_num_threads(DETECTION_CPU_THREADS)
        cv2.setNumThreads(DETECTION_CPU_THREADS)
        super().__init__(device="cpu", imgsz=DETECTION_IMGSZ)


# TODO(alfred): nominal D455 colour intrinsics at 848x480; the detector only needs them to
# place boxes in the camera frame. Replace with the factory ones off the camera_info stream.
D455_COLOR_INFO = CameraInfo(
    width=D455_WIDTH,
    height=D455_HEIGHT,
    K=[430.0, 0.0, 424.0, 0.0, 430.0, 240.0, 0.0, 0.0, 1.0],
    frame_id="camera_front_color_optical_frame",
)

alfred_full_perception = autoconnect(
    alfred_full,
    RealSenseCamera.blueprint(
        width=D455_WIDTH,
        height=D455_HEIGHT,
        fps=D455_FPS,
        enable_color=True,
        enable_depth=True,
        align_depth_to_color=True,
        enable_pointcloud=True,
        pointcloud_fps=5.0,
        # 3 keeps ~45k points a cloud: the self filter and the second voxel map each cost
        # about a quarter of a core at that, half a core or more at 2.
        pointcloud_decimation=3,
        depth_trunc_m=DEPTH_TRUNC_M,
        enable_infrared=False,
        enable_imu=False,
        frame_id="camera_front_link",
        serial_number=ALFRED_HARDWARE.d455_serial,
        # The mast D455 hangs upside down (as alfred_v1.urdf mounts it); pictures come out upright.
        rotate_180=True,
    ),
    # The planner's own model: whatever it drives is dropped from the cloud, so the lift
    # and arms never become obstacles to themselves.
    RoboPlanPointCloudSelfFilter.blueprint(
        model=ALFRED_SELF_FILTER_MODEL,
        # The coordinator publishes state at ~100 Hz, but a CAN read on the real arms stalls it
        # for up to ~0.3 s now and then; a state that old still describes a resting arm.
        state_tolerance_s=0.3,
        # base_link -> camera runs through AlfredMountTf's rigid mounts, published at 5 Hz:
        # a capture lands up to 100 ms from a sample, and a 300 ms old mount edge is exact.
        tf_tolerance_s=MOUNT_TF_TOLERANCE_S,
        tf_forward_tolerance_s=MOUNT_TF_TOLERANCE_S,
    ),
    # Kept in the planner's frame, the robot's own: the map rides with the base, so what was
    # seen before a drive sits wrong until the camera sees it again (ray tracing clears it).
    # Fine for arm work from a standstill, which is what this map is for.
    RayTracingVoxelMap.blueprint(
        instance_name=MANIP_VOXEL_MAP,
        voxel_size=MANIP_VOXEL_M,
        max_range=MANIP_MAP_RANGE_M,
        world_frame=BASE_FRAME,
        # base_link -> camera is AlfredMountTf's 5 Hz rigid chain; at the default 0.1 s
        # most clouds found no edge close enough to their stamp and were dropped.
        tf_match_tolerance_s=MOUNT_TF_TOLERANCE_S,
    ).remappings(
        [
            (MANIP_VOXEL_MAP, "lidar", "filtered_pointcloud"),
            (MANIP_VOXEL_MAP, "loaded_map", "manip_loaded_map"),
            (MANIP_VOXEL_MAP, "global_map", "manip_global_map"),
            (MANIP_VOXEL_MAP, "local_map", "manip_local_map"),
            (MANIP_VOXEL_MAP, "local_map_fine", "manip_local_map_fine"),
        ]
    ),
    # Lets a pick leave the object it is picking out of the map; its output is the
    # ManipulationModule's `voxel_map` input.
    PointCloudBoxFilter.blueprint().remappings(
        [
            (PointCloudBoxFilter, "cloud", "manip_global_map"),
            (PointCloudBoxFilter, "filtered_cloud", "voxel_map"),
        ]
    ),
    Detection2DModule.blueprint(
        camera_info=D455_COLOR_INFO,
        detector=_CpuYolo,
        max_freq=DETECTION_HZ,
        # The Controller draws Detection2DArray over the camera panel itself.
        publish_detection_images=False,
    ),
).global_config(n_workers=14, robot_model="alfred")
