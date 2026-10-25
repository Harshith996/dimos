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
object localization for the bench.

    dimos run alfred-full-perception

Layered on alfred-full. The D455 streams colour for the camera panel and a decimated depth cloud
at 5 Hz; the robot's own returns are self-filtered out, what is left is voxelised into a second,
short-range map and handed to the ManipulationModule as ``voxel_map``. ``PerceptionLocalize``
(SigLIP + OWLv2 + EdgeTAM, the perception stack's localize) embeds the colour feed and answers
``localize`` / ``localize_boxes`` with verified 3D instances lifted through the aligned depth, in
the planner's frame, so a found object is a place the arm can be sent. The venv's torch has no
CUDA: the models run on the CPU, a scan takes seconds.
"""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.manipulation.planning.utils.roboplan_point_cloud_self_filter import (
    RoboPlanPointCloudSelfFilter,
)
from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.perception.localize.module import PerceptionLocalize
from dimos.perception.localize.types import LocalizeParams
from dimos.robot.diy.alfred.alfred_model import ALFRED_SELF_FILTER_MODEL
from dimos.robot.diy.alfred.blueprints.alfred_full import BASE_FRAME, alfred_full
from dimos.robot.diy.alfred.config import ALFRED_HARDWARE
from dimos.utils.gpu_utils import is_cuda_available

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

# The bench rig: one D455 on the mast looking down at a table, the robot standing still. The
# defaults are desk scale (fit on the xArm recordings); a fixed camera gives one viewpoint, so
# an instance is confirmed from one view rather than two.
ALFRED_BENCH_PARAMS = LocalizeParams(min_views=1)
# The registry's "owlv2" is built in float16 for a GPU; the venv's torch is CPU-only, where
# float16 is slow or unsupported, so the detector class is named directly (its default dtype
# is float32). The embedder and segmenter already default to float32.
LOCALIZE_DETECTOR = "dimos.perception.detection.detectors.owlv2:Owlv2Detector"
# EdgeTAM refuses a CPU ("requires a CUDA or MPS device"): no segmenter, the detector's boxes
# lift as they are. With a CUDA torch, "edgetam" is the one to put back.
LOCALIZE_SEGMENTER = "none"
# Raw colour and depth frames kept for the 1 Hz embed tail: 3 s at 15 Hz. The default (300)
# would hold 600 MB of frames.
LOCALIZE_FEED_FRAMES = 45
LOCALIZE_HORIZON_S = 120.0
# Objects further than this from the camera are not registered: the bench, not the room.
SCENE_MAX_DISTANCE_M = 2.5
# OWLv2's scores on a cluttered bench peak around 0.3-0.6 (measured: can 0.50, box 0.57,
# tape 0.30); the module's default 0.6 registered nothing.
SCENE_DETECTOR_CONFIDENCE = 0.3
# What only runs with a GPU: the models, and the pick pipeline that is nothing without them.
GPU_MODULES = (
    PerceptionLocalize,
    ObjectSceneRegistrationModule,
    PickAndPlaceModule,
    HeuristicGraspModule,
)


def alfred_perception_modules(*, gpu: bool | None = None, localize: bool = True) -> tuple[Any, ...]:
    """What the mast D455 adds to a stack: colour for the camera panel, a self-filtered depth
    cloud voxelised into the arm planner's obstacle map, and, where a GPU is there to run it,
    the perception stack's localize. On the venv's CPU-only torch SigLIP embeds a frame in ~30 s
    and OWLv2 detects in ~90 s, which would peg the box: localize is left out until the CUDA
    torch for JetPack is in (then it is ~0.5 s a scan). ``gpu`` overrides the probe.

    ``localize=False`` leaves PerceptionLocalize out and keeps the scene registration: the two
    each load OWLv2 and EdgeTAM, and together with a localize window they filled the Orin's
    16 GB and swapped. The bench stack picks through the scene registration alone."""
    modules: list[Any] = [
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
        # Dan's localize: text queries to verified 3D instances, from the colour feed and the
        # aligned depth, posed by tf in the planner's frame. `lidar` is parked: depth lifts here.
        PerceptionLocalize.blueprint(
            world_frame=BASE_FRAME,
            geometry="depth",
            mobile=False,
            params=ALFRED_BENCH_PARAMS,
            detector=LOCALIZE_DETECTOR,
            segmenter=LOCALIZE_SEGMENTER,
            feed_frames=LOCALIZE_FEED_FRAMES,
            horizon_s=LOCALIZE_HORIZON_S,
        ).remappings([(PerceptionLocalize, "lidar", "localize_lidar_unused")]),
        # The pick pipeline: prompted detections registered as 3D objects in base_link
        # (OWLv2 + EdgeTAM on the GPU), one top-down grasp per object, and PickAndPlace
        # running pregrasp, grasp, close and retreat through the planner and the arms' gripper
        # tasks. The scene's merged cloud is renamed: the camera's own `pointcloud` feeds the map.
        ObjectSceneRegistrationModule.blueprint(
            target_frame=BASE_FRAME,
            detector_backend="owlv2",
            segmentation_backend="edgetam",
            detector_confidence=SCENE_DETECTOR_CONFIDENCE,
            detect_on_request=True,
            distance_threshold=0.08,
            min_detections_for_permanent=3,
            max_distance=SCENE_MAX_DISTANCE_M,
            use_aabb=True,
        ).remappings([(ObjectSceneRegistrationModule, "pointcloud", "scene_objects_pointcloud")]),
        PickAndPlaceModule.blueprint(planning_frame=BASE_FRAME),
        HeuristicGraspModule.blueprint(),
    ]
    present = gpu if gpu is not None else bool(is_cuda_available())  # type: ignore[no-untyped-call]
    if not localize:
        modules = [
            m
            for m in modules
            if all(atom.module is not PerceptionLocalize for atom in m.blueprints)
        ]
    if not present:
        modules = [
            m for m in modules if all(atom.module not in GPU_MODULES for atom in m.blueprints)
        ]
    return tuple(modules)


alfred_full_perception = autoconnect(alfred_full, *alfred_perception_modules()).global_config(
    n_workers=14, robot_model="alfred"
)
