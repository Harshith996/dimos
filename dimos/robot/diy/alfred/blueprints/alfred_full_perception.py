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
short-range map in ``odom`` (so a drive leaves the bench where it is) and handed to the
ManipulationModule as ``voxel_map``, re-expressed in its ``base_link``. ``PerceptionLocalize``
(SigLIP + OWLv2 + EdgeTAM, the perception stack's localize) embeds the colour feed and answers
``localize`` / ``localize_boxes`` with verified 3D instances lifted through the aligned depth, in
the planner's frame, so a found object is a place the arm can be sent. The venv's torch has no
CUDA: the models run on the CPU, a scan takes seconds.
"""

from __future__ import annotations

from typing import Any, Literal

from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXModule
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.pick_and_place_module import ArmReach, PickAndPlaceModule
from dimos.manipulation.planning.utils.roboplan_point_cloud_self_filter import (
    RoboPlanPointCloudSelfFilter,
)
from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.perception.localize.module import PerceptionLocalize
from dimos.perception.localize.types import LocalizeParams
from dimos.robot.diy.alfred.alfred_model import ALFRED_SELF_FILTER_MODEL
from dimos.robot.diy.alfred.blueprints.alfred_full import (
    BASE_FRAME,
    MANIP_VOXEL_M,
    ODOM_FRAME,
    alfred_full,
)
from dimos.robot.diy.alfred.config import ALFRED_HARDWARE
from dimos.robot.diy.alfred.gripper import (
    OPENARM_V10_GRASP_FRAME_TO_TCP,
    OPENARM_V10_SWEEP_VOLUME,
)
from dimos.utils.gpu_utils import is_cuda_available

# librealsense leaks usbfs fds (~8/s) in its worker; keep socket-accepting modules out of
# its fd table so EMFILE cannot take down the rest (see alfred_hardware.py).
RealSenseCamera.dedicated_worker = True

MANIP_VOXEL_MAP = "ManipVoxelMap"
"""The second voxel map's instance name; its streams are renamed so the nav map's stay put."""
# The map's voxels are MANIP_VOXEL_M (alfred_full), the planner's octree resolution.
# Measured from the camera on the mast (1.55 m up): to the far edge of the bench, not the
# floor beyond it. At 2.5 m the map held 7000 voxels, a third of them floor and far walls,
# and the planner spent 160 ms a collision check on them (2026-10-25).
MANIP_MAP_RANGE_M = 1.5
DEPTH_TRUNC_M = 3.0

MOUNT_TF_TOLERANCE_S = 0.3
MANIP_GRACE_DEPTH_M = 0.05
MANIP_GRAZE_COS = 0.5
MANIP_SHADOW_DEPTH_M = 0.0
SELF_FILTER_PADDING_M = 0.05
MANIP_MAP_EMIT_EVERY = 5  # of the camera's 5 Hz: 1 Hz
# How far an OpenArm gets on its own: shoulder (link1, on tf from the planner) to wrist with
# the arm straight, 0.066 + 0.154 + 0.096 + 0.121 m of URDF offsets, and the v1.0 hand's
# 0.186 m from wrist to grasp_frame. A bench object 0.70 m out needs the wrist 0.58 m from
# the shoulder for a top-down grasp: a pick says so at once instead of 12 s of IK.
OPENARM_REACH_M = 0.44
OPENARM_WRIST_TO_TIP_M = 0.186
ALFRED_ARM_REACH = {
    f"{side}_manipulator": ArmReach(
        radius=OPENARM_REACH_M,
        shoulder_frame=f"openarm_{side}_link1",
        wrist_to_tip=OPENARM_WRIST_TO_TIP_M,
    )
    for side in ("left", "right")
}

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
    GraspGenXModule,
)

GraspGenerator = Literal["heuristic", "graspgenx"]
"""Who answers the pick pipeline's ``propose_grasps``: the top-down heuristic, or GraspGenX, the
learned generator, in its own Python environment (``native/python/graspgenx``) with the v1.0
gripper's geometry (``alfred.gripper``)."""


def alfred_perception_modules(
    *, gpu: bool | None = None, localize: bool = True, grasp: GraspGenerator = "graspgenx"
) -> tuple[Any, ...]:
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
            # The filter's fingers are fixed closed; the real ones open 4.4 cm a side, and at
            # the default 1 cm an open finger in view became a voxel right where the grasp goes.
            padding_m=SELF_FILTER_PADDING_M,
        ),
        # Built in odom, on wheel odometry (good to a cm over a few metres): the bench stays
        # where it is when the base drives. Pinned to base_link, a whole-body pick that moved
        # the base 0.5 m and 0.9 rad left every voxel in the wrong place relative to the robot;
        # the planner refused all plans (COLLISION_AT_START) until the camera re-saw the bench
        # (2026-10-25). The box filter below hands it to the planner in base_link.
        RayTracingVoxelMap.blueprint(
            instance_name=MANIP_VOXEL_MAP,
            voxel_size=MANIP_VOXEL_M,
            max_range=MANIP_MAP_RANGE_M,
            world_frame=ODOM_FRAME,
            # odom -> camera is odom -> base_link (the odometry relay, ~47 Hz) and AlfredMountTf's
            # 5 Hz rigid base_link -> camera; at the default 0.1 s most clouds found no mount
            # edge close enough to their stamp and were dropped.
            tf_match_tolerance_s=MOUNT_TF_TOLERANCE_S,
            # Clearing, tuned on 60 s of this camera on the bench replayed through the mapper
            # (2026-10-25). A picked object has to leave the map, and a voxel within grace_depth
            # of a return is never cleared: at 20 cm (the default) or 10 cm a removed can kept
            # 67% or 34% of its voxels for good, at 5 cm it is gone in 3 s. graze_cos 0.5 spares
            # a surface from rays within 30 degrees of it (the far bench); at 0.7 a removed
            # can's top, seen as shallow as the bench, never quite cleared. No shadow: rays
            # carried past a return carved the layer under the bench top, which flickered
            # wherever the top straddles two voxel layers (3.4% of the workspace map changed
            # from one map to the next; 0.65% without).
            grace_depth=MANIP_GRACE_DEPTH_M,
            graze_cos=MANIP_GRAZE_COS,
            shadow_depth=MANIP_SHADOW_DEPTH_M,
            # ManipulationModule rebuilds its octree under the planner's lock for every map it
            # gets (60-70 ms each on the Orin): once a second is plenty for a bench.
            global_emit_every=MANIP_MAP_EMIT_EVERY,
        ).remappings(
            [
                (MANIP_VOXEL_MAP, "lidar", "filtered_pointcloud"),
                (MANIP_VOXEL_MAP, "loaded_map", "manip_loaded_map"),
                (MANIP_VOXEL_MAP, "global_map", "manip_global_map"),
                (MANIP_VOXEL_MAP, "local_map", "manip_local_map"),
                (MANIP_VOXEL_MAP, "local_map_fine", "manip_local_map_fine"),
            ]
        ),
        # Re-expresses the odom map in the planner's frame at each map's stamp (the planner
        # drops a map in any other frame) and lets a pick leave the object it is picking out
        # of it; a pick's box is in the planning frame, which is this one. Its output is the
        # ManipulationModule's `voxel_map` input.
        PointCloudBoxFilter.blueprint(
            target_frame=BASE_FRAME,
            tf_tolerance_s=MOUNT_TF_TOLERANCE_S,
            tf_forward_tolerance_s=MOUNT_TF_TOLERANCE_S,
        ).remappings(
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
        # The objects stay in base_link, unlike the map: PickAndPlaceModule hands an object's
        # cloud to the grasp generator as it is and requires the grasps in its planning frame,
        # so a scene in odom would fail every pick. A scan is good until the base moves.
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
            # The camera's pose comes through AlfredMountTf's 5 Hz chain (see the voxel map):
            # at the module's 0.1 s a scan's one frame now and then has no edge close enough.
            tf_tolerance_s=MOUNT_TF_TOLERANCE_S,
            tf_forward_tolerance_s=MOUNT_TF_TOLERANCE_S,
        ).remappings([(ObjectSceneRegistrationModule, "pointcloud", "scene_objects_pointcloud")]),
        PickAndPlaceModule.blueprint(planning_frame=BASE_FRAME, reach=ALFRED_ARM_REACH),
        _grasp_generator(grasp),
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


def _grasp_generator(grasp: GraspGenerator) -> Any:
    if grasp == "graspgenx":
        return GraspGenXModule.blueprint(
            gripper=OPENARM_V10_SWEEP_VOLUME,
            grasp_frame_to_tcp=OPENARM_V10_GRASP_FRAME_TO_TCP,
        )
    if grasp == "heuristic":
        return HeuristicGraspModule.blueprint()
    raise ValueError(f"unknown grasp generator {grasp!r}")


alfred_full_perception = autoconnect(alfred_full, *alfred_perception_modules()).global_config(
    n_workers=14, robot_model="alfred"
)
