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

"""The localize memory behind the scene-registration calls the pick stack makes.

PickAndPlaceModule binds ObjectSceneRegistrationSpec: a scan for prompted objects,
one object's point cloud by id, and the scene cloud with that object taken out for the
grasp contact check. ManipulationModule reads the ``objects`` stream for its planner
obstacles and the Rerun bridge draws ``detections_3d``. This module answers all of
that from PerceptionLocalize (SigLIP frame retrieval, OWLv2 boxes, EdgeTAM masks,
sightings fused across frames) instead of ObjectSceneRegistrationModule (Moondream
boxes on the latest frame), so the two stacks differ by this one module.

The overhead camera never moves, so the localize policy confirms an object from one
viewpoint. A scan waits for a frame embedded after it was asked, localizes over the
last few seconds of frames only, and keeps the hits seen in that fresh frame. Ids
survive between scans when an object stays within ``id_match_radius_m`` of where it was.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, cast
import uuid

import numpy as np
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.experimental.object import Object as DetObject, to_detection3d_array
from dimos.perception.experimental.object_scene_registration_spec import (
    ObjectSceneRegistrationSpec,
)
from dimos.perception.localize.module import PerceptionLocalize, PerceptionLocalizeConfig
from dimos.perception.localize.types import Localization, LocalizeParams
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Scene points this close (metres) to a point of the excluded object are that object.
_OBJECT_EDGE = 0.01


class LocalizeSceneRegistrationConfig(PerceptionLocalizeConfig):
    # One camera position: the overhead D405 is fixed, so a second viewpoint never comes.
    params: LocalizeParams = LocalizeParams(min_views=1)
    # Frames behind the scan's fresh frame that a scan localizes over, seconds.
    scan_window_s: float = Field(default=3.0, gt=0.0)
    # Weights load and the first posed frame; a scan before that raises.
    ready_timeout_s: float = Field(default=180.0, gt=0.0)
    # A scan reflects the table now: it waits this long for a frame embedded after it was asked.
    fresh_frame_timeout_s: float = Field(default=10.0, gt=0.0)
    # An object found within this distance (metres) of one of the same label from the
    # previous scan keeps that object's id.
    id_match_radius_m: float = Field(default=0.03, gt=0.0)
    # Planner obstacle boxes are capped to this width (metres), as the registration module does.
    max_obstacle_width: float = Field(default=0.06, ge=0.0)


@dataclass
class FoundObject:
    """An object a scan reported: its label, box centre (world, metres) and point cloud."""

    name: str
    center: np.ndarray
    cloud: PointCloud2


class LocalizeSceneRegistrationModule(PerceptionLocalize, ObjectSceneRegistrationSpec):
    """Find prompted objects with the localize memory, for pick-and-place.

    ``scan_scene`` returns one box per verified instance with a stable id,
    ``get_object_pointcloud_by_object_id`` its fused cloud, and
    ``get_full_scene_pointcloud`` the overhead depth frame of the scan lifted to the
    world frame, minus the excluded object, so the pick's contact check keeps its scene.
    """

    config: LocalizeSceneRegistrationConfig

    detections_3d: Out[Detection3DArray]
    objects: Out[list[DetObject]]

    @rpc
    def start(self) -> None:
        super().start()
        self._prompts: list[str] = []
        self._found: dict[str, FoundObject] = {}
        self._scene_snapshot: tuple[Image, Transform | None] | None = None
        self._scan_lock = threading.Lock()

    @rpc
    def set_prompts(self, text: list[str] | None = None) -> None:
        """Labels a later ``scan_scene()`` without its own looks for."""
        self._prompts = list(text or [])

    @rpc
    def scan_scene(self, text: list[str] | None = None) -> Detection3DArray:
        """The prompted objects the overhead camera sees now, one box each in the world frame.

        Raises RuntimeError when the memory is not ready within ``ready_timeout_s`` or
        no frame embedded after the call arrives within ``fresh_frame_timeout_s``.
        """
        if text is not None:
            self._prompts = list(text)
        prompts = list(dict.fromkeys(p.strip() for p in self._prompts if p.strip()))
        world = self.config.world_frame
        if not prompts:
            return to_detection3d_array([], frame_id=world)
        if not self._ready.wait(self.config.ready_timeout_s):
            raise RuntimeError(f"localize cannot answer yet: {self._stage}")
        with self._scan_lock:
            since = self._color_feed.last().ts
            depth = self._await_frame(since)
            first, head = self.index.get_time_range()
            lo = max(first, head - self.config.scan_window_s)
            index = self.index.time_range(lo, head + 1.0)
            results = self.detector.localize(
                prompts, index=index, rig=self.rig, params=self.config.params, fields=None
            )
            fresh = [
                [h for h in hits if h.point_cloud is not None and h.last_seen_timestamp >= since]
                for hits in results
            ]
            fresh = _one_label_per_object(fresh, self.config.params.cluster_radius_m)
            objects = self._register(prompts, fresh, depth)
            self._scene_snapshot = (depth, self._camera_transform(depth))
        detections = to_detection3d_array(objects, frame_id=world, ts=depth.ts)
        self.objects.publish(objects)
        self.detections_3d.publish(detections)
        self.detections.publish(detections)
        logger.info(
            f"Localize scan: {len(objects)} of {len(prompts)} prompts found "
            f"({', '.join(o.name for o in objects) or 'none'})"
        )
        return detections

    @rpc
    def get_object_pointcloud_by_object_id(self, object_id: str) -> PointCloud2 | None:
        """The world-frame point cloud of an object a scan reported, or None."""
        found = self._found.get(object_id)
        return None if found is None else found.cloud

    @rpc
    def get_object_pointcloud_by_name(self, name: str) -> PointCloud2 | None:
        """The point cloud of the object with this label a scan reported last, or None."""
        latest = [found for found in self._found.values() if found.name == name]
        return latest[-1].cloud if latest else None

    @rpc
    def get_full_scene_pointcloud(
        self,
        exclude_object_id: str | None = None,
        depth_trunc: float = 2.0,
        voxel_size: float = 0.01,
    ) -> PointCloud2 | None:
        """The overhead depth frame of the last scan in the world frame, minus one object.

        Table, neighbours and clutter included: what a grasp on the excluded object
        must keep clear of. None before the first scan or when the frame has no pose.
        """
        import open3d as o3d  # type: ignore[import-untyped]

        snapshot = self._scene_snapshot
        if snapshot is None or self._camera is None:
            return None
        depth, camera_transform = snapshot
        if camera_transform is None:
            return None
        depth_m = _depth_metres(depth)
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = self._camera.K[0], self._camera.K[4], self._camera.K[2], self._camera.K[5]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_m), intrinsic, depth_scale=1.0, depth_trunc=depth_trunc
        )
        if len(pcd.points) < 100:
            return None
        pcd = pcd.voxel_down_sample(voxel_size)
        scene = PointCloud2(pcd, frame_id=depth.frame_id, ts=depth.ts).transform(camera_transform)
        excluded = self._found.get(exclude_object_id) if exclude_object_id else None
        if excluded is None:
            return scene
        return _without(scene, excluded.cloud)

    @skill
    def detect(self, prompts: list[str]) -> str:
        """Detect objects matching the given text prompts.

        Do NOT call this tool multiple times for one query. Pass all objects in a single call.
        For example, to detect a cup and mouse, pass ["cup", "mouse"] in one call.

        Args:
            prompts: Text descriptions of concrete object categories to detect.

        Returns:
            str: Detected objects with their object_id and name.
        """
        if not prompts:
            return "No prompts provided."
        try:
            detections = self.scan_scene(prompts)
        except RuntimeError as exc:
            return str(exc)
        if not detections.detections:
            return "No objects detected."
        lines = [
            f"  - {d.results[0].hypothesis.class_id} (object_id='{d.id}')"
            for d in detections.detections
        ]
        return f"Detected {len(lines)} object(s):\n" + "\n".join(lines)

    def _await_frame(self, since: float) -> Image:
        """The first depth frame paired with a frame embedded at or after ``since``.

        ``since`` is on the camera's clock, in seconds.
        """
        deadline = time.monotonic() + self.config.fresh_frame_timeout_s
        while True:
            try:
                last = self._geometry_memory.last()
                if last.ts >= since:
                    return cast("Image", last.data)
            except LookupError:
                pass
            if self._stop.is_set() or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"no new camera frame within {self.config.fresh_frame_timeout_s:.0f}s"
                )
            time.sleep(0.05)

    def _camera_transform(self, depth: Image) -> Transform | None:
        """World pose of the depth frame at its timestamp, or None when tf has none."""
        world = self.config.world_frame
        if depth.frame_id == world:
            return None
        transform = self._tf_buffer.get(world, depth.frame_id, depth.ts, 0.1, forward_tolerance=0.2)
        if transform is None:
            logger.warning("Localize scan: no transform from the depth frame to the world frame")
        return transform

    def _register(
        self, prompts: list[str], hits: list[list[Localization]], depth: Image
    ) -> list[DetObject]:
        """One planner object per hit, with the id it had before when it is where it was."""
        radius = self.config.id_match_radius_m
        world = self.config.world_frame
        image = self._color_feed.last().data
        previous = self._found
        self._found = {}
        objects: list[DetObject] = []
        for track, (name, hit) in enumerate(
            (name, hit) for name, row in zip(prompts, hits, strict=True) for hit in row
        ):
            points = hit.point_cloud.as_numpy()[0]
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low + high) / 2.0
            near = [
                (float(np.linalg.norm(found.center - center)), object_id)
                for object_id, found in previous.items()
                if found.name == name and object_id not in self._found
            ]
            distance, object_id = min(near, default=(np.inf, ""))
            if distance > radius:
                object_id = uuid.uuid4().hex
            self._found[object_id] = FoundObject(name, center, hit.point_cloud)
            extent = high - low
            cap = self.config.max_obstacle_width
            size = Vector3(
                float(min(extent[0], cap) if cap > 0 else extent[0]),
                float(min(extent[1], cap) if cap > 0 else extent[1]),
                float(extent[2]),
            )
            position = Vector3(*(float(v) for v in center))
            # The planner's obstacle monitor and the viewer read its fields only.
            objects.append(
                DetObject(  # type: ignore[abstract]
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    track_id=track,
                    class_id=0,
                    confidence=float(hit.semantic_score),
                    name=name,
                    ts=hit.last_seen_timestamp,
                    image=image,
                    frame_id=world,
                    object_id=object_id,
                    pointcloud=hit.point_cloud,
                    center=position,
                    size=size,
                    pose=PoseStamped(
                        ts=depth.ts,
                        frame_id=world,
                        position=position,
                        orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                    ),
                    last_seen_ts=hit.last_seen_timestamp,
                )
            )
        return objects


def _one_label_per_object(
    hits: list[list[Localization]], radius: float
) -> list[list[Localization]]:
    """Each object answers to one label: of the labels that put a hit within ``radius``
    of the same spot, the best-scoring one keeps it.

    Open-vocabulary detectors give every can on the table some score for every can
    label; without this a scan reports three spam cans, three tuna cans and three
    soup cans at the same three places.
    """
    ranked = sorted(
        ((hit.semantic_score, label, hit) for label, row in enumerate(hits) for hit in row),
        key=lambda item: -item[0],
    )
    taken: list[np.ndarray] = []
    kept: list[list[Localization]] = [[] for _ in hits]
    for _, label, hit in ranked:
        center = np.asarray(hit.position_world_xyz)
        if any(np.linalg.norm(center - other) <= radius for other in taken):
            continue
        taken.append(center)
        kept[label].append(hit)
    return kept


def _depth_metres(depth: Image) -> np.ndarray:
    """The depth image as float32 metres, as the registration module reads it."""
    depth_cv = depth.to_opencv()
    if depth.format == ImageFormat.DEPTH16:
        return depth_cv.astype(np.float32) / 1000.0
    if depth_cv.dtype != np.float32:
        return depth_cv.astype(np.float32)
    return depth_cv


def _without(scene: PointCloud2, cloud: PointCloud2) -> PointCloud2:
    """``scene`` less every point within ``_OBJECT_EDGE`` of a point of ``cloud``."""
    from scipy.spatial import cKDTree

    points = scene.as_numpy()[0]
    own = cloud.as_numpy()[0]
    if not len(points) or not len(own):
        return scene
    distance, _ = cKDTree(own).query(points)
    kept = points[distance > _OBJECT_EDGE]
    return PointCloud2.from_numpy(
        kept.astype(np.float32), frame_id=scene.frame_id, timestamp=scene.ts
    )


def localize_params(**fields: Any) -> LocalizeParams:
    """``LocalizeParams`` for a fixed camera, with ``fields`` set on top of ``min_views=1``."""
    return LocalizeParams(min_views=1, **fields)
