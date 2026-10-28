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

"""The localize memory as the pick stack's scene registration, without models or cameras."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.experimental.object_scene_registration_spec import (
    ObjectSceneRegistrationSpec,
)
from dimos.perception.localize.types import Localization
from dimos.robot.manipulators.dual_openyam.localize_registration import (
    LocalizeSceneRegistrationConfig,
    LocalizeSceneRegistrationModule,
    localize_params,
)

WORLD = "world"
CAMERA = "camera_color_optical_frame"
TS = 1000.0


def _cloud(center: tuple[float, float, float], size: float = 0.04, n: int = 200) -> PointCloud2:
    rng = np.random.default_rng(int(abs(center[0] * 1000)))
    points = np.asarray(center) + rng.uniform(-size / 2, size / 2, size=(n, 3))
    return PointCloud2.from_numpy(points.astype(np.float32), frame_id=WORLD, timestamp=TS)


def _hit(center: tuple[float, float, float], ts: float = TS, score: float = 0.5) -> Localization:
    return Localization(
        instance_id="x",
        semantic_score=score,
        ambiguity_margin=0.1,
        position_world_xyz=center,
        frame_id=WORLD,
        last_seen_timestamp=ts,
        point_cloud=_cloud(center),
        n_views=1,
    )


class _Published:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def publish(self, message: Any) -> None:
        self.messages.append(message)


class _Stream:
    """The two memory streams the scan reads: last() and the index time range."""

    def __init__(self, data: Any, ts: float) -> None:
        self._obs = SimpleNamespace(data=data, ts=ts)

    def last(self) -> Any:
        return self._obs

    def get_time_range(self) -> tuple[float, float]:
        return (TS - 30.0, self._obs.ts)

    def time_range(self, lo: float, hi: float) -> _Stream:
        return self


def _depth_frame(ts: float = TS) -> Image:
    # A flat table 0.40 m below the camera, 32 x 24 pixels, 16-bit millimetres.
    depth = np.full((24, 32), 400, dtype=np.uint16)
    return Image(data=depth, format=ImageFormat.DEPTH16, frame_id=CAMERA, ts=ts)


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> LocalizeSceneRegistrationModule:
    """A module whose localize answers are scripted and whose camera looks straight down."""
    m = LocalizeSceneRegistrationModule.__new__(LocalizeSceneRegistrationModule)
    m.config = LocalizeSceneRegistrationConfig(params=localize_params(), fresh_frame_timeout_s=0.2)
    m._prompts = []
    m._found = {}
    m._scene_snapshot = None
    m._scan_lock = threading.Lock()
    m._ready = threading.Event()
    m._ready.set()
    m._stop = threading.Event()
    m._stage = "ready"
    m._camera = CameraInfo(
        frame_id=CAMERA, width=32, height=24, K=[30.0, 0, 16.0, 0, 30.0, 12.0, 0, 0, 1]
    )
    color = Image(
        data=np.zeros((24, 32, 3), dtype=np.uint8), format=ImageFormat.RGB, frame_id=CAMERA, ts=TS
    )
    m._color_feed = _Stream(color, TS)
    m._geometry_memory = _Stream(_depth_frame(), TS + 0.1)
    m.index = _Stream(color, TS + 0.1)
    m.rig = SimpleNamespace(world_frame=WORLD)
    # Camera 0.5 m above the table, looking down: optical z becomes world -z.
    down = Quaternion(1.0, 0.0, 0.0, 0.0)
    m._tf_buffer = SimpleNamespace(
        get=lambda *a, **k: Transform(
            translation=Vector3(0.0, 0.0, 0.5), rotation=down, frame_id=WORLD, child_frame_id=CAMERA
        )
    )
    m.detections = _Published()
    m.detections_3d = _Published()
    m.objects = _Published()
    m._answers: list[list[Localization]] = []
    m.detector = SimpleNamespace(localize=lambda prompts, **kw: m._answers)
    return m


def test_module_implements_the_registration_spec() -> None:
    for name in (
        "set_prompts",
        "scan_scene",
        "get_object_pointcloud_by_name",
        "get_object_pointcloud_by_object_id",
        "get_full_scene_pointcloud",
    ):
        assert callable(getattr(LocalizeSceneRegistrationModule, name))
    assert ObjectSceneRegistrationSpec in LocalizeSceneRegistrationModule.__mro__


def test_fixed_camera_policy_confirms_from_one_view() -> None:
    assert localize_params().min_views == 1
    assert localize_params(accept_score=0.2).accept_score == 0.2
    assert LocalizeSceneRegistrationConfig().params.min_views == 1


def test_scan_reports_fresh_hits_with_ids_clouds_and_planner_objects(
    module: LocalizeSceneRegistrationModule,
) -> None:
    module._answers = [[_hit((0.2, -0.1, 0.03))], [_hit((0.3, 0.1, 0.04))], []]

    detections = module.scan_scene(["spam can", "pudding box", "tuna fish can"])

    assert [d.results[0].hypothesis.class_id for d in detections.detections] == [
        "spam can",
        "pudding box",
    ]
    ids = [d.id for d in detections.detections]
    assert all(len(i) == 32 for i in ids)
    assert module.get_object_pointcloud_by_object_id(ids[0]) is not None
    assert module.get_object_pointcloud_by_name("pudding box") is not None
    assert module.get_object_pointcloud_by_name("tuna fish can") is None
    [published] = module.objects.messages
    assert [o.name for o in published] == ["spam can", "pudding box"]
    assert [o.object_id for o in published] == ids
    assert all(o.size.x <= 0.06 and o.size.y <= 0.06 for o in published)
    assert module.detections_3d.messages and module.detections.messages


def test_scan_drops_hits_seen_before_it_was_asked(module: LocalizeSceneRegistrationModule) -> None:
    module._answers = [[_hit((0.2, -0.1, 0.03), ts=TS - 5.0)]]

    assert module.scan_scene(["spam can"]).detections_length == 0


def test_an_object_that_stayed_put_keeps_its_id(module: LocalizeSceneRegistrationModule) -> None:
    module._answers = [[_hit((0.2, -0.1, 0.03))]]
    [first] = module.scan_scene(["spam can"]).detections
    module._answers = [[_hit((0.205, -0.095, 0.03))]]
    [second] = module.scan_scene(["spam can"]).detections
    module._answers = [[_hit((0.3, -0.1, 0.03))]]
    [moved] = module.scan_scene(["spam can"]).detections

    assert second.id == first.id
    assert moved.id != first.id


def test_scene_cloud_is_the_depth_frame_in_the_world_minus_the_object(
    module: LocalizeSceneRegistrationModule,
) -> None:
    assert module.get_full_scene_pointcloud() is None
    module._answers = [[_hit((0.0, 0.0, 0.10))]]
    [spam] = module.scan_scene(["spam can"]).detections

    scene = module.get_full_scene_pointcloud(voxel_size=0.005)
    assert scene is not None and scene.frame_id == WORLD
    points = scene.as_numpy()[0]
    # The table plane sits 0.40 m under a camera 0.50 m up: z = 0.10 in the world.
    assert np.allclose(points[:, 2], 0.10, atol=0.01)

    without = module.get_full_scene_pointcloud(exclude_object_id=spam.id, voxel_size=0.005)
    assert without is not None
    kept = without.as_numpy()[0]
    assert len(kept) < len(points)
    assert np.all(np.linalg.norm(kept[:, :2], axis=1) > 0.015)


def test_scan_waits_for_a_fresh_frame_and_says_when_none_comes(
    module: LocalizeSceneRegistrationModule,
) -> None:
    module._geometry_memory = _Stream(_depth_frame(TS - 1.0), TS - 1.0)

    with pytest.raises(RuntimeError, match="no new camera frame"):
        module.scan_scene(["spam can"])


def test_detect_skill_reads_like_the_registration_modules(
    module: LocalizeSceneRegistrationModule,
) -> None:
    module._answers = [[_hit((0.2, -0.1, 0.03))]]
    text = module.detect(["spam can"])
    assert text.startswith("Detected 1 object(s):") and "spam can (object_id='" in text
    module._answers = [[]]
    assert module.detect(["bowl"]) == "No objects detected."
