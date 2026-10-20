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

"""The shared server against the client robot stacks use, over a real local socket."""

from __future__ import annotations

from collections.abc import Iterator
from http.server import ThreadingHTTPServer
import threading
from typing import Any

from graspgenx_runtime.server import GraspGenXServer, handler_for, xarm_config
import numpy as np
import pytest

from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXError
from dimos.manipulation.grasping.grasp_gen_x.remote import RemoteGraspGenX
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.manipulation_msgs.GraspCandidate import GraspCandidate
from dimos.msgs.manipulation_msgs.GraspCandidateArray import GraspCandidateArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Header import Header


class FakeModule:
    """Two grasps at the cloud's first point; the second turned a little, scored lower."""

    def __init__(self) -> None:
        self.clouds: list[np.ndarray] = []

    def propose_grasps(self, cloud: PointCloud2) -> GraspCandidateArray:
        points = cloud.points_f32()
        self.clouds.append(points.copy())
        if not np.all(np.isfinite(points)):
            raise ValueError("object pointcloud XYZ values must be finite floats in metres")
        x, y, z = (float(v) for v in points[0])
        turned = Quaternion.from_euler(Vector3(np.pi, 0.0, 0.123456789))
        return GraspCandidateArray(
            Header(float(cloud.ts), cloud.frame_id),
            [
                GraspCandidate(Pose(Vector3(x, y, z), Quaternion(1.0, 0.0, 0.0, 0.0)), 0.9),
                GraspCandidate(Pose(Vector3(x, y, z + 0.01), turned), 0.4),
            ],
        )


@pytest.fixture
def served() -> Iterator[tuple[str, FakeModule]]:
    module = FakeModule()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(GraspGenXServer(module)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", module
    httpd.shutdown()
    httpd.server_close()


def cloud(points: Any) -> PointCloud2:
    return PointCloud2.from_numpy(np.asarray(points, np.float32), frame_id="world", timestamp=8.25)


def test_the_client_gets_exactly_what_the_module_proposed(served: tuple[str, FakeModule]) -> None:
    url, module = served
    sent = cloud([[0.123456, -0.5, 0.8000001], [0.2, 0.1, 0.9]])

    remote = RemoteGraspGenX(url, seed=None).propose_grasps(sent)
    local = module.propose_grasps(sent)

    assert np.array_equal(module.clouds[0], sent.points_f32())
    assert remote.header.frame_id == local.header.frame_id
    assert remote.header.timestamp == local.header.timestamp
    for r, lo in zip(remote.candidates, local.candidates, strict=True):
        assert r.score == lo.score
        assert (r.pose.position.x, r.pose.position.y, r.pose.position.z) == (
            lo.pose.position.x,
            lo.pose.position.y,
            lo.pose.position.z,
        )
        assert r.pose.orientation.to_tuple() == lo.pose.orientation.to_tuple()


def test_health_names_the_model(served: tuple[str, FakeModule]) -> None:
    url, _ = served

    health = RemoteGraspGenX(url).health()

    assert health["status"] == "ready"
    assert "GraspGenXModel@" in str(health["model"])


def test_a_bad_cloud_is_a_client_error(served: tuple[str, FakeModule]) -> None:
    url, _ = served

    with pytest.raises(GraspGenXError, match="400"):
        RemoteGraspGenX(url).propose_grasps(cloud([[np.nan, 0.0, 0.0]]))


def test_the_xarm_config_is_the_blueprints() -> None:
    config = xarm_config()

    assert config.gripper.extents_open == (0.0889, 0.030, 0.0370)
    assert config.grasp_frame_to_tcp[2][3] == pytest.approx(0.172)
