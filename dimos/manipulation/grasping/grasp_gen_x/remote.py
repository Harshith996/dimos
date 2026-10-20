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

"""GraspGenX proposals from a shared server (native/python/graspgenx, graspgenx_runtime.server)
instead of a model loaded in this robot's stack."""

from __future__ import annotations

import base64
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXError
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.manipulation_msgs.GraspCandidate import GraspCandidate
from dimos.msgs.manipulation_msgs.GraspCandidateArray import GraspCandidateArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Header import Header

# The request and response format this client speaks (graspgenx_runtime.server).
PROTOCOL_VERSION = 1


class RemoteGraspGenX(GraspGenSpec):
    """Sends the cloud losslessly (float32) and rebuilds the server's candidates exactly.

    ``seed`` seeds the server's model before each call, so the same cloud gives the same grasps;
    ``None`` leaves it unseeded, as a GraspGenX module in the stack runs.
    """

    def __init__(self, url: str, *, seed: int | None = 0, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.seed = seed
        self.timeout = timeout

    def health(self) -> dict[str, object]:
        with urlopen(f"{self.url}/health", timeout=self.timeout) as response:
            result: dict[str, object] = json.load(response)
            return result

    def propose_grasps(self, object_pointcloud: PointCloud2) -> GraspCandidateArray:
        if object_pointcloud.ts is None:
            raise ValueError("object pointcloud must have a timestamp")
        points = np.ascontiguousarray(object_pointcloud.points_f32(), dtype="<f4")
        body = json.dumps(
            {
                "points": base64.b64encode(points.tobytes()).decode(),
                "frame_id": object_pointcloud.frame_id,
                "ts": float(object_pointcloud.ts),
                "seed": self.seed,
            }
        ).encode()
        request = Request(
            f"{self.url}/propose", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                reply = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise GraspGenXError(f"GraspGenX server {self.url}: {exc.code} {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GraspGenXError(f"GraspGenX server {self.url} unreachable: {exc}") from exc
        if reply.get("protocol") != PROTOCOL_VERSION:
            raise GraspGenXError(
                f"GraspGenX server {self.url} speaks protocol {reply.get('protocol')}, "
                f"this client {PROTOCOL_VERSION}"
            )
        return GraspCandidateArray(
            Header(float(reply["ts"]), str(reply["frame_id"])),
            [
                GraspCandidate(
                    Pose(Vector3(*c["position"]), Quaternion(*c["orientation"])), c["score"]
                )
                for c in reply["candidates"]
            ],
        )
