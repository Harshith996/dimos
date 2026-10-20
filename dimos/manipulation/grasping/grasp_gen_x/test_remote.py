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

import base64
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

import numpy as np
import pytest

from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXError
from dimos.manipulation.grasping.grasp_gen_x.remote import PROTOCOL_VERSION, RemoteGraspGenX
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


class FakeServer:
    """Answers /propose with one grasp above the cloud's mean, scored by the seed it was sent."""

    def __init__(self, protocol: int = PROTOCOL_VERSION, status: int = 200) -> None:
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                points = np.frombuffer(base64.b64decode(body["points"]), "<f4").reshape(-1, 3)
                x, y, z = (float(v) for v in points.mean(axis=0))
                reply = {
                    "protocol": protocol,
                    "frame_id": body["frame_id"],
                    "ts": body["ts"],
                    "candidates": [
                        {
                            "position": [x, y, z + 0.1],
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                            "score": 0.5 if body["seed"] is None else body["seed"] / 10,
                        }
                    ],
                }
                data = json.dumps(reply if status == 200 else {"error": "boom"}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    yield fake
    fake.close()


def _cloud() -> PointCloud2:
    points = np.asarray([[0.5, 0.1, 0.8], [0.6, 0.1, 0.9], [0.55, 0.2, 0.85]], np.float32)
    return PointCloud2.from_numpy(points, frame_id="world", timestamp=3.5)


def test_sends_the_cloud_losslessly_and_rebuilds_the_candidates(server: FakeServer) -> None:
    cloud = _cloud()
    proposals = RemoteGraspGenX(server.url, seed=3).propose_grasps(cloud)

    sent = np.frombuffer(base64.b64decode(server.requests[0]["points"]), "<f4").reshape(-1, 3)
    assert np.array_equal(sent, cloud.points_f32())
    assert server.requests[0]["seed"] == 3
    assert proposals.header.frame_id == "world"
    assert proposals.header.timestamp == pytest.approx(3.5)
    (grasp,) = proposals.candidates
    assert grasp.score == pytest.approx(0.3)
    assert grasp.pose.position.z == pytest.approx(float(cloud.points_f32()[:, 2].mean()) + 0.1)
    assert grasp.pose.orientation.x == pytest.approx(1.0)


def test_no_seed_is_sent_as_null(server: FakeServer) -> None:
    RemoteGraspGenX(server.url, seed=None).propose_grasps(_cloud())

    assert server.requests[0]["seed"] is None


def test_a_server_error_is_a_graspgenx_error() -> None:
    fake = FakeServer(status=500)
    try:
        with pytest.raises(GraspGenXError, match="500"):
            RemoteGraspGenX(fake.url).propose_grasps(_cloud())
    finally:
        fake.close()


def test_another_protocol_is_refused() -> None:
    fake = FakeServer(protocol=PROTOCOL_VERSION + 1)
    try:
        with pytest.raises(GraspGenXError, match="protocol"):
            RemoteGraspGenX(fake.url).propose_grasps(_cloud())
    finally:
        fake.close()


def test_no_server_is_a_graspgenx_error() -> None:
    with pytest.raises(GraspGenXError, match="unreachable"):
        RemoteGraspGenX("http://127.0.0.1:9", timeout=2.0).propose_grasps(_cloud())
