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

"""One GraspGenX model shared by many robot stacks over HTTP, sized for the xArm gripper.

Run in this project's environment, from this directory:

  uv run --frozen --with-editable <repo> python -m graspgenx_runtime.server --port 8701

``GET /health`` answers once the model is loaded. ``POST /propose`` takes a JSON object
``{"points": <base64 little-endian float32 XYZ rows>, "frame_id", "ts", "seed"}`` and returns
the same candidates ``GraspGenXRuntimeModule.propose_grasps`` gives for that cloud, as
``{"candidates": [{"position": [x, y, z], "orientation": [x, y, z, w], "score"}], ...}`` with
the version, the time spent waiting for the model and the time inferring. Requests are served
one at a time: there is one model on one GPU. A ``seed`` (int) seeds Python, NumPy and torch
before the call; with torch's deterministic algorithms, on unless ``--nondeterministic``, the
same cloud and seed give bit-identical grasps (without them they differ by about 1e-6 m, from
GPU reduction order). ``null`` leaves the call unseeded. The server keeps nothing between
requests.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import random
import threading
import time
from typing import Any

from graspgenx_runtime.runtime import GraspGenXRuntimeModule
import numpy as np

from dimos.manipulation.grasping.grasp_gen_x.module import (
    GRASPGENX_MODEL_REPO,
    GRASPGENX_MODEL_REVISION,
    GraspGenXConfig,
)
from dimos.manipulation.grasping.grasp_gen_x.xarm import (
    XARM_GRASP_FRAME_TO_TCP,
    XARM_GRIPPER_SWEEP_VOLUME,
)
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

# Bumped when the request or response format changes.
PROTOCOL_VERSION = 1
MAX_POINTS = 200_000


def xarm_config() -> GraspGenXConfig:
    return GraspGenXConfig(
        gripper=XARM_GRIPPER_SWEEP_VOLUME,  # type: ignore[arg-type]
        grasp_frame_to_tcp=XARM_GRASP_FRAME_TO_TCP,
    )


def make_deterministic() -> None:
    """Deterministic torch kernels; cuBLAS needs its workspace size set before CUDA starts."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_cloud(body: dict[str, Any]) -> PointCloud2:
    raw = base64.b64decode(body["points"])
    points = np.frombuffer(raw, dtype="<f4")
    if points.size % 3 or not 0 < points.size // 3 <= MAX_POINTS:
        raise ValueError(f"points must hold 1 to {MAX_POINTS} float32 XYZ rows")
    return PointCloud2.from_numpy(
        points.reshape(-1, 3).astype(np.float32),
        frame_id=str(body["frame_id"]),
        timestamp=float(body["ts"]),
    )


class GraspGenXServer:
    """The loaded model, its lock and the request logic, apart from HTTP."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.lock = threading.Lock()
        self.served = 0

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "protocol": PROTOCOL_VERSION,
            "model": f"{GRASPGENX_MODEL_REPO}@{GRASPGENX_MODEL_REVISION}",
            "served": self.served,
        }

    def propose(self, body: dict[str, Any]) -> dict[str, Any]:
        cloud = decode_cloud(body)
        seed = body.get("seed")
        queued = time.monotonic()
        with self.lock:
            started = time.monotonic()
            if seed is not None:
                seed_everything(int(seed))
            proposals = self.module.propose_grasps(cloud)
            self.served += 1
        done = time.monotonic()
        return {
            "protocol": PROTOCOL_VERSION,
            "model": f"{GRASPGENX_MODEL_REPO}@{GRASPGENX_MODEL_REVISION}",
            "frame_id": proposals.header.frame_id,
            "ts": proposals.header.timestamp,
            "seed": seed,
            "wait_s": round(started - queued, 4),
            "infer_s": round(done - started, 4),
            "candidates": [
                {
                    "position": [c.pose.position.x, c.pose.position.y, c.pose.position.z],
                    "orientation": [
                        c.pose.orientation.x,
                        c.pose.orientation.y,
                        c.pose.orientation.z,
                        c.pose.orientation.w,
                    ],
                    "score": c.score,
                }
                for c in proposals.candidates
            ],
        }


def handler_for(server: GraspGenXServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200, server.health())
            else:
                self._reply(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/propose":
                self._reply(404, {"error": f"no route {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError) as exc:
                self._reply(400, {"error": f"bad request: {exc}"})
                return
            try:
                self._reply(200, server.propose(body))
            except (KeyError, ValueError) as exc:
                self._reply(400, {"error": f"bad request: {exc!r}"})
            except Exception as exc:
                self._reply(500, {"error": f"GraspGenX failed: {exc!r}"})

        def log_message(self, format: str, *args: Any) -> None:
            pass  # one line per request would flood the log at campaign rates

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8701)
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args()
    if not args.nondeterministic:
        make_deterministic()
    module = GraspGenXRuntimeModule(
        _isolated_python_runtime=True,
        **xarm_config().model_dump(exclude={"rpc_transport", "tf_transport", "g"}),
    )
    module.start()
    httpd = ThreadingHTTPServer((args.host, args.port), handler_for(GraspGenXServer(module)))
    print(f"graspgenx server ready on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        module.stop()


if __name__ == "__main__":
    main()
