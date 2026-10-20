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


"""localize's three models served by a shared perception server (``server.py``).

Each class keeps the interface of the local model it stands in for, so localize picks them
by config alone: ``PERCEPTIONLOCALIZE__EMBEDDER=remote`` (and ``__DETECTOR``,
``__SEGMENTER``) with ``DIMOS_PERCEPTION_URL`` naming the server. Images go as raw pixels,
so the server's models see what local ones would. Unset, localize loads its own models.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, overload
import uuid

import numpy as np

from dimos.models.embedding.base import Embedding, EmbeddingModel
from dimos.perception.localize.remote import wire
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D

logger = setup_logger()

URL_ENV = "DIMOS_PERCEPTION_URL"
DEFAULT_URL = "http://perception-server:8765"
# A call slower than this, waiting included, is logged on its own.
SLOW_CALL_S = 2.0
# How often a client logs its calls' counts and latencies.
SUMMARY_EVERY_S = 60.0


class PerceptionClient:
    """HTTP to the server with retries, and per-endpoint latency counts. One per process,
    shared by the three models; its id namespaces this process's frames on the server."""

    _shared: PerceptionClient | None = None
    _shared_lock = threading.Lock()

    def __init__(
        self,
        url: str | None = None,
        timeout_s: float = 60.0,
        retries: int = 3,
        client_id: str | None = None,
    ) -> None:
        import httpx

        self.url = (url or os.environ.get(URL_ENV, DEFAULT_URL)).rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex[:8]
        self.retries = retries
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._lock = threading.Lock()
        self.stats: dict[str, dict[str, float]] = {}
        self._summary_at = time.monotonic() + SUMMARY_EVERY_S
        self._version_seen = False

    @classmethod
    def shared(cls) -> PerceptionClient:
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        payload = {**payload, "client": self.client_id}
        body = wire.pack(payload)
        started = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                response = self._http.post(
                    f"{self.url}/{endpoint}",
                    content=body,
                    headers={"content-type": wire.CONTENT_TYPE},
                )
                if response.status_code >= 500 and attempt < self.retries:
                    raise httpx.HTTPStatusError(
                        "server error", request=response.request, response=response
                    )
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"perception server {self.url}/{endpoint}: {error}"
                    ) from error
                time.sleep(0.5 * 2**attempt)
        reply = wire.unpack(response.content)
        self._count(endpoint, time.monotonic() - started, reply.get("wait_ms", 0.0) / 1e3)
        return reply

    def close(self) -> None:
        self._http.close()

    def health(self) -> dict[str, Any]:
        return self._http.get(f"{self.url}/health").json()

    def _count(self, endpoint: str, total_s: float, wait_s: float) -> None:
        with self._lock:
            s = self.stats.setdefault(
                endpoint, {"calls": 0, "total_s": 0.0, "wait_s": 0.0, "max_s": 0.0}
            )
            s["calls"] += 1
            s["total_s"] += total_s
            s["wait_s"] += wait_s
            s["max_s"] = max(s["max_s"], total_s)
            summary = time.monotonic() >= self._summary_at
            if summary:
                self._summary_at = time.monotonic() + SUMMARY_EVERY_S
        if total_s > SLOW_CALL_S:
            logger.warning(
                f"perception {endpoint}: {total_s:.2f}s, {wait_s:.2f}s of it queued on the server"
            )
        if summary:
            logger.info(
                "perception calls: "
                + "; ".join(
                    f"{name} {int(v['calls'])} x {v['total_s'] / v['calls'] * 1e3:.0f} ms "
                    f"(queued {v['wait_s'] / v['calls'] * 1e3:.0f} ms, max {v['max_s'] * 1e3:.0f} ms)"
                    for name, v in self.stats.items()
                )
            )


class RemoteEmbedder(EmbeddingModel):
    """SigLIP embeddings from the server; vectors arrive as float32 on the CPU."""

    device = "cpu"

    def __init__(self, client: PerceptionClient | None = None) -> None:
        self.client = client or PerceptionClient.shared()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @overload
    def embed(self, image: Image, /) -> Embedding: ...
    @overload
    def embed(self, *images: Image) -> list[Embedding]: ...
    def embed(self, *images: Image) -> Embedding | list[Embedding]:
        reply = self.client.call(
            "embed_images", {"images": [wire.image_to_wire(i) for i in images]}
        )
        embeddings = [
            Embedding(vector=np.array(v), timestamp=img.ts)
            for v, img in zip(reply["vectors"], images, strict=True)
        ]
        return embeddings[0] if len(images) == 1 else embeddings

    @overload
    def embed_text(self, text: str, /) -> Embedding: ...
    @overload
    def embed_text(self, *texts: str) -> list[Embedding]: ...
    def embed_text(self, *texts: str) -> Embedding | list[Embedding]:
        reply = self.client.call("embed_text", {"texts": list(texts)})
        embeddings = [Embedding(vector=np.array(v)) for v in reply["vectors"]]
        return embeddings[0] if len(texts) == 1 else embeddings


class RemoteDetector:
    """OWLv2 query scoring on the server (``Detector`` in ``dandetect.py``)."""

    def __init__(self, client: PerceptionClient | None = None) -> None:
        self.client = client or PerceptionClient.shared()

    def query_score_rows(
        self, image: Image, queries: list[str], threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        reply = self.client.call(
            "query_score_rows",
            {
                "image": wire.image_to_wire(image),
                "queries": list(queries),
                "threshold": float(threshold),
            },
        )
        return np.array(reply["boxes"]), np.array(reply["rows"])

    def stop(self) -> None:
        pass


class RemoteSegmenter:
    """EdgeTAM box-prompted masks from the server (``Segmenter`` in ``dandetect.py``)."""

    def __init__(self, client: PerceptionClient | None = None) -> None:
        self.client = client or PerceptionClient.shared()

    def segment(self, detections: ImageDetections2D[Any]) -> ImageDetections2D[Any]:
        from dimos.perception.detection.type.detection2d.imageDetections2D import (
            ImageDetections2D,
        )
        from dimos.perception.detection.type.detection2d.seg import Detection2DSeg

        image = detections.image
        if not len(detections):
            return ImageDetections2D(image, [])
        boxes = np.asarray([det.bbox for det in detections], dtype=np.float32)
        reply = self.client.call("segment", {"image": wire.image_to_wire(image), "boxes": boxes})
        masks = wire.unpack_masks(reply["masks"])
        dets = list(detections)
        return ImageDetections2D(
            image,
            [
                Detection2DSeg.from_sam2_result(
                    mask,
                    dets[i].track_id,
                    image,
                    class_id=dets[i].class_id,
                    name=dets[i].name,
                    confidence=dets[i].confidence,
                )
                for i, mask in zip(reply["track_ids"], masks, strict=True)
            ],
        )

    def stop(self) -> None:
        pass
