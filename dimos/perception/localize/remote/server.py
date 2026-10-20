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


"""One process holding localize's three models on the GPU, shared by every localize that
points at it (``DIMOS_PERCEPTION_URL``); see ``client.py``.

Stateless per call: images, queries and boxes in, embeddings, scores and masks out. The
models run on one worker thread, in arrival order; image embeddings that arrive together run
as one batch. Each reply carries how long the call waited and computed.

    python -m dimos.perception.localize.remote.server --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from dimos.perception.localize.remote import wire
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Bumped when a reply's meaning changes; clients log it on first contact.
VERSION = "1"


@dataclass
class _Pending:
    images: list[Any]
    future: asyncio.Future[Any]
    received: float


class PerceptionServer:
    """The models and their single GPU worker. ``embedder``, ``detector`` and ``segmenter``
    are built models, or ``None`` to build localize's defaults (SigLIP, OWLv2, EdgeTAM)."""

    def __init__(
        self,
        embedder: Any = None,
        detector: Any = None,
        segmenter: Any = None,
        max_batch: int = 16,
        batch_wait_s: float = 0.005,
    ) -> None:
        if embedder is None or detector is None or segmenter is None:
            from dimos.perception.localize.dandetect import DETECTORS, EMBEDDERS, SEGMENTERS, _build

            embedder = embedder or _build(EMBEDDERS["siglip"])
            detector = detector or _build(DETECTORS["owlv2"])
            segmenter = segmenter or _build(SEGMENTERS["edgetam"])
        self.embedder, self.detector, self.segmenter = embedder, detector, segmenter
        self.max_batch, self.batch_wait_s = max_batch, batch_wait_s
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception-gpu")
        self._embeds: asyncio.Queue[_Pending] | None = None
        self.calls = 0

    async def _run(self, fn: Any, *args: Any) -> tuple[Any, float, float]:
        """``fn(*args)`` on the GPU worker: its result, the wait for the worker and the compute time."""
        queued = time.monotonic()
        started: list[float] = []

        def timed() -> Any:
            started.append(time.monotonic())
            return fn(*args)

        result = await asyncio.get_running_loop().run_in_executor(self._gpu, timed)
        done = time.monotonic()
        return result, started[0] - queued, done - started[0]

    async def embed_images(self, images: list[Any]) -> tuple[np.ndarray, float, float]:
        if self._embeds is None:
            self._embeds = asyncio.Queue()
            asyncio.get_running_loop().create_task(self._batch_embeds())
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._embeds.put(_Pending(images, future, time.monotonic()))
        return await future

    async def _batch_embeds(self) -> None:
        """Gather the image embeddings that arrive within ``batch_wait_s`` into one forward."""
        assert self._embeds is not None
        while True:
            batch = [await self._embeds.get()]
            deadline = time.monotonic() + self.batch_wait_s
            while sum(len(p.images) for p in batch) < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._embeds.get(), timeout))
                except TimeoutError:
                    break
            images = [img for p in batch for img in p.images]
            try:
                vectors, _, compute = await self._run(self._embed_vectors, images)
            except Exception as error:
                for p in batch:
                    p.future.set_exception(error)
                continue
            start = 0
            now = time.monotonic()
            for p in batch:
                n = len(p.images)
                p.future.set_result(
                    (vectors[start : start + n], now - p.received - compute, compute)
                )
                start += n

    def _embed_vectors(self, images: list[Any]) -> np.ndarray:
        embeddings = self.embedder.embed(*images)
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        return np.stack([e.to_numpy().astype(np.float32) for e in embeddings])

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.embed_text(*texts)
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        return np.stack([e.to_numpy().astype(np.float32) for e in embeddings])

    def _segment(self, image: Any, boxes: np.ndarray) -> tuple[list[int], np.ndarray]:
        from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
        from dimos.perception.detection.type.detection2d.imageDetections2D import (
            ImageDetections2D,
        )

        boxed = ImageDetections2D(
            image,
            [
                Detection2DBBox(
                    bbox=tuple(float(c) for c in box),
                    track_id=i,
                    class_id=0,
                    confidence=1.0,
                    name="",
                    ts=image.ts,
                    image=image,
                )
                for i, box in enumerate(boxes)
            ],
        )
        segmented = self.segmenter.segment(boxed)
        ids = [det.track_id for det in segmented]
        masks = np.stack([det.mask > 0 for det in segmented]) if ids else np.zeros((0, 1, 1), bool)
        return ids, masks

    def close(self) -> None:
        self._gpu.shutdown(wait=True)

    def health(self) -> dict[str, Any]:
        reply: dict[str, Any] = {
            "version": VERSION,
            "models": [type(m).__name__ for m in (self.embedder, self.detector, self.segmenter)],
            "calls": self.calls,
        }
        try:
            import torch

            if torch.cuda.is_available():
                reply["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20)
                reply["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20)
        except ImportError:
            pass
        return reply

    def warm(self) -> None:
        """Load every model's weights now, not on the first call."""
        from dimos.msgs.sensor_msgs.Image import Image

        image = Image(data=np.zeros((64, 64, 3), np.uint8), frame_id="warmup", ts=0.0)
        self._embed_vectors([image])
        self._embed_text(["object"])
        self.detector.query_score_rows(image, ["object"], threshold=1.0)
        self._segment(image, np.asarray([[8.0, 8.0, 40.0, 40.0]], np.float32))

    def app(self) -> Any:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        def reply(payload: dict[str, Any], wait: float, compute: float) -> Response:
            self.calls += 1
            payload["wait_ms"], payload["compute_ms"] = wait * 1e3, compute * 1e3
            return Response(wire.pack(payload), media_type=wire.CONTENT_TYPE)

        async def embed_images(request: Request) -> Response:
            body = wire.unpack(await request.body())
            images = [wire.image_from_wire(i, body.get("client", "")) for i in body["images"]]
            vectors, wait, compute = await self.embed_images(images)
            return reply({"vectors": vectors}, wait, compute)

        async def embed_text(request: Request) -> Response:
            body = wire.unpack(await request.body())
            vectors, wait, compute = await self._run(self._embed_text, list(body["texts"]))
            return reply({"vectors": vectors}, wait, compute)

        async def query_score_rows(request: Request) -> Response:
            body = wire.unpack(await request.body())
            image = wire.image_from_wire(body["image"], body.get("client", ""))
            (boxes, rows), wait, compute = await self._run(
                self.detector.query_score_rows,
                image,
                list(body["queries"]),
                float(body["threshold"]),
            )
            return reply({"boxes": boxes, "rows": rows}, wait, compute)

        async def segment(request: Request) -> Response:
            body = wire.unpack(await request.body())
            image = wire.image_from_wire(body["image"], body.get("client", ""))
            (ids, masks), wait, compute = await self._run(self._segment, image, body["boxes"])
            return reply({"track_ids": ids, "masks": wire.pack_masks(masks)}, wait, compute)

        async def health(request: Request) -> Response:
            return JSONResponse(self.health())

        return Starlette(
            routes=[
                Route("/embed_images", embed_images, methods=["POST"]),
                Route("/embed_text", embed_text, methods=["POST"]),
                Route("/query_score_rows", query_score_rows, methods=["POST"]),
                Route("/segment", segment, methods=["POST"]),
                Route("/health", health, methods=["GET"]),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-batch", type=int, default=16, help="image embeddings per forward")
    args = parser.parse_args()

    import uvicorn

    server = PerceptionServer(max_batch=args.max_batch)
    logger.info("perception server: loading weights")
    server.warm()
    logger.info(f"perception server: ready on {args.host}:{args.port}, {server.health()}")
    uvicorn.run(server.app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
