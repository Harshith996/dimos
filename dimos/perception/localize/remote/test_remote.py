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


"""The remote models against the same stand-in models run locally: same outputs, by value."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from dimos.models.embedding.base import Embedding
from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg
from dimos.perception.localize.dandetect import DanDetector
from dimos.perception.localize.remote import wire
from dimos.perception.localize.remote.client import (
    PerceptionClient,
    RemoteDetector,
    RemoteEmbedder,
    RemoteSegmenter,
)
from dimos.perception.localize.remote.server import PerceptionServer


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, *images: Image) -> Any:
        self.batches.append(len(images))
        out = [
            Embedding(
                vector=img.data.reshape(-1, 3).mean(0).astype(np.float32) / 255, timestamp=img.ts
            )
            for img in images
        ]
        return out[0] if len(images) == 1 else out

    def embed_text(self, *texts: str) -> Any:
        out = [Embedding(vector=np.asarray([len(t), 1.0, 0.5], np.float32)) for t in texts]
        return out[0] if len(texts) == 1 else out


class FakeDetector:
    def __init__(self) -> None:
        self.frame_ids: list[str] = []

    def query_score_rows(
        self, image: Image, queries: list[str], threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        self.frame_ids.append(image.frame_id)
        boxes = np.asarray([[2.0, 3.0, 20.0, 30.0], [5.5, 6.25, 40.0, 41.0]], np.float32)
        rows = np.asarray([[0.9, 0.1], [0.3, 0.05]], np.float32)[:, : len(queries)]
        keep = rows.max(1) > threshold
        return boxes[keep], rows[keep]


class FakeSegmenter:
    def segment(self, detections: ImageDetections2D[Any]) -> ImageDetections2D[Any]:
        h, w = detections.image.data.shape[:2]
        out = []
        for det in detections:
            mask = np.zeros((h, w), bool)
            x1, y1, x2, y2 = (int(c) for c in det.bbox)
            mask[y1:y2, x1:x2] = True
            out.append(
                Detection2DSeg.from_sam2_result(
                    mask, det.track_id, detections.image, det.class_id, det.name, det.confidence
                )
            )
        return ImageDetections2D(detections.image, out)


def _image(seed: int = 0, ts: float = 1.5) -> Image:
    data = np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    return Image(data=data, frame_id="wrist_camera", ts=ts)


@pytest.fixture
def models() -> tuple[FakeEmbedder, FakeDetector, FakeSegmenter]:
    return FakeEmbedder(), FakeDetector(), FakeSegmenter()


@pytest.fixture
def client(models: tuple[FakeEmbedder, FakeDetector, FakeSegmenter]) -> Iterator[PerceptionClient]:
    server = PerceptionServer(*models)
    client = PerceptionClient(url="http://testserver", client_id="run7")
    client._http.close()
    with TestClient(server.app()) as http:
        client._http = http
        yield client
    server.close()


def test_wire_round_trips_arrays_and_masks() -> None:
    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    back = wire.unpack(wire.pack({"a": a, "n": [1, 2]}))
    assert back["a"].dtype == a.dtype and np.array_equal(back["a"], a)
    masks = np.random.default_rng(0).random((2, 5, 7)) > 0.5
    assert np.array_equal(wire.unpack_masks(wire.unpack(wire.pack(wire.pack_masks(masks)))), masks)


def test_remote_embeddings_match_local(client: PerceptionClient, models: Any) -> None:
    local, remote = models[0], RemoteEmbedder(client)
    images = [_image(1), _image(2)]
    for got, want in zip(remote.embed(*images), local.embed(*images), strict=True):
        assert np.array_equal(got.to_numpy(), want.to_numpy())
        assert got.timestamp == want.timestamp
    assert np.array_equal(
        remote.embed_text("blue circle").to_numpy(), local.embed_text("blue circle").to_numpy()
    )


def test_remote_scores_match_local_and_frames_are_namespaced(
    client: PerceptionClient, models: Any
) -> None:
    detector = models[1]
    boxes, rows = RemoteDetector(client).query_score_rows(_image(), ["a", "b"], threshold=0.2)
    want_boxes, want_rows = FakeDetector().query_score_rows(_image(), ["a", "b"], threshold=0.2)
    assert np.array_equal(boxes, want_boxes) and np.array_equal(rows, want_rows)
    assert detector.frame_ids == ["run7/wrist_camera"]  # never mixed with another client's frame


def test_remote_masks_match_local(client: PerceptionClient, models: Any) -> None:
    image = _image()
    boxed = ImageDetections2D(
        image,
        [
            Detection2DBBox(
                bbox=(2.0, 3.0, 20.0, 30.0),
                track_id=4,
                class_id=1,
                confidence=0.7,
                name="blue circle",
                ts=image.ts,
                image=image,
            ),
            Detection2DBBox(
                bbox=(30.0, 10.0, 60.0, 40.0),
                track_id=9,
                class_id=0,
                confidence=0.4,
                name="cube",
                ts=image.ts,
                image=image,
            ),
        ],
    )
    got, want = RemoteSegmenter(client).segment(boxed), models[2].segment(boxed)
    assert len(got) == len(want) == 2
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(g.mask, w.mask)
        assert (g.bbox, g.track_id, g.class_id, g.name, g.confidence) == (
            w.bbox,
            w.track_id,
            w.class_id,
            w.name,
            w.confidence,
        )


def test_image_embeddings_that_arrive_together_run_as_one_batch(models: Any) -> None:
    server = PerceptionServer(*models, batch_wait_s=0.05)

    async def three() -> list[Any]:
        return await asyncio.gather(*(server.embed_images([_image(i)]) for i in range(3)))

    results = asyncio.run(three())
    server.close()
    assert models[0].batches == [3]
    for i, (vectors, _, _) in enumerate(results):
        assert np.array_equal(vectors[0], models[0].embed(_image(i)).to_numpy())


def test_remote_is_picked_by_name_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIMOS_PERCEPTION_URL", "http://nowhere:1")
    monkeypatch.setattr(PerceptionClient, "_shared", None)
    dan = DanDetector("remote", "remote", "remote")
    dan.start()
    assert (type(dan.embedder), type(dan.detector), type(dan.segmenter)) == (
        RemoteEmbedder,
        RemoteDetector,
        RemoteSegmenter,
    )
    assert dan.embedder.client is dan.detector.client is dan.segmenter.client
    assert dan.embedder.client.url == "http://nowhere:1"
    shared = dan.embedder.client
    dan.stop()
    shared.close()


def test_an_unreachable_server_fails_after_its_retries() -> None:
    client = PerceptionClient(url="http://127.0.0.1:9", timeout_s=1.0, retries=1)
    with pytest.raises(RuntimeError, match="perception server"):
        RemoteDetector(client).query_score_rows(_image(), ["a"], threshold=0.1)
    client.close()
