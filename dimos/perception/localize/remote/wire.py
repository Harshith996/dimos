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


"""The perception server's wire format: a JSON header, then every numpy array's raw bytes.

Arrays travel losslessly (dtype, shape and buffer), so a remote model sees the exact pixels a
local one would, and its outputs come back bit for bit. A body is a 4-byte big-endian header
length, the header (the payload as JSON, each array replaced by its dtype, shape and byte
range) and the arrays' bytes.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

CONTENT_TYPE = "application/x-dimos-arrays"


def pack(payload: Any) -> bytes:
    buffers: list[bytes] = []
    offset = 0

    def encode(obj: Any) -> Any:
        nonlocal offset
        if isinstance(obj, np.ndarray):
            data = np.ascontiguousarray(obj).tobytes()
            buffers.append(data)
            ref = {"__nd__": [offset, len(data)], "dtype": obj.dtype.str, "shape": list(obj.shape)}
            offset += len(data)
            return ref
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [encode(v) for v in obj]
        return obj

    header = json.dumps(encode(payload)).encode()
    return struct.pack(">I", len(header)) + header + b"".join(buffers)


def unpack(body: bytes) -> Any:
    (length,) = struct.unpack(">I", body[:4])
    data = memoryview(body)[4 + length :]

    def decode(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "__nd__" in obj:
                start, size = obj["__nd__"]
                return np.frombuffer(
                    data[start : start + size], dtype=np.dtype(obj["dtype"])
                ).reshape(obj["shape"])
            return {k: decode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [decode(v) for v in obj]
        return obj

    return decode(json.loads(body[4 : 4 + length]))


def image_to_wire(image: Image) -> dict[str, Any]:
    return {
        "data": image.data,
        "format": image.format.value,
        "frame_id": image.frame_id,
        "ts": image.ts,
    }


def image_from_wire(wire: dict[str, Any], namespace: str = "") -> Image:
    """The image as sent; ``namespace`` prefixes its frame id so caches keyed by
    (frame id, ts) never mix two clients' frames."""
    frame_id = f"{namespace}/{wire['frame_id']}" if namespace else wire["frame_id"]
    return Image(
        data=np.array(wire["data"]),
        format=ImageFormat(wire["format"]),
        frame_id=frame_id,
        ts=wire["ts"],
    )


def pack_masks(masks: np.ndarray) -> dict[str, Any]:
    """Binary ``(n, h, w)`` masks as packed bits."""
    return {"bits": np.packbits(masks.astype(bool), axis=None), "shape": list(masks.shape)}


def unpack_masks(wire: dict[str, Any]) -> np.ndarray:
    shape = wire["shape"]
    count = int(np.prod(shape))
    return np.unpackbits(wire["bits"], count=count).astype(bool).reshape(shape)
