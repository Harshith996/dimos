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

"""Local experiment: carry one LCM message in JSON-RPC calls, two ways.

Run with ``python -m dimos.protocol.rpc.demo_lcm_jsonrpc``. This does not send
network traffic or change the RPC backend. The binary frame is demo-only.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import median
import struct
from time import perf_counter_ns
from typing import Any, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.protocol import DimosMsg
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec.json_rpc import (
    JsonRpcId,
    JsonRpcParams,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
)

Mode: TypeAlias = Literal["base64", "binary"]
MODES: tuple[Mode, ...] = ("base64", "binary")
MAGIC = b"DJR1"
PREFIX = struct.Struct("!4sI")  # Magic/version, then uint32 big-endian JSON length.
REQUEST_ID = "lcm-demo:1"
METHOD = "Demo/inspect"
DEFAULT_POINTS = 100_000
DEFAULT_ITERATIONS = 20
SEED = 42


class LcmValue(BaseModel):
    """Demo convention for one message argument or result, not a general type system."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: str
    encoding: Literal["lcm"] = "lcm"
    byte_length: int = Field(ge=0)
    data: str | None = None  # Base64 text in JSON mode; absent in binary mode.


def _describe(raw: bytes, type_name: str, mode: Mode) -> dict[str, Any]:
    value = LcmValue(type=type_name, byte_length=len(raw))
    if mode == "base64":
        value.data = base64.b64encode(raw).decode("ascii")
    return value.model_dump(exclude_none=True)


def _frame(header: bytes, raw: bytes, mode: Mode) -> bytes:
    if mode == "base64":
        return header
    return PREFIX.pack(MAGIC, len(header)) + header + raw


def pack(raw: bytes, type_name: str, mode: Mode) -> bytes:
    """Wrap already-encoded LCM bytes; no message conversion is timed here."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        id=REQUEST_ID,
        method=METHOD,
        params=JsonRpcParams(args=[_describe(raw, type_name, mode)]),
    )
    return _frame(request.model_dump_json().encode("utf-8"), raw, mode)


def pack_response(
    raw: bytes, type_name: str, mode: Mode, request_id: JsonRpcId = REQUEST_ID
) -> bytes:
    """Place one LCM return value under result and echo the request ID."""
    response = JsonRpcSuccessResponse(
        jsonrpc="2.0", id=request_id, result=_describe(raw, type_name, mode)
    )
    return _frame(response.model_dump_json().encode("utf-8"), raw, mode)


def split_frame(wire: bytes, mode: Mode) -> tuple[bytes, bytes]:
    """Separate the JSON document from its optional inline binary payload."""
    if mode == "base64":
        return wire, b""
    if len(wire) < PREFIX.size:
        raise ValueError("Truncated binary prefix")
    magic, header_length = PREFIX.unpack_from(wire)
    if magic != MAGIC:
        raise ValueError("Unknown binary framing version")
    end = PREFIX.size + header_length
    if end > len(wire):
        raise ValueError("Truncated JSON header")
    return wire[PREFIX.size : end], wire[end:]


def unpack(wire: bytes, expected_type: str, mode: Mode) -> bytes:
    """Validate the request and expected parameter type before selecting a decoder."""
    header, raw = split_frame(wire, mode)
    request = JsonRpcRequest.model_validate_json(header)
    if request.id != REQUEST_ID or request.method != METHOD:
        raise ValueError("Unexpected request ID or method")
    if len(request.params.args) != 1 or request.params.kwargs:
        raise ValueError("Demo/inspect expects exactly one positional message")
    return _decode_value(request.params.args[0], raw, expected_type, mode)


def unpack_response(
    wire: bytes, expected_type: str, mode: Mode, expected_id: JsonRpcId = REQUEST_ID
) -> bytes:
    """Validate a successful response, its ID and its declared return type."""
    header, raw = split_frame(wire, mode)
    response = JsonRpcSuccessResponse.model_validate_json(header)
    if response.id != expected_id:
        raise ValueError("Unexpected response ID")
    return _decode_value(response.result, raw, expected_type, mode)


def _decode_value(descriptor: Any, raw: bytes, expected_type: str, mode: Mode) -> bytes:
    value = LcmValue.model_validate(descriptor)
    if value.type != expected_type:
        raise ValueError(f"Expected {expected_type}, received {value.type}")
    if mode == "base64":
        if value.data is None:
            raise ValueError("Missing base64 data")
        raw = base64.b64decode(value.data, validate=True)
    elif value.data is not None:
        raise ValueError("Binary mode must carry data outside JSON")
    if len(raw) != value.byte_length:
        raise ValueError("LCM payload length does not match its descriptor")
    return raw


def verify_message(original: DimosMsg, decoded: DimosMsg) -> None:
    """Check actual values in addition to byte-for-byte re-encoding."""
    assert decoded.lcm_encode() == original.lcm_encode(), "LCM bytes changed"
    if isinstance(original, Twist):
        assert isinstance(decoded, Twist)
        assert decoded == original, "Twist velocities changed"
    else:
        assert isinstance(original, PointCloud2) and isinstance(decoded, PointCloud2)
        np.testing.assert_array_equal(decoded.as_numpy()[0], original.as_numpy()[0])
        np.testing.assert_array_equal(decoded.intensities_f32(), original.intensities_f32())
        assert (decoded.frame_id, decoded.ts, decoded.seq) == (
            original.frame_id,
            original.ts,
            original.seq,
        ), "PointCloud2 metadata changed"


def check_rejections(raw: bytes, type_name: str, mode: Mode) -> None:
    """Assert both formats reject truncated payloads and a wrong declared type."""
    wire = pack(raw, type_name, mode)
    header, _ = split_frame(wire, mode)
    if mode == "base64":
        request = json.loads(header)
        request["params"]["args"][0]["data"] = base64.b64encode(raw[:-1]).decode("ascii")
        truncated = json.dumps(request).encode("utf-8")
    else:
        truncated = wire[:-1]
    wrong_type = PointCloud2.msg_name if type_name == Twist.msg_name else Twist.msg_name
    for payload, expected_type, error in (
        (truncated, type_name, "LCM payload length"),
        (pack(raw, wrong_type, mode), type_name, "Expected"),
    ):
        try:
            unpack(payload, expected_type, mode)
        except ValueError as exc:
            assert error in str(exc), f"Unexpected rejection: {exc}"
        else:
            raise AssertionError(f"Failed to reject malformed {mode} request")


def median_us(operation: Callable[[], Any], iterations: int) -> float:
    operation()  # Warm imports and caches before sampling.
    samples = []
    for _ in range(iterations):
        start = perf_counter_ns()
        operation()
        samples.append((perf_counter_ns() - start) / 1_000)
    return median(samples)


def make_messages(point_count: int) -> tuple[Twist, PointCloud2]:
    rng = np.random.default_rng(SEED)
    points = rng.uniform(-10, 10, (point_count, 3)).astype(np.float32)
    intensities = (np.arange(point_count) % 256).astype(np.float32)
    cloud = PointCloud2.from_numpy(
        points, frame_id="rpc-demo", timestamp=1_700_000_000.25, intensities=intensities
    )
    cloud.seq = 7
    return Twist(linear=(1.0, -2.0, 0.5), angular=(0.0, 0.25, -0.5)), cloud


def run(point_count: int, iterations: int, output_dir: Path | None) -> dict[str, Any]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dimos_lcm": importlib.metadata.version("dimos-lcm"),
        "point_count": point_count,
        "seed": SEED,
        "iterations": iterations,
        "timing_scope": "Median local pack/unpack; excludes LCM conversion and network",
        "results": [],
    }
    print(f"Local framing experiment: {point_count:,} points, {iterations} samples, seed={SEED}")
    print("Times are framing only (including JSON validation); not RPC latency.")
    print(
        f"{'message':<13} {'mode':<7} {'LCM bytes':>10} {'wire bytes':>11} "
        f"{'overhead':>9} {'pack us':>10} {'unpack us':>10}"
    )
    for message in make_messages(point_count):
        raw = message.lcm_encode()
        type_name = message.msg_name
        short_name = type_name.rsplit(".", 1)[-1]
        stem = short_name.lower()
        verify_message(message, type(message).lcm_decode(raw))
        if output_dir is not None:
            (output_dir / f"{stem}.lcm").write_bytes(raw)
        for mode in MODES:
            wire = pack(raw, type_name, mode)
            recovered = unpack(wire, type_name, mode)
            assert recovered == raw, "Framing changed the LCM payload"
            verify_message(message, type(message).lcm_decode(recovered))
            check_rejections(raw, type_name, mode)
            overhead = 100 * (len(wire) / len(raw) - 1)
            pack_us = median_us(partial(pack, raw, type_name, mode), iterations)
            unpack_us = median_us(partial(unpack, wire, type_name, mode), iterations)
            print(
                f"{short_name:<13} {mode:<7} {len(raw):>10,} {len(wire):>11,} "
                f"{overhead:>8.2f}% {pack_us:>10.1f} {unpack_us:>10.1f}"
            )
            report["results"].append(
                {
                    "message": type_name,
                    "mode": mode,
                    "lcm_bytes": len(raw),
                    "wire_bytes": len(wire),
                    "overhead_percent": overhead,
                    "pack_us": pack_us,
                    "unpack_us": unpack_us,
                    "verified": True,
                }
            )
            if output_dir is not None:
                suffix = "json" if mode == "base64" else "bin"
                (output_dir / f"{stem}.{mode}.{suffix}").write_bytes(wire)
                if mode == "binary":
                    header, _ = split_frame(wire, mode)
                    (output_dir / f"{stem}.binary.header.json").write_bytes(header)
    print("PASS: values, metadata and LCM bytes preserved; truncation and wrong types rejected.")
    if output_dir is not None:
        (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Artifacts: {output_dir}")
    return report


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=positive_int, default=DEFAULT_POINTS)
    parser.add_argument("--iterations", type=positive_int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--output-dir", type=Path, help="Optionally save wire samples and results.json"
    )
    args = parser.parse_args()
    run(args.points, args.iterations, args.output_dir)


if __name__ == "__main__":
    main()
