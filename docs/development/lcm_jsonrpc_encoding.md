# Encoding LCM messages in JSON-RPC

Status: design draft. Python framing and decoding have been verified locally.
Zenoh RPC integration and Rust/C++ interoperability have not been implemented or tested.

This document extends the [JSON-RPC proposal](/docs/json-rpc.md) with an encoding
contract for application messages. See the [experiment](/dimos/protocol/rpc/demo_lcm_jsonrpc.py),
the [visual demo](/dimos/protocol/rpc/demo_lcm_jsonrpc_view.py), and the
[reproduction guide](/docs/development/lcm_jsonrpc_demo.md).

## 1. Goal and recommendation

Allow existing DimOS messages such as `Twist` and `PointCloud2` to be passed as
JSON-RPC arguments and return values using the same data encoding convention.

**Reuse existing LCM message definitions and codecs, and add message descriptors
and framing at the RPC layer.** Ordinary JSON values follow the existing
proposal. Messages with an LCM definition use `lcm_encode()` / `lcm_decode()`
directly, without requiring a new JSON field mapping for each message type.

The experiment compares two candidates:

- **Format A: JSON + base64.** Encode the complete LCM byte sequence as base64
  inside a JSON parameter or result. This preserves a complete JSON document and provides
  a compatibility option and comparison baseline.
- **Format B: JSON header + LCM binary.** A custom frame joins a JSON header to
  the original LCM bytes. The current results support investigating this format
  for large messages over Zenoh next.

Format B's final protocol identifier, activation mechanism and compatibility
rules still need review. Local measurements alone do not establish a production default.

## 2. Responsibilities

| Layer | Responsibility | Current experiment |
| --- | --- | --- |
| JSON-RPC | Method names, request IDs, parameters and response structure | Reuses request and success-response models; the viewer simulates a local echo call |
| LCM | Application message definitions and binary codecs | Uses existing Twist and PointCloud2 codecs |
| Framing | Describe message types and combine JSON with LCM bytes | Implements experimental formats A and B |
| Zenoh | Route and transmit requests and responses | No network traffic in this experiment; integration is deferred |

LCM marshalling can be used independently of its communication facilities, so
Zenoh can carry LCM bytes. The [LCM type specification](https://lcm-proj.github.io/lcm/content/lcm-type-ref.html)
describes type definitions, code generation and type fingerprints.

JSON-RPC uses JSON's type system, which has no native binary value. Format A is
a JSON-RPC document whose LCM parameter descriptor is a DimOS convention.
Format B adds custom binary framing around that document.
See the [JSON-RPC 2.0 specification](https://www.jsonrpc.org/specification).

## 3. Message descriptor

The experiment follows DimOS's `params.args` / `params.kwargs` convention.
Requests call `Demo/inspect` with one positional argument and empty `kwargs`.
The argument or return value is a descriptor with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | String | Stable message name, such as `geometry_msgs.Twist` or `sensor_msgs.PointCloud2` |
| `encoding` | String, fixed to `lcm` | Codec for the original application message |
| `byte_length` | Nonnegative integer | Length of the entire LCM encoding, including its type fingerprint; not the base64 length or PointCloud2's internal data length |
| `data` | String | Standard base64 text in format A; omitted by the sender in format B |

`encoding` identifies the message codec, not the framing format. The script
selects A or B through an explicit `mode` argument. A production protocol must
also establish the format before sending.

The receiver knows the parameter's or return value's expected type and checks the
descriptor's `type` against it. The name selects a known codec rather than a
Python class to import dynamically. The experiment uses a message class already
known to the caller; generic binding from method signatures to codecs is not implemented.

Descriptors are interpreted only at positions explicitly defined as message
parameters or results. The experiment does not recursively scan arbitrary dictionaries or
treat every dictionary containing `type` or `data` as a message.

A successful response carries the descriptor under `result`, using the existing
`JsonRpcSuccessResponse` model. It echoes the request's `id` and has no `method`
or `params`. Format A stores the base64 text in `result.data`. Format B omits
`data` and appends the raw LCM return bytes after the response JSON header,
using the same `DJR1` framing as requests. The client validates the response ID,
expected return type and byte length before reconstructing the returned message.

The visual demo shows a complete local echo: the server decodes the argument,
returns that Twist unchanged, encodes a success response, and the client decodes
the result. It checks the values and LCM bytes in both directions. Ordinary JSON
results, including `null`, use the existing response model directly; the demo's
LCM response helpers handle only a single message-valued result.

The examples below use the same message: linear velocity `(1.0, 0.0, 0.0)` and
angular velocity `(0.0, 0.0, 0.5)`. Its LCM encoding is 56 bytes: an eight-byte
fingerprint followed by six eight-byte floating-point values. The displayed
content is generated by executable examples that also verify decoding.

## 4. Format A: LCM bytes inside JSON

The sender calls `lcm_encode()`, base64-encodes the resulting bytes, and puts the
ASCII text in `data`. It serializes the complete request as UTF-8 JSON.

The receiver validates the JSON-RPC structure and descriptor, performs strict
base64 decoding, checks the resulting byte length, then calls the expected
message type's `lcm_decode()`.

For an LCM message of `N` bytes, the base64 text occupies `4 * ceil(N / 3)` bytes,
plus JSON metadata. Large messages approach one-third size overhead; fixed
metadata dominates the percentage overhead for small messages.

The method and type names remain readable. The message fields inside `data`
require base64 and LCM decoding to inspect.

### Example A

This encodes the Twist as a complete JSON request and verifies the decoded value:

```python session=twist-example
import json

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.protocol.rpc.demo_lcm_jsonrpc import pack, split_frame, unpack

message = Twist(linear=(1.0, 0.0, 0.0), angular=(0.0, 0.0, 0.5))
lcm_bytes = message.lcm_encode()
wire_a = pack(lcm_bytes, message.msg_name, "base64")
assert Twist.lcm_decode(unpack(wire_a, message.msg_name, "base64")) == message
print(json.dumps(json.loads(wire_a), indent=2))
```

```results
{
  "jsonrpc": "2.0",
  "id": "lcm-demo:1",
  "method": "Demo/inspect",
  "params": {
    "args": [
      {
        "type": "geometry_msgs.Twist",
        "encoding": "lcm",
        "byte_length": 56,
        "data": "LnwH18334Cc/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP+AAAAAAAAA="
      }
    ],
    "kwargs": {}
  }
}
```

The string in `data` represents the original 56 bytes. The receiver recovers
those bytes and calls `Twist.lcm_decode()` to reconstruct both velocity vectors.
The compact request occupies 242 bytes; indentation and line breaks above are
added only for readability.

## 5. Format B: JSON and binary in one frame

The experimental format carries everything in one byte frame. It needs no
additional topic, object store or second lookup. A later experiment can send
the complete frame in one Zenoh request payload.

Let `H` be the JSON header's UTF-8 byte length and `N` the LCM byte length:

| Byte offset | Length | Content |
| --- | --- | --- |
| `0` | 4 bytes | ASCII `DJR1`, identifying the experimental framing version |
| `4` | 4 bytes | Unsigned 32-bit big-endian integer `H` |
| `8` | `H` bytes | Complete JSON-RPC request header, with no descriptor `data` field |
| `8 + H` | `N` bytes | Complete original LCM message, without base64 conversion |

The total length is `8 + H + N`. Header length counts UTF-8 **bytes**. The prefix
uses big-endian encoding; the LCM payload is exactly what the existing codec
produces. `DJR1` is an experimental identifier, not a published DimOS protocol.

The receiver checks the prefix and version, reads the header length, rejects a
truncated header, and parses the JSON-RPC request. All remaining bytes form the
single LCM payload and must match `byte_length`. Missing payload bytes or extra
trailing bytes are rejected.

The implementation supports one message argument or result and one binary block per frame, so it
does not need an offset table or attachment IDs. Multiple messages, nested
messages and mixed ordinary/binary values need later definitions.

The `DJR1` version describes framing; the LCM fingerprint checks agreement on
message structure. Fingerprints do not migrate fields between versions or
replace distribution and version management of message definitions.

### Example B

Encode the same Twist, omitting `data` from the JSON header:

```python session=twist-example
wire_b = pack(lcm_bytes, message.msg_name, "binary")
header, body = split_frame(wire_b, "binary")
assert body == lcm_bytes
assert Twist.lcm_decode(unpack(wire_b, message.msg_name, "binary")) == message
print(json.dumps(json.loads(header), indent=2))
print("Frame prefix (hex):", wire_b[:8].hex(" "))
print("Length:", len(wire_b), "= 8 +", len(header), "+", len(body))
```

```results
{
  "jsonrpc": "2.0",
  "id": "lcm-demo:1",
  "method": "Demo/inspect",
  "params": {
    "args": [
      {
        "type": "geometry_msgs.Twist",
        "encoding": "lcm",
        "byte_length": 56
      }
    ],
    "kwargs": {}
  }
}
Frame prefix (hex): 44 4a 52 31 00 00 00 9c
Length: 220 = 8 + 156 + 56
```

The complete frame concatenates these three parts:

| Part | Content in this example | Length |
| --- | --- | ---: |
| Prefix | `44 4a 52 31 00 00 00 9c`: `DJR1` followed by a JSON header length of 156 | 8 bytes |
| JSON header | The displayed JSON without indentation or line breaks | 156 bytes |
| LCM payload | The same bytes used before base64 encoding in example A | 56 bytes |

The total is 220 bytes. The receiver reads the eight-byte prefix, then 156 bytes
of JSON, and passes the remaining 56 bytes directly to `Twist.lcm_decode()`.

Format A puts a string representation of the LCM bytes inside JSON. Format B
appends the original bytes after JSON. The message type, LCM representation and
decoded values are identical.

## 6. Local verification and errors

Both formats use the same input messages and verify this sequence:

1. Construct a message and encode it using its existing LCM codec.
2. Package the bytes in the selected framing format.
3. Parse the request and check the method, argument count, type and payload length.
4. Confirm that unpacking preserves the original LCM bytes exactly.
5. Decode using the expected type and compare values and metadata.
6. Re-encode the reconstructed message and compare bytes again.

Twist checks cover linear and angular velocity. Point cloud checks cover every
XYZ and intensity value, frame ID, timestamp and sequence number. The experiment
also injects truncated payloads and incorrect declared types and verifies that
both formats reject them.

The demo reports validation failures as Python exceptions; JSON-RPC error
responses are not implemented. The parser also checks unknown framing versions,
damaged JSON headers and invalid base64, but injected negative cases currently
cover only truncation and type mismatch.

The visual demo also validates and decodes a successful response for each format,
matching its ID to the request. The benchmark below still measures requests only.

The fixed ID `lcm-demo:1` and method `Demo/inspect` simplify reproduction.
Production integration must follow the existing proposal's request ID generation
and response matching rules, and establish header and payload size limits.

## 7. Measurements

These results record one local run on 2026-10-16: Linux x86_64, Python 3.12.3,
`dimos-lcm 0.1.4`, with 20 samples after warmup and median timings. The cloud
contains 100,000 synthetic XYZ points generated with seed 42 and intensity
values; it is not a sensor recording.

| Message | Raw LCM bytes | Format A total bytes | Format B total bytes |
| --- | ---: | ---: | ---: |
| `Twist` | 56 | 242 | 220 |
| `PointCloud2`, 100,000 points | 1,600,127 | 2,133,679 | 1,600,300 |

| Message | Format | Median pack time (µs) | Median unpack time (µs) |
| --- | --- | ---: | ---: |
| `Twist` | A: JSON + base64 | 7.54 | 4.66 |
| `Twist` | B: JSON + binary | 6.52 | 4.31 |
| `PointCloud2` | A: JSON + base64 | 2,476.09 | 3,431.52 |
| `PointCloud2` | B: JSON + binary | 79.94 | 78.36 |

Both formats passed value, metadata, byte preservation and the two negative
checks. For this cloud, format A adds approximately 33.34% over raw LCM;
format B adds 173 bytes. For Twist, the formats differ by only 22 bytes.

Timings include framing, unpacking and JSON model validation. They exclude
message construction, LCM encoding/decoding, imports, method execution and network
transport. Both implementations copy Python bytes; neither is zero-copy. These
results support investigating binary framing, but do not predict the same
speedup for complete RPC calls. Hardware, library versions and load affect timing.

Preservation is established only for the tested LCM representation. The cloud
fixture covers XYZ/intensity and the listed metadata, not all PointCloud2 fields
or additional Python object state.

## 8. Reproduce and inspect requests

With the checkout's development environment installed, run inside the repository:

```sh no-result
cd "$(git rev-parse --show-toplevel)"
.venv/bin/python -m dimos.protocol.rpc.demo_lcm_jsonrpc \
  --points 100000 --iterations 20 --output-dir /tmp/dimos-lcm-jsonrpc
```

The final `PASS` confirms all built-in checks succeeded. The output directory
contains original `.lcm` bytes, both request formats, separate binary-format JSON
headers, and `results.json` with environment and measurement details.

Display the actual Twist request headers, including real base64-encoded LCM data:

```sh no-result
cd "$(git rev-parse --show-toplevel)"
.venv/bin/python -m json.tool /tmp/dimos-lcm-jsonrpc/twist.base64.json
.venv/bin/python -m json.tool /tmp/dimos-lcm-jsonrpc/twist.binary.header.json
```

Use `--points 3 --iterations 3` for a quick check. Generated files go only to the
specified directory. The demo starts no robot, host daemon or network service.

## 9. Implementation sequence

| Step | Work and completion criteria |
| --- | --- |
| Local experiment: complete | Both formats restore tested messages and provide reproducible size and framing measurements |
| Zenoh integration | Define protocol identifiers and explicit activation; implement real requests/responses, method routing, IDs and errors; measure complete calls |
| Cross-language verification | Decode Python fixtures in Rust and Rust fixtures in Python using matching message definitions; start with Twist, then point clouds |
| Parameter and result extensions | Define multiple messages, mixed arguments and nested containers; settle type binding, size limits and compatibility |

Choose the format before sending. After a request is sent, a timeout or response
decoding failure must not automatically retry with another format or pickle:
the remote method may already have executed.

Async functions, generators, async generators, progress and cancellation belong
to the call lifecycle and need a separate design. This proposal covers message
typing and encoding only.
