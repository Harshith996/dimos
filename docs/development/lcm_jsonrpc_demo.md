# LCM messages inside JSON-RPC: local framing experiment

This experiment compares two ways to carry existing LCM messages as a JSON-RPC
parameter. It reuses DimOS `Twist`, `PointCloud2`, their LCM codecs, and the
existing JSON-RPC request model. It does not change the RPC backend or contact
hardware. See [the demo script](/dimos/protocol/rpc/demo_lcm_jsonrpc.py).

## View arguments and return values

The [visual demo](/dimos/protocol/rpc/demo_lcm_jsonrpc_view.py) shows the actual
request and successful response for a local Twist echo in both formats:

```sh no-result
cd "$(git rev-parse --show-toplevel)"
.venv/bin/python -m dimos.protocol.rpc.demo_lcm_jsonrpc_view
```

The request puts the LCM descriptor in `params.args[0]`; the response puts it in
`result` and echoes the request ID. Format A carries base64 in `result.data`.
Format B carries the original return bytes after the JSON header in the same
frame. Both server and client decode their inputs, and the final tables verify
that argument and return values match. No network call is made.

Use `--linear 2 1 -1 --angular 0 0 -0.5` to change the values. Wide terminals
show the two formats side by side; narrower terminals stack them vertically.

## Reproduce

Use the checkout's existing development environment. If needed, set it up using
the repository's normal development installation instructions first. Run from
any directory inside the checkout:

```sh no-result
cd "$(git rev-parse --show-toplevel)"
.venv/bin/python -m dimos.protocol.rpc.demo_lcm_jsonrpc \
  --points 100000 --iterations 20 --output-dir /tmp/dimos-lcm-jsonrpc
```

The script creates a fixed `Twist` and a deterministic XYZ/intensity cloud with
seed 42, a fixed timestamp, frame ID and sequence number. For each format it:

1. Encodes the actual message using `lcm_encode()`.
2. Packages it in a request, validates that request, and extracts the LCM bytes.
3. Reconstructs it with `lcm_decode()` and checks the velocities or every XYZ and
   intensity value, plus the cloud's frame ID, timestamp and sequence number.
4. Checks that the extracted and re-encoded LCM bytes match the originals.
5. Verifies that a truncated LCM payload and a wrong declared type are rejected.
6. Reports wire size and median pack/unpack time, after warmup.

Success ends with `PASS`. An assertion or validation failure exits nonzero.
For a quick run, use `--points 3 --iterations 3`; for a larger cloud, use
`--points 1000000`. Point count and iterations must both be positive.

## The two formats

Both requests call `Demo/inspect` with exactly one positional argument, using the
existing `params.args` / `params.kwargs` convention. The descriptor includes
`type`, `encoding: "lcm"`, and the raw LCM `byte_length`.

| Format | Bytes on the wire | Consequence |
| --- | --- | --- |
| `base64` | A complete UTF-8 JSON-RPC request; the descriptor's `data` contains base64 text | Ordinary JSON, with base64 expansion and encoding/decoding work |
| `binary` | `DJR1` magic/version, uint32 big-endian JSON byte length, JSON-RPC header, then one raw LCM payload | Avoids base64; needs a custom framing contract |

The binary descriptor omits `data`. Its bytes follow the JSON header in the same
frame, so this experiment needs no second topic, external blob store or payload
lookup. `DJR1` is an experimental format identifier, not an existing DimOS or
JSON-RPC standard. The binary frame as a whole is not a JSON document.

The receiver knows the expected parameter type before decoding. It checks the
declared name and uses that known class's codec, without importing a class named
by the sender. This is deliberately limited to one message-valued parameter;
it does not introduce tags for arbitrary dictionaries or recursive containers.

## Inspect the saved bytes

The output directory contains `twist` and `pointcloud2` samples: original `.lcm`
bytes, `.base64.json` requests, `.binary.bin` frames, and `.binary.header.json`
files for readable inspection. `results.json` records sizes, timings, input
settings, Python/platform details and the installed `dimos-lcm` version.

```sh no-result
cd "$(git rev-parse --show-toplevel)"
.venv/bin/python -m json.tool /tmp/dimos-lcm-jsonrpc/twist.base64.json
.venv/bin/python -m json.tool /tmp/dimos-lcm-jsonrpc/twist.binary.header.json
```

## Interpret the result

`wire bytes` includes the whole request, including the binary prefix when used.
`overhead` is relative to the same raw LCM message. Base64 uses
`4 * ceil(LCM bytes / 3)` bytes before JSON metadata. The binary format adds its
eight-byte prefix and JSON metadata. A small Twist can therefore have large
percentage overhead even though the absolute difference is small.

Timings measure local packaging/unpackaging, including request validation and
Python byte copies. Message construction, LCM encoding/decoding and imports are
outside the timed region. These are microbenchmarks, not network throughput or
end-to-end RPC latency; repeat on representative hardware before drawing a
performance conclusion. Neither path is zero-copy.

This demonstrates preservation of the tested LCM representation. The cloud
fixture covers XYZ and intensity, not every PointCloud2 field or Python-only
attribute. It does not establish arbitrary Python object preservation, Rust/C++
interoperability, error handling, async execution, streaming or cancellation.
The separate viewer verifies successful Twist responses; the benchmark and saved
artifacts cover requests only.
The next experiment would send the same frames through Zenoh and decode a saved
fixture in Rust using the matching message definitions.
