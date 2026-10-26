# CBOR RPC proposal: JSON-RPC structure, LCM tags, and pickle fallback

## Proposed approach

Keep Zenoh as the transport and encode each RPC request or response as one CBOR message. Reuse the request, response, notification, and error conventions from JSON-RPC. Encode ordinary values directly, existing registered LCM messages with an LCM tag, and remaining supported Python objects with a pickle tag. Validate the message against CDDL before restoring tagged objects.

This combines `yimo/feat/json-rpc-draft`'s message structure with `yimo/feat/new-rpc-draft`'s binary payload and LCM reuse. CBOR replaces MessagePack, and pickle fallback replaces the earlier requirement to write an adapter for every Python type. The fallback keeps Python-specific calls possible; it does not make them portable to Rust or C++.

The scope is `RPCClient / RpcCall → RPCSpec → ZenohRPC`, including module lifecycle and management calls. Preserve the existing `@rpc`, `@skill`, and Spec call interfaces. Actor IPC, MCP, and typed streams are separate protocols.

**This branch contains a proposal, CDDL contracts, and an in-memory example. It does not change the running ZenohRPC implementation.**

## Message contract

This is a DimOS CBOR protocol inspired by JSON-RPC, not standard JSON-RPC 2.0: standard clients expect JSON. Use `rpc: "dimos-cbor/1"` to identify this protocol instead of advertising `jsonrpc: "2.0"`.

| Message | Required fields | Behavior |
| --- | --- | --- |
| Request | `rpc`, `id`, `method`, `params` | Execute a call and return one response |
| Notification | `rpc`, `method`, `params` | Execute without returning a response; omit `id` |
| Success | `rpc`, `id`, `result` | Echo the request ID; `result` is required even when null |
| Failure | `rpc`, `id`, `error` | Echo the ID, or use null when the request ID cannot be determined |

`params` always contains `args` (array) and `kwargs` (text-keyed map), including when both are empty. The server binds these to `f(*args, **kwargs)`. IDs are nonempty strings of at most 128 UTF-8 bytes; methods are nonempty strings of at most 1024 UTF-8 bytes. Generate unique IDs for pending calls, for example UUID hex strings. Requests with a null ID are invalid. Responses contain exactly one of `result` and `error`. Extra envelope fields and batch arrays are excluded from v1.

Use a versioned Zenoh key, `dimos/cbor-rpc/v1/<module-instance>/<method>`, with one Queryable per method. The body method is `<module-instance>/<method>` and must match the key. Zenoh still routes the call and carries the reply within the same query; there is no separate response topic. Use `application/cbor` for payload encoding, with the key and body identifying the DimOS profile.

## Value encoding

| Python value | Wire representation | Restoration |
| --- | --- | --- |
| Exact `None`, `bool`, `float`, `str`, `bytes`, and CBOR-range `int` | Native CBOR | Native value |
| Exact `list`, or exact `dict` with only string keys | Native array/map; recursively encode members | List/dict with restored members |
| Registered LCM type | Tag 60003 around `[type_id, lcm_bytes]` | Select its registered decoder |
| Other picklable objects | Tag 60004 around pickle bytes | Unpickle only when explicitly allowed for that trusted peer |
| Unsupported/unpicklable object | Encoding error | No request is sent |

The two tag numbers are provisional: they were unassigned in the [IANA registry](https://www.iana.org/assignments/cbor-tags) when checked on 2026-10-26. They are not a private-use allocation or registered DimOS tags. Resolve their allocation before interoperable deployment. Do not reuse 60000/60001; those numbers already identify Intel attestation data.

An LCM type ID is a stable protocol name such as `geometry_msgs.Vector3`, not an instruction to dynamically import a Python class. Maintain an explicit registry mapping IDs and supported Python types to codecs. Include the complete existing LCM encoding, including its fingerprint. Reject unknown IDs and decode failures. A failed registered LCM encoder must not silently fall back to pickle.

Use pickle protocol 5 for the v1 sender convention, with in-band buffers so the message remains self-contained. Pickle can still fail for some objects; peers also need compatible Python classes, dependencies, and import paths. Do not claim that every arbitrary object is transferable.

The native subset is deliberately explicit. A library's default conversion can turn tuples into arrays or enums into primitive values, losing Python types. In this draft, tuples, enums, dataclasses, large integers outside CBOR's untagged integer range, and dictionaries with non-string keys take the pickle path unless a dedicated adapter is added. Standard CBOR semantic tags are not implicitly enabled by this v1 schema.

The native containers use value semantics. Preserving cycles or shared object identity across separate tagged objects is outside v1; pickle only preserves references within its own blob. LCM reuse also requires checking the existing codec's round trip: Image and PointCloud2 may change timestamps, precision, or array writability. Objects such as GraspCandidate that internally use pickle must not be classified as portable LCM just because they expose an encoding method.

There is one application encoding entry point and one final CBOR payload. LCM/pickle objects still first produce their own bytes; CBOR then writes those bytes with a tag and length. This avoids JSON/Base64 and whole-message fallback, but is not a promise of one traversal, zero copies, or one underlying codec operation.

## CDDL files and validation

| File | Responsibility |
| --- | --- |
| [rpc.cddl](/docs/development/cbor-rpc/rpc.cddl) | Request, notification, success, failure, IDs, and Python argument envelope |
| [values.cddl](/docs/development/cbor-rpc/values.cddl) | Recursive native values and the allowed extensions |
| [extensions.cddl](/docs/development/cbor-rpc/extensions.cddl) | LCM and pickle tag content |

Concatenate these files in the listed order. The first rule, `rpc-message`, is the general validation root. These are fragments of one schema, not three independently complete schemas; no nonstandard CDDL include syntax is required. For a server or client, select the narrower request/notification or response root appropriate to that receive path. The example uses the general root for its fixed round trip.

CDDL checks field presence, value types, tag numbers, tag content shapes, and mutually exclusive response forms. It does not inspect LCM or pickle byte strings, resolve type IDs, validate LCM fingerprints, bind Python signatures, check request IDs against pending calls, or make pickle safe. Method-specific contracts and business constraints remain separate checks.

The receiver should:

1. Apply transport authentication and message size limits before parsing. Require exactly one CBOR item, definite-length containers, unique map keys, and bounded nesting.
2. Validate the wire message against CDDL without running LCM or pickle decoders. Retain the extension tags as data.
3. Check message direction, method/key consistency, and method authorization on the server; check pending request IDs on the client. Check whether this peer and method permit pickle.
4. Restore LCM values through the registry and pickle values only after the trust decision. Reject unsupported tags/types; do not silently return opaque values to a method.
5. Bind arguments, enforce method-specific constraints, and invoke the method. Encode the response using the same rules.

Do not put `pickle.loads()` in the initial CBOR tag hook: it would execute before whole-message validation and authorization. A successful CDDL check only confirms the pickle wrapper's shape. [Python's pickle documentation](https://docs.python.org/3/library/pickle.html) requires trusted input because unpickling can execute code. Disable the pickle path for peers that cannot satisfy that trust requirement; never retry through another protocol after rejection.

The example uses the Ruby `cddl` CLI for validation and `cbor2` to retain tags and decode the structure. It therefore starts a validator subprocess and parses the CBOR twice. This is a demonstration, not a proposed hot-path implementation; production needs an in-process validator/decoder integration. CDDL alone does not enforce every parser constraint above.

Validator selection must include negative cases. In local checks, `pycddl==0.6.4` incorrectly accepted unknown tags and invalid envelope alternatives for these schemas. The example uses `cddl==0.12.14`, which was checked against those cases, instead of relying on the Python package.

## Minimal example

Install the Ruby validator once with `gem install cddl --version 0.12.14 --no-document`, ensuring its `cddl` executable is on PATH. Run [demo_cbor_rpc.py](/dimos/spec/demo_cbor_rpc.py) from the repository root:

```sh
uv run --no-sync --with cbor2==6.1.5 python -m dimos.spec.demo_cbor_rpc
```

The client calls `calculate(Vector3(1, 2, 3), scale=2.0, label=Label("demo"))`. Vector3 uses LCM, scale uses native CBOR, and the small Label dataclass uses pickle. The server validates the request, restores the arguments, executes the function, and returns the scaled vector and label. The client validates the response, checks its ID, and verifies both restored types and the coordinates `(2.0, 4.0, 6.0)`.

This demonstration trusts its own locally generated request and reply and explicitly opts in to pickle. It uses one fixed method and one LCM type; it does not implement network authentication, a dispatcher, or a production codec registry. The pinned dependencies are only for running the example and do not alter project dependencies or the lockfile.

## Errors and rollout

Reuse the JSON-RPC error shape: integer `code`, string `message`, and optional `data`. Suggested codes are `-32700` for malformed CBOR, `-32600` for an invalid RPC envelope, `-32601` for an unavailable method reached by a handler, `-32602` for invalid parameters, `-32603` for framework failure, and `-32000` for a method exception. Return these in a normal Zenoh reply. Keep transport failures distinct. An unbound per-method key still times out because no handler exists to produce an error response.

For Python exception compatibility, put `type_name`, `type_module`, `args`, and the remote traceback in `error.data` and reuse the existing explicit exception reconstruction policy. A `TypeError` raised inside a method is an application exception, not automatically an argument-binding error. If result serialization fails, return a minimal native-only internal error when possible. Notifications log errors without returning a response.

1. Extract a shared codec from the demonstrated rules, with explicit LCM registrations, pickle policy, parser limits, and direction-specific CDDL validation.
2. Verify actual management payloads, including RPCClient, Transport, SkillInfo, tasks, and Blueprint, as well as exception and LCM round trips. Pickle fallback reduces adapter work but does not prove compatibility. Audit embedded pickle such as StreamDescriptor separately.
3. Integrate the codec into ZenohRPC requests, results, and errors while preserving synchronous/asynchronous call behavior and timeouts. Both endpoints select v1 before sending. Do not resend with legacy pickle after a timeout or decode failure, since the first call may have executed.
4. Verify module startup, wiring, calls, and shutdown before switching defaults. Native-language peers may implement the CBOR + LCM subset and explicitly reject pickle-tagged values.

References: [CBOR tags](https://www.rfc-editor.org/rfc/rfc8949.html#section-3.4), [CDDL](https://www.rfc-editor.org/rfc/rfc8610.html), [JSON-RPC conventions](https://www.jsonrpc.org/specification), and [the CDDL validator](https://rubygems.org/gems/cddl/versions/0.12.14).
