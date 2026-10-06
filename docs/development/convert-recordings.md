# Convert an old recording to CDR

This offline tool is for reviewing historical DimOS recordings on the CDR proposal.
It does not start a blueprint, router, robot, viewer or network connection.

From this proposal checkout, use an environment containing DimOS, its matching
`dimos-generated` package and the optional historical decoder:

```sh
pip install dimos-lcm==0.1.4
python scripts/convert_recording.py old.mcap converted.mcap
```

Open `converted.mcap` directly in Foxglove with **Open local file**. Select your
image topic in an Image panel and point cloud / pose topics in a 3D panel. A local
Foxglove application or an existing signed-in web session may be needed; the
converter does not create an account or upload your recording.

For an old memory SQLite recording, or for CDR SQLite output:

```sh
python scripts/convert_recording.py old.db converted.mcap
python scripts/convert_recording.py old.mcap converted.db
python scripts/convert_recording.py old.db converted.db
```

Choose a **new output filename**. The original file is opened read-only. Existing
outputs and reports are refused, including a destination created concurrently.
Conversion writes to a temporary sibling directory; no output is published if
any stream or payload fails. Output directories must already exist.

Inspect either output through the current CDR memory reader:

```sh
python -m dimos.cli.dimos mem summary converted.mcap
python -m dimos.cli.dimos mem summary converted.db
```

The optional `dimos-lcm` decoder is used only by this offline tool. It is not added
to DimOS's normal runtime dependencies and no old-wire fallback is restored.
For JSON-only input it is unnecessary. `mcap`/`lz4` and SQLite support come from
the prepared DimOS environment. No dependencies are downloaded by conversion.

## Accepted input and mapping

| Input | Output |
| --- | --- |
| DimOS MCAP `lcm`, `jpeg`, `json`, optionally `lz4+…` | Generated standard ROS-compatible CDR values |
| Memory SQLite `_streams` registry, in-file observation/blob tables, same codecs | Same CDR values through the actual `SqliteStore` |
| Known CDR MCAP with `ros2msg` exactly matching the installed schema | Decode/validate/re-encode without changing the type |
| LCM Image with explicit `jpeg` codec | `sensor_msgs/msg/CompressedImage`, `format="jpeg"`, **original JPEG bytes** |
| Raw LCM Image / PointCloud2 / Imu / PoseStamped / Odometry / TFMessage and listed standard messages | Same ROS message shape; `std_msgs.Time.nsec` becomes `builtin_interfaces/Time.nanosec` |
| Explicit JSON `std_msgs.String` | Same UTF-8 JSON document inside generated `std_msgs/msg/String` |

The full, explicit allowlist is `TYPES` in
`dimos/memory/convert_recording.py`. Only standard messages with compatible field
semantics are admitted; unknown fields/types are errors. LCM length fields are
checked against their arrays, then represented by CDR sequence lengths. The
installed historical decoder checks the LCM fingerprint.

MCAP reads `dimos.payload_type` metadata. Historical
`dimos/<stream>/<Type>` wire names are also recognized when the type is uniquely
listed. It never imports a Python class or SQLite component named by an input
file. **Pickle is never loaded.** Unknown streams are listed together before an
output is created; corrupt later payloads also leave no published output.

Go2 DDS recordings use different conventions (`rt/utlidar/cloud`,
`rt/frontvideo`, proprietary Unitree types). A topic name alone does not specify
its bytes. Standard CDR channels need their matching full `ros2msg` definition;
undeclared raw video, proprietary DDS types and old Go2 pickle directories are
rejected rather than guessed. They need a separately reviewed, explicit export
from their trusted original producer.

## What is preserved

- Stream/topic names and per-stream message count/order, including empty streams.
  SQLite requires SQL identifier stream names; use MCAP for names containing `/`.
- All mapped message fields, including Header frame and exact sec/nanosec values.
- MCAP input's exact integer `log_time`, `publish_time` and envelope `sequence`.
  They are **not** replaced with conversion wall time or guessed Header time.
- SQLite's existing floating-point observation time. For MCAP output it supplies
  log time (rounded to nanoseconds); Header supplies publish time when present.
  Precision absent from the source float cannot be recovered.
- SQLite observation ID, value, pose and tags, and legacy ROS1 `Header.seq`, in a
  mandatory adjacent `converted.mcap.conversion.jsonl` / `.db.conversion.jsonl`
  audit file. MCAP container metadata is retained in the audit manifest. SQLite output additionally carries these in `cdr_conversion` tags
  and retains the spatial pose. ROS2 Header has no `seq`; for SQLite input its
  legacy value also supplies the MCAP sequence. For MCAP input the envelope
  sequence wins and the original Header sequence remains in the audit file.

MCAP uses `message_encoding="cdr"`, XCDR1 little endian,
`schema_encoding="ros2msg"`, full nested definitions, the `ros2` profile and ZSTD
chunks. It is not LCM bytes relabeled as CDR. SQLite output uses the current
`CdrCodec` and real store registry/blob format. Runtime import paths in output
refer to generated classes.

This migrates message recordings, not arbitrary Python object databases. External
blob stores, vector embeddings, MCAP attachments and pickle payloads are rejected instead of
silently discarded. Keep the audit file with the converted recording.

## Validation

With the optional historical decoder installed:

```sh
python -m pytest dimos/memory/test_convert_recording.py dimos/protocol/test_cdr_mcap.py
```

Tests cover both containers, LZ4/JSON/JPEG/LCM, independent rosbags decoding,
Header/type mapping, timestamp/sequence retention, unsupported types, corrupt
payloads, empty streams and exclusive output publication. Legacy-specific tests
are skipped if the optional decoder is not installed; that is not a full
converter acceptance run.
