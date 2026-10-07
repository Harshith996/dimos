# Convert an old recording to CDR

This offline tool is for reviewing historical DimOS recordings on the CDR proposal.
It does not start a blueprint, router, robot, viewer or network connection.

From this proposal checkout, use an environment containing DimOS, its matching
`dimos-generated` package and the retained historical decoder:

```sh
pip install dimos-lcm==0.1.4
dimos mem convert old.mcap converted.mcap
```

Open `converted.mcap` directly in Foxglove with **Open local file**. Select your
image topic in an Image panel and point cloud / pose topics in a 3D panel. A local
Foxglove application or an existing signed-in web session may be needed; the
converter does not create an account or upload your recording.

For an old memory SQLite recording, or for CDR SQLite output:

```sh
dimos mem convert old.db converted.mcap
dimos mem convert old.mcap converted.db
dimos mem convert old.db converted.db
```

Choose a **new output filename**. The original file is opened read-only. Existing
outputs and reports are refused, including a destination created concurrently.
Conversion writes to a temporary sibling directory; no output is published if
any stream or payload fails. Output directories must already exist.

Inspect either output through the current CDR memory reader:

```sh
dimos mem summary converted.mcap
dimos mem summary converted.db
```

The original `dimos.msgs` legacy classes and their `lcm_encode`/`lcm_decode`
interfaces are temporarily retained with the existing `dimos-lcm` dependency for
migration. This reuses the original implementation; it introduces no new codec.
New runtime transports and recordings still use generated CDR types; there is
no automatic old-wire fallback.
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

With the historical decoder installed:

```sh
python -m pytest dimos/memory/test_convert_recording.py dimos/protocol/test_cdr_mcap.py
```

Tests cover both containers, LZ4/JSON/JPEG/LCM, independent rosbags decoding,
Header/type mapping, timestamp/sequence retention, unsupported types, corrupt
payloads, empty streams and exclusive output publication. Legacy-specific tests
are skipped if the historical decoder is missing; that is not a full
converter acceptance run.

## Convert a directory explicitly

```sh
dimos mem convert ./recordings ./recordings-cdr --dry-run
dimos mem convert ./recordings ./recordings-cdr
dimos mem convert ./recordings ./recordings-cdr-sqlite --format db
```

The destination must be new, with an existing parent, and must not overlap the
source tree. Relative directories are preserved: `trip/run.db` becomes
`trip/run.db.cdr.mcap`. No home-directory scan or automatic download occurs.
Dry-run reads schema declarations and counts rows; it does not prove payload
integrity. Both dry-run and conversion exit nonzero when candidates are blocked.
Archives, LFS pointers, pickle, raw captures and symlinks are reported as blocked;
unrelated non-recording files are ignored. Extract/materialize trusted archives
separately before selecting an expanded source directory.

All candidates pass metadata preflight before any conversion starts. Each output
and its audit file are published exclusively. If a later payload fails, earlier
successful files remain, later files are marked not attempted, and the command
fails. `migration-summary.json` records every candidate and result. Retain the
originals; a successful conversion is not authorization to delete them.

## Ivan's offline acceptance checklist

Run from this proposal checkout with its prepared development environment activated
(`source .venv/bin/activate`). It must contain matching `dimos-generated`, MCAP,
NumPy, Open3D and Rerun packages. These commands do not build/install dependencies
or start a robot. Use a fresh directory for each run:

```sh
export CDR_REVIEW_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dimos-cdr-review.XXXXXX")"
```

### Real point-cloud fixture

Use `unitree_go2_detection_cdr`, five real Go2 captures with cloud, image and
odometry payloads. Its approximately 5.8 MB project LFS archive is available in
this proposal's data inventory. Unlike `alfred_fusion_short`, it contains point
clouds. The manifest traces these captures to source LFS SHA-256
`51a817f2b5664c9e2f2856293db242e030f0edce276e21da0edc2821d947aad2`.
Set `CDR_FIXTURE_DIR` to an already extracted copy to avoid downloading anything.
If unset, `get_data` uses the checkout's data cache and downloads/extracts that
named project fixture if missing; it does not fetch the full 1.21 GB source.

The following packages the existing CDR bytes, checking every payload hash. It
is **not a legacy LCM conversion test** and never opens the original pickle files.

```sh
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from dimos_generated.geometry_msgs.msg import PoseStamped
from dimos_generated.sensor_msgs.msg import Image, PointCloud2
from dimos.protocol.cdr_mcap import CdrMcapWriter
from dimos.utils.data import get_data

source = Path(os.environ["CDR_FIXTURE_DIR"]) if "CDR_FIXTURE_DIR" in os.environ else get_data("unitree_go2_detection_cdr")
manifest = json.loads((source / "manifest.json").read_text())
classes = {cls.msg_name: cls for cls in (PointCloud2, Image, PoseStamped)}
output = Path(os.environ["CDR_REVIEW_DIR"]) / "source.mcap"
assert not output.exists(), output
with CdrMcapWriter(output) as writer:
    for sequence, moment in enumerate(manifest["moments"]):
        for topic, entry in moment["streams"].items():
            payload = (source / entry["file"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["cdr_sha256"]
            cls = classes[entry["msg_name"]]
            cls.decode(payload)
            writer.write(topic, payload, schema_name=cls.msg_name, schema=cls.schema,
                         log_time_ns=round(entry["recorded_ts"] * 1e9),
                         publish_time_ns=entry["source_sec"] * 10**9 + entry["source_nanosec"],
                         sequence=sequence)
print(output)
PY

dimos mem convert "$CDR_REVIEW_DIR/source.mcap" "$CDR_REVIEW_DIR/converted.mcap"
dimos mem summary "$CDR_REVIEW_DIR/converted.mcap"
```

Expect 15 messages: five each on `lidar`, `video` and `odom`. For **legacy**
acceptance, separately supply a supported old `.db` or `.mcap` to `dimos mem
convert`, keeping its original and conversion audit report. The automated
converter tests below exercise old LCM encodings; packaging this CDR fixture does
not replace them.

### Memory rendering, mapping and replay

```sh
dimos mem rerun "$CDR_REVIEW_DIR/converted.mcap" --no-gui --out "$CDR_REVIEW_DIR/memory.rrd"
dimos map global "$CDR_REVIEW_DIR/converted.mcap" --device CPU:0 --block-count 10000 --no-gui --out "$CDR_REVIEW_DIR/global.rrd"
dimos map replay "$CDR_REVIEW_DIR/converted.mcap" --map-final --map-device CPU:0 --no-gui --out "$CDR_REVIEW_DIR/replay.rrd"
```

There is no `dimos mem replay` command. `mem rerun` renders recorded messages;
`map replay` writes cloud/image/trajectory visualization. Neither is live
blueprint transport replay. Expect global mapping to retain five clouds, and map
replay to process five clouds, images and odometry samples. These world-frame
clouds require no stored `obs.pose`. Spatial dedup defaults off (`--pgo-tol 0`);
explicit dedup or PGO requires trajectory metadata. Separate recorded odometry
is not automatically attached as `obs.pose`. Sensor-frame clouds need recorded
TF registration. Missing required data is an error, not an invented pose.

Stream roles are selected automatically only when there is one compatible
candidate. For a recording with multiple clouds/images, select explicitly:

```sh
dimos map replay "$CDR_REVIEW_DIR/converted.mcap" --lidar lidar --image video --no-gui --out "$CDR_REVIEW_DIR/selected.rrd"
```

`map global --markers` and `map replay-marker` additionally need usable camera
calibration and poses; this fixture/checklist does not claim marker acceptance.
`map rename` and `map pose-fill` remain SQLite-only.

### Independent validation and manual viewers

If the separately installed official Foxglove MCAP CLI is on PATH:

```sh
mcap doctor "$CDR_REVIEW_DIR/converted.mcap"
```

The Python `mcap` package is not that executable. No installation is performed by
these commands. Doctor has not been run for this acceptance; CRC/payload checks
are not a substitute for it.

Open the generated files locally (manual visual acceptance):

```sh
rerun "$CDR_REVIEW_DIR/global.rrd"
rerun "$CDR_REVIEW_DIR/replay.rrd"
```

In Foxglove, **Open local file** `converted.mcap`; select `lidar` in a 3D panel
with fixed frame `world`, `video` in an Image panel, and inspect `odom` in Raw
Messages. The image header has no frame, and odometry uses `odom`; do not assume
a camera/odometry-to-world TF exists or that all overlays align. Do not upload
private recordings to obtain acceptance. Existing application/session access
may be required. Successful RRD export does not constitute human visual review.

### Focused regression entry

```sh
python -m pytest dimos/memory/test_convert_recording.py dimos/protocol/test_cdr_mcap.py dimos/memory/store/test_mcap.py dimos/mapping/cli/test_stream_selection.py dimos/mapping/cli/test_pgo_accumulate.py dimos/mapping/loop_closure/test_pgo.py
```

At `656efff71958e9c9c981d9e08ec95d1bebc4049d`, 40 focused mapping/reader/PGO tests
passed separately from earlier converter tests. The local real-data audit also
preserved all 15 payloads, schemas, timestamps and sequence numbers exactly,
kept all five clouds in CPU global mapping and exported five clouds/images/odom
in replay. This is bounded real sensor-payload acceptance using a repackaged CDR
fixture, not a full legacy archive migration, live robot test, full-suite pass,
or completed manual viewer/doctor gate.
