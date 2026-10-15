# Recorded wrist target streams

`RecordedTargetSource` reads local JSON files containing already-retargeted wrist poses. It does not read human skeleton formats, generate action labels, or establish that a recording is feasible. Human recordings are not available in this project. The checked-in `configs/r1pro_rl_target_stream_example.json` is explicitly a synthetic loader fixture.

The version 1 schema has three top-level fields: `schema_version`, `metadata`, and `frames`. The loader validates timestamps, pose dimensions, finite values, unit quaternions, confidence ranges, and masks before returning targets.

Metadata requires `reference_frame="world"`, `position_unit="m"`, `angle_unit="rad"`, `quaternion_order="wxyz"`, and `clock="relative_seconds"`. The `source`, `calibration_id`, and `session_id` strings must be nonempty. `kind` is either `user_owned` or `synthetic_fixture`. Additional metadata, including operator identity and calibration details, is preserved. A calibration identifier records provenance; it does not prove that calibration has been performed. Body-relative inputs require an explicit external coordinate transform before loading.

Each frame requires a nonnegative `stamp_s`, two three-dimensional `positions_m`, two Boolean `position_valid` masks, two Boolean `orientation_valid` masks, and two confidence values in `[0, 1]`. Timestamps must increase strictly. Wrists are ordered left, then right. `valid` optionally invalidates the entire frame. Confidence is preserved as recording metadata; the loader does not invent a confidence threshold or change the supplied masks.

`quaternions_wxyz` contains two unit quaternions or null entries. Missing orientation requires a false orientation mask. The internal storage placeholder is excluded from actor observations, reward, and tracking metrics. Optional `elbow_positions_m` and `chest_quaternion_wxyz` are preserved without enabling body rewards.

`sample(now_s)` returns only the latest frame whose timestamp is at or before `now_s`. It never interpolates with a future frame and never refreshes the original timestamp. Before the first frame it returns no target. After the last frame it holds the last original timestamp, allowing the environment watchdog to expire. There is no automatic replay loop or session concatenation.

The source retains the original file SHA-256, metadata, frames and confidence values. Session and operator identifiers stay outside actor observations. Future train/test partitioning must group complete sessions/operators and their augmentations together. Real recordings still need ownership, coordinate calibration, quality review, and a separately frozen evaluation split.

Tests use synthetic files to verify causal sampling, original timestamps, EOF timeout in the actual MuJoCo environment, orientation masking, provenance, optional preferences, and rejection of inconsistent units or malformed inputs. Passing those tests does not establish any human-data result.
