# R1 Pro target and evaluation reference

The current baseline tests fixed-base, fixed-torso arm tracking in MuJoCo. Its FK references are a regression distribution. They do not represent human teleoperation, prove dynamic feasibility, or validate hardware safety. Human recordings are unavailable. SEW is not a prerequisite.

## Current target contract

`TargetFrame` in `dimos/robot/galaxea/r1pro/rl/targets.py` contains two wrist positions in meters and two unit quaternions in wxyz order. The order is left wrist, then right wrist. Coordinates are world-fixed. The current model uses vendor gripper origins as wrist sites. Those sites are uncalibrated TCP approximations.

Each frame has a source timestamp in the simulation clock domain, a validity flag, and separate per-wrist position and orientation masks. Missing orientation requires `orientation_valid=False`. A normalized placeholder quaternion can occupy a masked field, but the actor and reward ignore that field. Optional elbow positions and a chest quaternion express soft preferences. The initial environment does not consume these preferences.

`provenance` identifies the source, and `calibration_id` can identify an external calibration. These fields do not enter the actor observation. A user-owned teleoperation adapter implements `TargetSource.sample(now_s)` and returns a current `TargetFrame` or `None`. Retargeting, time synchronization, coordinate calibration, confidence-to-mask conversion, and validation remain the adapter's responsibility. There is no human-motion file loader or licensed dataset in this baseline.

`CurrentTargetBuffer` rejects out-of-order timestamps. Its `current` method returns no frame for dropout, stale input, invalid frames, or future timestamps. Reset clears the previous episode's target. The environment independently checks target age and terminates on dropout or timeout. Reward does not implement the watchdog.

The deployable interface contains no future frames, trajectory phase, clip ID, or centered finite differences. The offline generator can store a complete trajectory, but its `sample` method returns one current frame. Body-relative references require an explicit coordinate transformation before this interface.

## Procedural trajectories

`generate_joint_trajectory` uses bounded quintic interpolation between joint knots. The default amplitude is at most 0.08 radians. The default joint margin is 0.02 radians, and the generator bounds analytic joint speed through the quintic blend's maximum slope. A caller supplies model-specific validity checks. The generator checks 401 evenly spaced samples by default and rejects a candidate if any sample fails.

Sampled collision checks can miss contacts between samples. Interpolation and FK do not establish torque, acceleration, balance, contact, or hardware feasibility. The MuJoCo adapter checks the audited mesh model with its documented penetration tolerance. Convex collision approximations remain a model limitation.

Training has an excursion-and-return scenario. Held-out scenarios add a pause or a reversal, and use a separate random-seed domain. `HELDOUT_MANIFEST` freezes four cases before policy inspection: pause, reversal, pause with dropout, and reversal with stale input. Each case has a fixed seed. The short CLI scales fault intervals to its requested duration and records the resulting exact case in its output manifest.

The baseline does not yet include the proposed reach, orientation-only, carry, asymmetric-arm, chest-motion, mobile-base, or unreachable-target families. A manifest is frozen for each run; this small suite is not a teleoperation generalization benchmark.

## Independent metrics

`EpisodeTrace` and `evaluate_episode` in `evaluation.py` do not import reward code. They require strictly increasing timestamps and finite values. Physical derivatives use actual sample intervals, including a short final physics substep after a safety termination. Per-wrist masks exclude missing target components. Missing or entirely stale references yield unavailable tracking metrics rather than zero error.

The evaluator reports unaligned wrist position RMSE and position p95. It reports shortest SO3 orientation RMSE, and each wrist's position and orientation p50, p95, and maximum. Quaternion sign changes do not affect orientation error. It also reports target dropout, timeout, future-timestamp, collision, and joint-limit sample counts.

Lag is a separate bounded alignment diagnostic. A positive lag means actual motion follows an older target. Every candidate uses the same interior samples, and the entire alignment window must have fresh references for both wrists. Lag never replaces unaligned tracking error. Short runs with insufficient interior samples report lag as unavailable. Irregularly sampled traces also report lag as unavailable.

Wrist speed, acceleration, and jerk RMS use physical units and all actual motion samples, including fault intervals. Offline finite differences belong only to evaluation. The actor does not receive these derivatives. Collision sample counts are not contact-event counts.

The executable runner records requested duration, observed duration, completion fraction, termination reason, raw wrist references, raw actions, and limited joint commands. It never autoresets or deletes early-failure frames. A failed short episode remains incomplete even when its pre-failure pose error is low. Runs with fewer than four samples preserve their raw trace and report failure without fabricated metrics.

Overshoot, settling time, a production IK comparison, calibrated hardware latency, and a deliberately unreachable-target recovery suite remain unsupported. The hold controller is a zero-rate debugging control, not a production IK baseline. Any future PPO versus IK comparison must use identical targets, preprocessing, control rates, limits, and failure accounting.

## Executable evaluator

The module entry point is `python -m dimos.robot.galaxea.r1pro.rl.evaluate`. It accepts `--output` for a new output directory, `--source-root` for the pinned local model assets, and `--duration` in 0.02-second multiples from 0.08 to 2 seconds. `--case` selects one of the four held-out cases by zero-based index. Without `--checkpoint`, it uses the zero-rate hold control. With `--checkpoint`, it loads an existing RSL-RL checkpoint and runs inference without policy updates.

`run_case(env, source, action_fn, requested_duration_s=...)` supports another policy through a callable. The runner installs the current reference before obtaining the actor observation. It uses separate MuJoCo state for FK so target generation cannot mutate the evaluated robot.

Each run writes `manifest.json` and `evaluation.json`. The manifest contains the exact scenario, seed, fault intervals, controller, environment configuration, and hashes for the manifest, model XML, asset manifest, configuration, and optional checkpoint. The evaluation file contains the raw trace and independent metrics. No success thresholds are applied. Proposed 3 cm or 10 degree thresholds remain drafts and do not imply p95 guarantees or hardware clearance.

## Integration gate

Larger training requires an approved control contract, actual hardware and SDK identification, calibrated wrist frames, credible collision geometry, model-dynamics validation, and an agreed frozen evaluation distribution. The current 14-arm-joint position-rate adapter is a simulation debugging contract. Torso release and chassis velocity control need separate adapters and validation. No part of this evaluation sends robot commands.
