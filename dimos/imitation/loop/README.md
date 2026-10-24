# OpenYAM collection and ACT loop

An experimental single-arm learning application operated in **DimOS Desktop**:
Quest demonstrations → saved episodes → ACT → rollout → human corrections → fine-tuning.
The Desktop app lives in its own repository. This directory owns the experiment session,
recording, model handoff and app worker; it contains no separate web interface.

## Ownership

```mermaid
flowchart LR
  Desktop[DimOS Desktop] --> Session[Session: authority + episode acceptance]
  Quest[Existing WebXR module] --> IK[Existing teleop IK task]
  Session --> Trajectory[Existing joint trajectory task]
  ACT[Isolated ACT inference] --> Session
  IK --> Coordinator[ControlCoordinator]
  Trajectory --> Coordinator
  Coordinator --> Device[YamPositionDevice: sole writable client]
  Device --> Sim[Existing sim2 + exported YAM artifact]
  Sim -->|joint feedback + acknowledged commands| Capture[Typed MCAP + provenance]
  Sim -->|wrist RGB| Capture
  Sim -->|wrist RGB + joint positions| Session
  Session --> ACT
  Capture --> Selection[Saved human segments, train split, alignment checks]
  Selection --> Dataset[Immutable standard LeRobot dataset]
  Dataset --> Train[Official LeRobot ACT training]
  Train -->|weights + processors + contract + ancestry| ACT
```

The exported YAM artifact is the physical robot definition. `openyam/workcell.py`
adds the task scene and cameras. It preserves native position servos and the passive
finger reset contract; it does not import the RL trainer or add a second PD loop.
The existing coordinator owns interpolation, joint velocity limits and task arbitration.
`YamPositionDevice` fences in-flight writes when authority changes and journals accepted
commands until their exact sequence is acknowledged by sim2.

The session changes authority and submits targets. It never writes a simulator client.
A takeover invalidates queued and in-flight policy predictions, including after a later
resume. ACT steps that elapsed during inference are skipped. At the end of a prediction
horizon the controller holds measured position while requesting a new observation-based
chunk. Inference taking over 500 ms, stale observations, reset or an operator Hold require
an explicit policy restart. Fresh Quest input can maintain the operator lease independently
of the Desktop page. This worker configures a unique loopback-only simulation namespace.

## Operate in Desktop

Prepare a normal DimOS worktree and its isolated trainer:

```sh
uv sync --extra all
uv sync --frozen --project native/python/lerobot
```

In the separate OpenYAM Workcell app's **Setup**, choose this checkout and the directory
containing the exported `contract.json`, `robot.xml` and assets. The artifact loader checks
its hashes and physical contract. No runtime download or training checkout is required.
Changing Setup requires an idle experiment. Each app installation has its own data folder.

1. Start the simulation, held. Set a task instruction before collecting.
2. Start Quest teleop on an available port and open the displayed HTTPS address on the
   headset. Hold the right grip to move through the existing IK task; the trigger closes
   the gripper. Release holds. A starts/saves a **training** episode; B discards it. Wrist
   video and recording state appear in the headset. Desktop also offers joint controls.
3. Save useful episodes; discard the rest. Use Desktop's eval split for held-out collection.
4. Train ACT. The simulation stops while the training subprocess owns the job. Normal
   training uses ResNet18, a 512-dimensional transformer and 50-action chunks. A short
   run checks integration; the default 2,000 steps is a budget, not a success guarantee.
5. Load a completed checkpoint. Record a rollout, Run policy, and take over with the grip
   or joint controls. Saving accepts the human corrections. Fine-tune from that checkpoint
   using the expanded dataset. Fine-tuning starts a fresh optimizer and retains the
   parent's normalization processors and file identity.

Opening the app alone does not start simulation. Closing a session discards an unfinished
recording. No physical arm connection is created by this application.

## Data and compatibility

Policy input is 320×180 wrist RGB plus seven joint positions at a nominal 20 Hz. It uses
the exported artifact's named `camera_d405`, preserving its mount, optical calibration
and 16:9 aspect ratio. This is model calibration; real camera extrinsics and hardware
gripper calibration still need measurement. Overview RGB and
scene geometry are operator views and never enter ACT. Arm actions are absolute public
joint positions in radians. In this artifact the public gripper coordinate increases
closure: 0..1 maps to left-finger displacement 0..0.0355 m. Older prototype coordinates
and checkpoints require an explicit migration and are rejected.

Each raw episode contains native typed `JointState`/`Image` MCAP channels, the shared
`CollectionProfile`/`RecordingSchema`, authority and reset generations, exact acknowledged
command sequences, dispatch timestamps, task/split/acceptance, and policy identity.
Sensor timestamps follow sim2's clock. Acknowledged action samples use that same clock;
original host dispatch time and command age remain separate provenance. They must not be
subtracted as if the two clocks were identical.

Training selects finalized, saved **human/correction** segments in the train split.
Policy actions, holds, unacknowledged commands, discarded/partial recordings and evaluation
are excluded. Takeover, reset and camera gaps split episodes. Alignment checks reject bad
segments with reasons in the snapshot manifest. No interpolation across these boundaries
or silent quality relaxation occurs.

The frozen manifest hashes its MCAP/schema inputs. The isolated native LeRobot writer
creates a standard dataset with each accepted segment as an episode; official LeRobot
padding prevents ACT chunks crossing segment boundaries. The completed dataset has a file
manifest. Checkpoints include weights, processors, configuration, losses, dataset identity,
parent ancestry and a runtime file manifest. Loading/fine-tuning checks the artifact,
camera profile, joint order, action semantics, FPS, image shape and file hashes.

The former hand-written LeRobot parquet/video writer is replaced by the native dataset
API. Generic `run_dataprep` delegates LeRobot conversions to the locked environment; HDF5
and explicit-format CLI inspection remain supported. The earlier loop's NPZ handoff is gone.

Offline entry points, using a new output directory each time:

```sh
python -m dimos.imitation.loop snapshot /path/to/learning /path/to/snapshot
python -m dimos.imitation.loop train /path/to/snapshot /path/to/checkpoint --steps 2000 --device cpu
```

## Verification and extension

Behavior tests cover authority fencing across threads, acknowledgements/reset, stale
inference, delayed action steps, dataset selection and immutable inputs. Native tests
exercise official LeRobot encoding/loading and action padding at correction boundaries.
See `verification.json` for the recorded local integration run and its limits.

```sh
python -m pytest dimos/imitation/loop dimos/imitation/dataprep dimos/imitation/collection/test_profile.py dimos/imitation/collection/test_recording.py dimos/robot/manipulators/openyam/test_device.py dimos/robot/manipulators/openyam/test_simulation.py
uv run --frozen --project native/python/lerobot --group tests --with-editable . python -m pytest -c native/python/lerobot/pyproject.toml native/python/lerobot/dimos_lerobot/act_loop_tests.py native/python/lerobot/dimos_lerobot/dataprep_tests.py
```

This reuses the collection profile/schema and native export design from
`cc/feat/flexible-policy-module` at `fde4d8e047729de20cb2e4162e2b60698b5738bc`, the existing
coordinator/WebXR/IK stack, and the reusable sim2 YAM integration. It does not merge the
competing station recorders or add a second hardware control path. Future physical use
needs a verified hardware adapter, sensor/action calibration and measured teleoperation
and policy evaluations. RL, SONIC, bimanual stations, cloud training, semantic labeling and
policy chaining remain separate extensions.
