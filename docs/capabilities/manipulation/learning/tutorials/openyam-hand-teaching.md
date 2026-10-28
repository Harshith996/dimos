# Collect demonstrations and train an OpenYAM policy

Use one OpenYAM arm and one wrist RGB camera to record a demonstration, prepare a
LeRobot dataset, and start ACT training. Quest is not required.

This tutorial ends with training outputs. Running a checkpoint is a separate
[rollout workflow](/docs/capabilities/manipulation/learning/guides/run-a-checkpoint.md) with compatibility checks.
The automated profile-to-dataset test uses synthetic Python recordings and real
LeRobot conversion. It does not validate native recording, hardware, or training
as a complete workflow.

## 1. Prepare the arm and camera

Start with a working DimOS installation, a configured OpenYAM CAN interface, and
a connected RGB camera. Follow the [OpenYAM integration guide](/docs/capabilities/manipulation/openyam_integration.md)
for hardware setup. The example uses CAN interface `follower_l` and camera
`/dev/video0`; replace them with your devices. Keep the same camera placement
throughout collection.

The teaching stack starts gravity compensation when the robot starts, including
between episodes. Its hand-guiding task sends zero-stiffness commands on every
control tick; the hardware adapter adds gravity torque and configured damping.
Support the arm before stopping the runtime.

## 2. Start the teaching stack

Choose a new recording directory and a task description for the demonstration:

```bash skip
dimos --can-port follower_l run openyam-teach-collection --daemon \
  --recorder.recording recordings/session-001 \
  --recorder.format mcap \
  --episodes.task "pick up the cube" \
  --wrist.hardware.camera-index /dev/video0

dimos imitation collect
```

The panel displays the task, recording state, elapsed time, and episode counts.
With the arm supported, confirm that you can guide it by hand. The profile uses
measured seven-dimensional joint positions as both state and teaching action.

## 3. Record a demonstration

1. Reset the scene for the task.
2. Press **Space** to start an episode.
3. Guide the arm through the demonstration.
4. Press **Space** again to save the episode. The panel returns to ready and the
   saved count increases.

To discard an active take, press **D** instead. Repeat these steps to collect
additional demonstrations. Saving an episode marks a dataset boundary; it does
not stop gravity compensation or assess whether the task succeeded.

## 4. Finish collection

Save or discard the final take, then press **Q** to detach. During an active
episode, Q asks for confirmation; press Q again to detach.

Recording and the robot continue after the panel exits. You can reattach with
`dimos imitation collect`. Support the arm, then shut down the stack:

```bash skip
dimos stop
```

The recording directory contains `schema.json` and `recording.mcap`.

## 5. Prepare and view the dataset

Inspect the recording, export saved episodes, and inspect the result:

```bash skip
dimos imitation inspect recordings/session-001
dimos imitation prepare recordings/session-001 --output datasets/session-001
dimos imitation inspect datasets/session-001
```

Preparation prints `Wrote dataset:` followed by the output path. The dataset
summary shows frame counts and features. Discarded and incomplete episodes are
not exported. If validation fails, use the
[data preparation guide](/docs/capabilities/manipulation/learning/guides/prepare-and-inspect-data.md).

On a machine with a graphical display, open the first exported episode:

```bash skip
dimos imitation visualize datasets/session-001 --episode 0
```

Use Rerun's playback controls to check the wrist image, joint state, and teaching
action on the same timeline. Confirm that the images show the demonstration and
that joint values correspond to its movement before collecting more data.

## 6. Start ACT training

On a machine configured to train ACT, run:

```bash skip
dimos imitation train \
  --dataset.repo_id=local/openyam-teach \
  --dataset.root=datasets/session-001 \
  --policy.type=act \
  --output_dir=outputs/openyam-act
```

The command streams LeRobot training output. Training writes its outputs under
`outputs/openyam-act`; a last checkpoint's pretrained-model directory is
`outputs/openyam-act/checkpoints/last/pretrained_model`. Training success does
not establish task success or compatibility with another robot.

See `dimos imitation train --help` for the installed LeRobot training options.
To use a compatible checkpoint, continue with
[Run a checkpoint](/docs/capabilities/manipulation/learning/guides/run-a-checkpoint.md).
