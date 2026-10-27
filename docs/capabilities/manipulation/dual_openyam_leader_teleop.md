# Dual OpenYAM Leader Teleop

Two YAM leader arms (the uynitsuj `yam_active_leader` design: seven Feetech
STS3215 servos per arm, IDs 1 to 6 for the joints and 7 for the gripper
trigger) drive the two OpenYAM followers. The leader module reads the Feetech
buses directly and streams follower `JointState` targets into the coordinator's
`joint_trajectory` task, plus a normalized opening per gripper. It reuses the
OpenArm Mini Feetech reader, calibration format and CLI; see
[OpenArm Mini Teleop](/docs/capabilities/manipulation/openarm_mini_teleop.md).

## Install

```bash
uv sync --extra manipulation --extra openarm-mini-teleop --inexact
```

## Motor IDs and calibration

Each arm is one serial bus. Write the IDs one motor at a time with
`dimos hardware feetech-leader setup-motor-id` (joint 1 to 6, then 7 for the
trigger). Calibrate each arm with the leader held in the follower's zero pose,
which for the YAM is the folded rest pose on its supports, the same pose the
Damiao motors are zeroed in. Calibration artifacts live under
`STATE_DIR/teleop/yam_leader/<side>/calibration.json`:

```bash
dimos hardware feetech-leader calibrate --side left \
  --port-left /dev/ttyACM0 --baudrate 1000000 \
  --left-calibration-path ~/.local/state/dimos/teleop/yam_leader/left \
  --left-flips none
```

Then watch the readout while moving one joint at a time and add every joint
that turns the wrong way to `--left-flips`, recalibrate, and repeat for the
right arm. Include `joint_7` in the flip list if the trigger reads negative
when squeezed. The trigger's full travel in radians is the
`gripper_travel_radians` module setting (default 1.0); read it off the TUI
with the trigger fully squeezed:

```bash
dimos hardware feetech-leader joint-tui --side left --port /dev/ttyACM0 \
  --baudrate 1000000 --calibration-path ~/.local/state/dimos/teleop/yam_leader/left
```

## Run

One machine, in-memory follower (no CAN ports), Viser on 127.0.0.1:8080:

```bash
dimos run teleop-dual-openyam-leader --port-left /dev/ttyACM0 --port-right /dev/ttyACM1
```

Hardware follower on the same machine:

```bash
dimos run teleop-dual-openyam-leader --port-left /dev/ttyACM0 --port-right /dev/ttyACM1 \
  --controlcoordinator.left-can-port follower_l --controlcoordinator.right-can-port follower_r
```

Split deployment, leaders with the operator and the follower on the robot
(zenoh router on the robot; the follower dials `tcp/127.0.0.1:7447`, override with
`--zenoh-connect`; expose Viser with `--manipulationmodule.visualization.host`):

```bash
# robot
dimos run teleop-dual-openyam-leader-follower --left-can-port follower_l --right-can-port follower_r
# operator machine
dimos run teleop-dual-openyam-leader-operator --robot-ip <robot-ip> --port-left /dev/ttyACM0 --port-right /dev/ttyACM1
```

The ports given select the sides; one leader blueprint serves one or two arms.
Leader joint N drives follower joint N, clamped to the `dual_openyam.urdf`
limits before publishing. A mapped jump above 0.75 rad between two reads drops
that command, which is the intended response to a leader segment flipping
past a limit while being handled. The follower trajectory task caps the arm
joints at 2 rad/s and the wrist joints at 4 rad/s.

The wrist motors run main's gains, kp 10 and kd 1.5, which follow a leader
smoothly but late. `--controlcoordinator.wrist-kp` and
`--controlcoordinator.wrist-kd` change them for a run. On the rig, kp 25 to 80
with kd 1.5 made the wrists timely but showed the 50 Hz staircase of streamed
targets as a ticking; kd above 1.5 made the small wrist motors vibrate on
their own, so leave kd alone.

## Smooth streaming

On one machine the leader module is linked to the coordinator and streams
each tick as a two-point trajectory from the last target to the new one over
`trajectory_horizon_s` (default 0.04 s, two ticks). A single streamed point
completes within a tick whenever the hand moves slowly, and the next point
then restarts from the measured position, which lags the command by the
motors' tracking error; that restart every 20 ms is a ratchet the operator
feels. With two points the task interpolates from its own last command while
it is executing, so there is no restart. A rejected anchor (the task went
idle and the measured start drifted past its 0.05 rad tolerance) falls back
to a single point for that tick. The mapped targets also pass a first-order
low-pass at `command_filter_cutoff_hz` (default 6 Hz, `None` to disable):

```bash
dimos run teleop-dual-openyam-leader --port-right /dev/ttyACM0 \
  --yamleaderteleopmodule.trajectory-horizon-s 0.04 \
  --yamleaderteleopmodule.command-filter-cutoff-hz 6
```

Without a coordinator link (the split operator blueprint) or with
`trajectory_horizon_s` set to `None`, targets go out on `joint_command` as
before.

## Record a session

`learning-collect-leader-dual-openyam` is the leader graph plus the episode
monitor, the overhead and wrist cameras and a recorder. It stores the
measured joint state, the coordinator's intended joint commands (the dataset
action), the leaders' targets, the gripper commands, the three colour streams
and the episode status, as `learning-collect-webxr-dual-openyam` does for the
headset.

```bash
dimos run learning-collect-leader-dual-openyam --port-left /dev/ttyACM0 --port-right /dev/ttyACM1 \
  --yamleaderteleopmodule.gripper-travel-radians 1.5 \
  --controlcoordinator.left-can-port follower_l --controlcoordinator.right-can-port follower_r
```

The leaders have no buttons, so episodes are marked over RPC from a second
terminal: `toggle` starts an episode and saves the one in progress, `discard`
drops it.

```bash
python -c "from dimos import Dimos; app = Dimos.connect(); print(app.EpisodeMonitorModule.command('toggle')); app.stop()"
```

Export with the same DataPrep config as the headset sessions:
`action` reads `coordinator_joint_command` with
`source_kind: joint_position_updates`.

## Before enabling the follower

- Start with the followers at their zero pose so the first streamed target is
  within the trajectory task's 0.05 rad start tolerance.
- Hold the leaders still and in their calibrated zero pose when the blueprint
  starts; read a leader in its straight pose after handling it.
- Keep the e-stop in reach for the first motions and move one joint at a time.
