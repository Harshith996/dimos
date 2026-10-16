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
(zenoh router on the robot, `DIMOS_OPENYAM_ROUTER` defaults to
`tcp/127.0.0.1:7447`):

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
past a limit while being handled. The follower trajectory task caps every
joint at 2 rad/s.

## Before enabling the follower

- Start with the followers at their zero pose so the first streamed target is
  within the trajectory task's 0.05 rad start tolerance.
- Hold the leaders still and in their calibrated zero pose when the blueprint
  starts; read a leader in its straight pose after handling it.
- Keep the e-stop in reach for the first motions and move one joint at a time.
