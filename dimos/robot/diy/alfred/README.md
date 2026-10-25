# Alfred

A DIY mobile manipulator: FlowBase holonomic base, pillar lift, bimanual OpenArm v2.0, a Mid-360 on
the mast and a D455 facing forward. The robot computer is an Orin NX; DimOS Desktop runs on it and
its Controller is the UI.

## Blueprints

| Blueprint | What runs |
| --- | --- |
| `alfred-full` | Mid-360 navigation on the coordinator's holonomic follower, the pillar and both arms planned on that coordinator. The demo stack. |
| `alfred-full-perception` | `alfred-full` plus the D455: an obstacle map for the arm planner and 2 Hz detections. |
| `alfred-nav` | Navigation alone, on the native trajectory follower and MovementManager. |
| `alfred-mls-nav`, `alfred-mls-nav-lidar` | dimSLAM variants (D455 odometry). |
| `alfred-keyboard-teleop` | Driving only. |
| `alfred-pillar` | The lift alone, for homing and bench tests. |
| `alfred-sim` | Mock base, lift and arms planned in viser; no robot. |

```bash
uv sync --extra alfred --extra manipulation          # pyserial for the pillar, the planner's deps

dimos run alfred-full                                                # arms mocked, pillar real
OPENARM_LEFT_CAN=can2 OPENARM_RIGHT_CAN=can3 dimos run alfred-full   # real arms; `dimos hardware can setup` first
```

Everything rides the default transport (zenoh), which is what Desktop reads: open the Controller
while a blueprint runs, drive on `tele_cmd_vel`, and plan the arms from its arm view (install the
Controller's planner hook once, from its panel, and restart the blueprint).

## Driving and navigating from the Controller

- **Drive**: the Controller picks its drive topic by catalog *port* name, so it misses the
  coordinator's `twist_command` (wired to `tele_cmd_vel`). Set Settings → Drive → Topics to
  `/tele_cmd_vel` once (done on this robot:
  `dimos-desktop run dim-controller patch-settings` with `lv.drive.wheeled`).
- **Navigate**: right-click the 3D view → **Go here**. That publishes `clicked_point`, which
  the MLS planner listens on directly (no MovementManager in alfred-full). Click on floor the
  Mid-360 can see; the robot's own cell is in its blind zone.
- **Stop navigating**: the action bar's **Nav: stop** button
  (`python -m dimos.robot.diy.alfred.tools.tool_nav stop`) publishes the nav stack's cancel, a
  NaN `clicked_point`. Teleop keys outrank the follower only while held.

## Planning the arms from the Controller

The planner's world is the robot itself: `base_link`, with tf's `world` an identity edge under
it. A gizmo is therefore a pose on the robot, which rides along when the base drives and does not
care where Point-LIO woke up or whether its odometry jumps (it does, metres at a time, on a
cluttered bench). Tick `left_manipulator` and `right_manipulator` (the lift is its own group),
then **Preset → Current** puts a gizmo on each real hand; drag, **Plan**, **Preview**,
**Execute**. Tick `moving_base` too and the planner may drive to reach: its base joints are the
base's motion *from where it stands* (zero at every plan), and the coordinator's base task lays
that motion down from its odometry pose only while it runs — so **Execute moves the robot**, and
the preview's ghost does not show the base going (the Controller's URDF has no base joints).
Things that look like planner failures and are not:

- **Plan fails with "requires interval-only joint coordinates"**: the arm view is in
  *cartesian* mode (a straight TCP line), which RoboPlan cannot run on a model with a planar
  base. Set the mode to *joint* (`dimos-desktop run dim-controller patch-manipulation-options
  mode: joint`, done on this robot): a dragged gizmo still goes through the planner's IK, and
  the path to those joints is collision-checked.
- **The gizmo looks left-handed after a turn**: the Controller redraws each arrow on your side
  of the camera; the frame underneath is unchanged.

The Damiao motors are torque-free until their first command, so the coordinator's `hold_arms`
task holds the arms where they are from its first tick (`always=True`) and re-latches wherever a
plan leaves them: a restart no longer drops the arms.
- **Small turns plan, bigger ones do not**: from the hanging pose the wrist (`joint6`, ±45°) and
  the straight elbow (`joint4` ≥ 0) sit at their limits. Go **Preset → Home** (the ready pose)
  first; it is reachable from hanging.

## How the base is shared

The FlowBase is three virtual joints (`flowbase/vx|vy|wz`) on one `ControlCoordinator`. Teleop
(priority 20), a whole-body plan's base segment (15, a relative motion from wherever the base is)
and the navigation follower (10) are ordinary joint claims on them. The coordinator's base adapter publishes the arbitrated twist on
`/flowbase/cmd_vel`; `AlfredHighLevel` reads that and stays the only FlowBase writer. The adapter
reads the base pose back off `/flowbase/odom`, which `StartRelay` fills from tf.

The follower is `holonomic_pose_follower` on Alfred's measured plant model,
`alfred_posedomain.json`, characterized on the real FlowBase (`mb/alfred-locomanip` has the autotune
that wrote it). The file is config: a checkout without it still starts, on the Go2's gains, with a
warning.

## Pillar bring-up

The Nano enumerates as a CH340 at `/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0`, the
connection's default. Opening the port resets the board, and the firmware does not retain its home
reference across a reset: the connection waits out the reset, re-applies `set echo 0` and
`set rate` on every `ready` event, and refuses to move until homed.

```bash
dimos run alfred-pillar                                     # or --pillarconnection.device-path /dev/ttyUSBn
```

```python
# dimos shell, from another terminal
pillar = app.PillarConnection
pillar.home()
pillar.get_status()        # repeat until homed=True and phase="idle"
pillar.set_position(-0.10)  # down: more negative away from the top switch
pillar.set_position(-0.05)  # back to the post-home parking height
pillar.stop_motion()        # ramped stop; not an e-stop
```

Do not home until the physical UP direction, normally-closed top switch and SSR brake polarity have
been commissioned. The present firmware has no bottom limit switch. The safe command range is
`-0.500` to `-0.002` m, zero at the top switch, `-0.050` m parking. Motion commands preempt each
other; a reversal still rides the ramp down first (`v^2/2a`, 2.7 mm at the default 40 mm/s).

Once homed, the lift is one joint in metres on the coordinator:

```bash
dimos topic send /pillar/joint_command 'JointState(name=["pillar/lift"], position=[-0.10])'
dimos topic echo /pillar/joints
```

## Model

The robot description is the LFS archive `alfred_description`: `alfred_v1.urdf` (casters welded)
and `alfred_v2.urdf` (eight steer/drive caster joints, drawn but not driven), built from the Onshape
CAD by the bundled `build_alfred_urdf.py`; its README has the frame table. `alfred_model.py` wraps
it with the coordinator joint names. `pillar/lift` is zero at the top limit switch, positive up,
range -0.500 .. -0.002 m, the same convention as the firmware. `AlfredMountTf` publishes the mount
tree re-rooted at `mid360_link`, since Point-LIO owns the lidar's parent edge.
