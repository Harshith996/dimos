# Alfred

A DIY mobile manipulator: FlowBase holonomic base, pillar lift, bimanual OpenArm with the v1.0
parallel gripper, a Mid-360 on the mast and a D455 facing forward. The robot computer is an Orin NX; DimOS Desktop runs on it and
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

## Two stacks for one Orin

The Orin NX overloads with everything on. Run one stack at a time and switch:

| Stack | Runs | Does not run |
|-------|------|--------------|
| `alfred-manipulation` | arms, lift, whole-body planner, D455 (obstacle map, scene registration, the pick pipeline with the top-down heuristic grasps), teleop, wheel odometry | lidar, Point-LIO, nav planners, follower, Rerun |
| `alfred-manipulation-graspgenx` | the same, GraspGenX proposing grasps in its own environment (first start syncs it, ~5 min) | — |
| `alfred-navigation` | Mid-360, Point-LIO, voxel map, planners, follower, teleop, the lift (parked) | arms' CAN, planner, camera, Rerun |
| `alfred-full-perception` | everything, Rerun included (capped at 256 MB: its default quarter of RAM OOM-killed the bench stack once) | — |

Neither split stack runs the Rerun bridge: the Desktop reads zenoh itself, and the bridge's
late-viewer buffer is what ran the 16 GB Orin out of memory.

The Manipulation app's **Stack** card switches them (stop, then launch); the Launcher does the
same. After every start the lift must be homed before it moves (**Home the lift** in the app,
**Pillar: home** in the Controller).

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

## Picking from the bench

**Scan**, pick an object, pick an arm (or `auto`: the nearer one), **Pick**. What the pick
pipeline does: one grasp per object (the heuristic's is top-down, GraspGenX's are ranked and
angled), a pregrasp 10 cm above it through the planner, a straight descent, close, lift.

- **Reach is the limit, not the map.** Shoulder to wrist is 0.44 m with the arm straight, the
  hand another 0.186 m. A top-down grasp on an object 0.70 m in front of `base_link` needs the
  wrist 0.58 m from the shoulder: impossible, and before the reach check Pink spent 7-15 s
  finding that out. Stand so the objects are within ~0.55 m of `base_link` (the base's front
  face nearly at the bench edge), or let GraspGenX propose a side grasp. The pick refuses an
  out-of-reach object at once and says by how much.
- **Scan again after the base moves.** Objects are registered in `base_link`; a drive leaves
  them where the robot was (a whole-body pick that drove 0.5 m put them 50 cm off). The pick
  pipeline hands an object's cloud to the grasp generator as it is and wants the grasps in its
  planning frame, so the scene cannot follow the map into `odom` until it re-expresses them.
- **"Let the base drive"** gives the planner `moving_base` as an auxiliary group: the pregrasp
  may then be a base move plus an arm move, executed together; the descent and lift stay the
  arm's. The base follows its segment through the coordinator's relative planar task on wheel
  odometry, rebased to where the run started (the Pi's own count runs since its driver booted).
  The task commands through `command_gains`, 1/K per axis of the measured plant
  (`alfred_posedomain.json`: yaw 2.85x the command, translation 0.77x); without them a yaw
  segment aborts, so a checkout without the artifact runs uncompensated with a warning.
- The planner's obstacle map is the D455's self-filtered cloud voxelised at 2 cm within 1.5 m of
  the camera, and the planner builds its octree at the same 2 cm (`MANIP_VOXEL_M`, one constant
  for both). Tuned on 60 s of the bench replayed through the mapper (2026-10-25): at 5 cm the
  bench's layer reached 3 cm above its surface, in the way of a low grasp, a can was one or two
  cells, and the Desktop drew the bench as dots 5 cm apart; at 2 cm the layer reaches 1.5 cm, a
  can is three cells across, and the map holds ~3400 voxels. The mapper then costs 19% of a core
  (13% at 5 cm, both measured on the live cloud), a collision check 2.5 ms (2.1 ms with no map),
  installing a map in the planner 35 ms once a second. At 1.5 cm the voxels nearly doubled again
  for 3 mm less layer, and 14% of a removed can never cleared.
- Clearing is for a picked object, whose voxels have to go: `grace_depth` 0.05 (a removed can is
  gone in 3 s; at 0.1 a third of it stayed for good), `graze_cos` 0.5 (rays within 30 degrees of
  a surface spare it; at 0.7 a removed can's top, seen as shallow as the bench, never quite
  cleared) and no shadow: rays carried past a return carved the layer under the bench top, and
  wherever the top straddles two voxel layers 3.4% of the map changed from one map to the next
  (0.65% without).
- The map is built in `odom` on wheel odometry and re-expressed in `base_link` for the planner at
  each map's stamp (the box filter does it; the planner drops a map in any other frame), so the
  bench stays put when the base drives: pinned to `base_link` it rode along, and after a
  whole-body pick every plan was COLLISION_AT_START until the camera had re-seen the bench. The
  octree snaps each voxel onto `base_link`'s grid, so once the base has turned since the run
  started some of the bench lands in neighbouring cells: 18% of its cells were empty with the
  base 42 degrees round (30% at 5 cm), holes of 2 cm. Open fingers sit 4.4 cm a side outside the
  self-filter model (its fingers are fixed closed), covered by 5 cm of padding at any voxel size.

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
