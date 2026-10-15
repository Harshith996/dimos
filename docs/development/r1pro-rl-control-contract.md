# R1 Pro RL control contract audit

Audit date: 2026-10-15. Status: simulation preparation only. Deployment is blocked.

The initial 14-arm position adapter is a simulation debugging interface. It is not deployment matched. An 18-element DimOS message does not establish that the actual robot accepts 18 joint targets. No robot connection, ROS node, SDK process, or hardware command ran during this audit.

## Evidence and version scope

The selected integration checkout is based on DimOS commit `8b17f2ff38979a79112ad8c38c9a5e88bcde4c77`. The existing `/home/cc/codes/dimos` checkout is at `0a8979e10ec60c77c09a82f66ccb0f1ec518b962`. The audited command conversion, joint order, generic transport adapters, and velocity timeout behavior are unchanged between those commits. Changes in `connection.py` concern cameras and the lidar mount. All local source line references below describe the selected integration commit.

The [DimOS README](/dimos/robot/galaxea/r1pro/README.md#L12) expects a robot-side `galaxea-dimos` stack with local fixes and documents V2.3.0 configuration at line 26. That is a configuration expectation, not evidence of the installed SDK version. Searches of the existing source workspaces and caches found DimOS copies and cached URDF assets, but no separately identifiable vendor SDK checkout or robot-specific installed configuration. No firmware identifier, hardware serial, or verified SDK build was established.

The project carries the explicit restriction that official ROS2 V2.1.4 disables torso joint targets for safety. Historical primary documentation for that version could not be retrieved during this audit. The restriction remains binding. Newer documentation does not authorize bypassing it or establish that the actual robot has different support.

Current [2026 motion documentation](https://docs.galaxea-dynamics.com/R1Pro/docs/2026/software_interface/r1pro_motion_control/) lists seven arm targets per side, four torso targets, and chassis velocity targets. It describes arm target `velocity` as a speed ceiling. Listed arm ceilings are `[3, 3, 3, 3, 5, 5, 5]`, and torso ceilings are `[1.5, 1.5, 1.5, 1.5]`. Chassis ranges are ±1.5 m/s in x and y and ±3 rad/s in yaw. These are documentation values, not approved operating limits.

Current [2026 driver documentation](https://docs.galaxea-dynamics.com/R1Pro/docs/2026/software_interface/r1pro_driver/) prohibits proactive publishing to `/motion_control/control_torso`. It marks torso effort feedback unused. Arm driver messages contain position, speed limits, gains, and feedforward fields, which do not establish a standalone torque-control mode. The [SDK download page](https://docs.galaxea-dynamics.com/R1Pro/en/docs/2026/development/r1pro_get_sdk/) currently advertises V2.4.0 for the 2026 model. None of these pages identifies this project's hardware.

## DimOS command mapping

| Component | Configured target | Conversion and units | Availability for this baseline |
| --- | --- | --- | --- |
| Left arm | `/motion_target/target_joint_state_arm_left`, ROS `JointState` | Seven `q` values become `position`. Seven `dq` values become tracking speed limits. DimOS uses radians and rad/s for these revolute joints. Actual SDK units and signs still need version-specific verification. | Sim-only position adapter. Hardware execution unverified. |
| Right arm | `/motion_target/target_joint_state_arm_right`, ROS `JointState` | Same conversion as the left arm. | Sim-only position adapter. Hardware execution unverified. |
| Torso | `/motion_target/target_joint_state_torso`, ROS `JointState` | Four `q` and `dq` values are forwarded by the current bridge. | Fixed initially. No deployment authorization inferred. The V2.1.4 restriction remains binding. |
| Chassis | `/motion_target/target_speed_chassis`, ROS `TwistStamped` | `linear.x`, `linear.y`, and `angular.z` pass through. Documentation gives m/s, m/s, and rad/s. DimOS odometry treats translation as body-relative. Actual SDK frame semantics remain to be verified. | Fixed initially. Future adapter uses chassis velocity targets, not wheel or planar-position targets. |

The [joint layout](/dimos/robot/galaxea/r1pro/joints.py#L22) is torso 1–4, left arm 1–7, then right arm 1–7. Coordinator names prepend `r1pro/`. The bridge uses array order, not message names, for segment conversion. Grippers and wheel/steer joints are outside this upper-body message.

The [command handler](/dimos/robot/galaxea/r1pro/connection.py#L511) accepts only an 18-joint message and slices it into 4, 7, and 7 values. It sets ROS names to `[""]` and effort to `[0.0]`. It forwards neither `kp`, `kd`, nor `tau`. The robot-side tracker owns the gains. Consequently, generic `MotorCommand` fields do not imply usable torque or gain control through this bridge.

The `_tracking_velocities` conversion at line 538 changes both zero and `VEL_STOP` to the configured `tracking_speed`, whose default is 0.5 rad/s at line 136. Sending zero `dq` therefore does not request a stop. Negative nonzero values pass through unchanged. The audit did not establish how the actual SDK handles them.

The [planning configuration](/dimos/robot/galaxea/r1pro/config.py#L33) pins the official URDF and defines synthetic planar limits of `(1, 1, 2)` for velocity and `(2, 2, 4)` for acceleration. Those are planning settings. They do not establish measured actuator limits. The generic [whole-body transport adapter](/dimos/hardware/whole_body/transport/adapter.py#L129) returns no joint limits. The connection handler performs no numeric clamping, finite-value validation, or collision check.

## Feedback and timeouts

The bridge subscribes to `/hdas/feedback_torso`, `/hdas/feedback_arm_left`, and `/hdas/feedback_arm_right`. It copies positions, velocities, and efforts in array order. Short segment messages update only their present prefix, leaving cached entries for the rest. A segment callback marks the segment seen even when its position vector is short. See [feedback callbacks](/dimos/robot/galaxea/r1pro/connection.py#L559).

The 5-second discovery timeout only logs a warning. Aggregated feedback waits until all three segments have appeared once, then publishes cached values at a configured 100 Hz. The aggregate timestamp is the oldest segment timestamp. There is no subsequent age cutoff in the [publish loop](/dimos/robot/galaxea/r1pro/connection.py#L652). A zero source stamp falls back to wall-clock receipt time. The generic whole-body transport adapter retains values and discards their timestamps, so its `has_motor_states()` flag proves only that a frame arrived once.

DimOS chassis odometry integrates `/motion_control/chassis_speed`. It rejects nonpositive time deltas and gaps over one second. It is dead reckoning, not independent footprint ground truth. See [chassis speed callback](/dimos/robot/galaxea/r1pro/connection.py#L598). The [twist transport adapter](/dimos/hardware/drive_trains/transport/adapter.py#L99) returns last commanded velocities, not measured velocities.

The default [velocity task](/dimos/control/tasks/velocity_task/velocity_task.py#L46) sends zeros after 0.2 seconds without an update when it remains active. This is a host task timeout. It does not establish robot behavior after the host process, transport, or network fails. The connection docstring delegates chassis gating to the robot-side gatekeeper, whose source, timeout, and failure semantics were unavailable. No arm command deadline or robot-side hold behavior was verified. Connection shutdown stops its nodes without an explicit final chassis zero in `R1ProConnection.stop()`.

## Integration gate

Larger training remains gated on the model and control review. Model loading alone is not dynamics validation. The simulation must preserve the distinction between a position target, a tracking-speed limit, and chassis velocity. Simulation gains are provisional until the selected tracker response is established.

Before declaring any adapter deployment matched, the integration record must identify the actual hardware variant and installed SDK build, establish authorized component targets, and verify joint order, signs, units, frames, speed limits, update rates, and target interpolation. It must also establish timestamp behavior, stale-feedback rejection, target dropout behavior, disconnect behavior, and watchdog ownership. Torso remains disabled unless the authorized installed configuration explicitly permits the intended high-level target. Direct torso motor publishing remains prohibited.

Safety constraints and watchdogs are independent of reward. Future-reference observations require separate review against live target availability. Evaluation thresholds are drafts until measured. SEW is not a prerequisite.

## Local verification record

The audit read `AGENTS.md`, relevant source, Git object IDs, and file differences. No tracked source changed. The accompanying JSON is descriptive metadata and does not enable a controller. JSON parsing and internal link line checks ran locally. The repository's `doclinks` executable was not present in PATH. No hardware tests were run or claimed.
