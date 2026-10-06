# G1 SONIC controller

SONIC runs the planner, encoder and decoder at a 50 Hz policy rate. It accepts
standard Twist walking commands, selectable gaits, motion clips and optional
upper-body reference targets. No headset or teleop service is required.

The default gait is `SLOW_WALK`. Velocity commands set direction and speed
within that gait; they never select walking or running automatically. With a
walking gait selected, zero input idles while preserving that selection. Use
`set_locomotion_mode` on G1SonicConnection to change modes; `None` restores
`SLOW_WALK`.

Install the workstation dependencies and NVIDIA assets:

```bash
uv sync --extra control --extra cuda --extra sim \
  --no-install-package onnxruntime --reinstall-package onnxruntime-gpu
source .venv/bin/activate
dimos-sonic-models
dimos-sonic-models --check
dimos --transport zenoh --simulation mujoco --viewer none run unitree-g1-sonic-wbc
```

The CPU and GPU ONNX Runtime distributions share the same Python package;
excluding the CPU distribution prevents it overwriting the CUDA provider.
Keep these options when syncing this environment again.

Assets live in the DimOS cache (`~/.cache/dimos/sonic` by default), outside
the Git LFS data directory. `SONIC_MODEL_DIR` overrides this location for
both the installer and blueprint. The installer pins the Hugging Face revision and verifies SHA-256 hashes for
the policy, planner and observation configuration. The 13 example motion clips
come from a pinned NVIDIA deployment revision. `--check` performs no downloads.
SONIC v1.1 is the single supported policy bundle.

MuJoCo starts in the SONIC pose and waits for a complete control command before
advancing physics. Simulation arms automatically. Commands from a second shell:

```bash
dimos --transport zenoh shell
```

```python
c = app.ControlCoordinator
c.list_tasks()  # ['sonic_wbc', 'joint_trajectory']
c.task_invoke("sonic_wbc", "state_snapshot")
c.task_invoke("sonic_wbc", "set_locomotion_mode", {"mode": "SLOW_WALK"})
c.task_invoke("sonic_wbc", "list_motion_clips")
c.task_invoke("sonic_wbc", "play_motion_clip", {"name": "macarena_001__A545"})
c.task_invoke("sonic_wbc", "stop_motion_clip")
c.set_estop(True)
```

E-stop latches damping. Releasing a control, resetting runtime state, or arming
again cannot clear it; recovery requires restarting the stack. Hardware also
checks feedback and command freshness in its independent motor publisher.

For JetPack 5 or 6, use `bin/hardware/g1/setup-sonic`. It detects the release and
installs the matching ONNX Runtime and model assets; `--check` checks system
prerequisites only. JetPack 5 requires CUDA 11.8 with its compatibility driver
and cuDNN 8; JetPack 6 requires CUDA 12.6 and cuDNN 9.
Hardware starts unarmed with policy outputs in dry-run.
After checking startup and feedback, the hardware activation RPCs are:

```python
c = app.ControlCoordinator
c.set_dry_run(False)
c.set_activated(True)
```

The connection/coordinator refactor has not been validated on physical hardware.

Squat and kneeling request zero translation. Centered sticks stop crawling
while retaining its posture. Face-down mode 7 is unavailable, matching NVIDIA's
selectable motion menu. Floor transitions remain timed; crawl stability and
the earlier hardware instability remain open validation issues.

## Module interface

The blueprint runs both `ControlCoordinator` (the G1-specific `SonicCoordinator`)
and `G1SonicConnection`. Command flow is:

```text
navigation / teleop -> ControlCoordinator -> G1SonicConnection -> SonicController -> motor IO
                      selects references                       runs SONIC at 50 Hz
```

The coordinator uses the existing velocity task (`sonic_wbc`, retaining its shell
name) for walking, and the existing `joint_trajectory` task for arm references.
Task priority arbitration chooses their outputs. `cmd_vel` enters its
`twist_command`; named arm targets enter `joint_command` on `/g1/joint_command`.
Arm trajectories can also use `execute_trajectory()`. These resources describe
policy inputs, not independent control of the legs or arms. SONIC keeps balancing
when a reference task finishes or walking input expires.

The connection exposes streams and RPCs and delegates to `SonicController` in
`sonic_controller.py`. That controller owns the 50 Hz loop, pose ramp and fault
latch; `SonicPipeline` handles model inference. `G1WholeBodyConnection` owns real
motor IO and its independent 500 Hz publisher. MuJoCo uses the existing
shared-memory whole-body adapter. SONIC sends no motor commands before its models load.

- `base_command: In[Twist]`: `linear.x`, `linear.y`, and `angular.z`, on
  `/g1_base/cmd_vel`. The coordinator publishes its selected walking command
  here. After one second without fresh upstream input it sends zero velocity;
  the connection also expires commands if the coordinator stops publishing.
- `position_command: In[JointState]`: named arm references, on
  `/g1/position_command` in this blueprint. Use the arm names from measured
  `joint_state`, for example `g1/left_shoulder_pitch`. Partial arm updates hold
  the other reference values. Unknown names, leg targets, duplicate names,
  missing positions and non-finite values are rejected. These are encoder
  references, not independent joint servos.
- `joint_state: Out[JointState]`: measured state for all 29 joints, remapped to
  `g1_joints` (`/g1/joints`) for existing visualization.
- `imu: Out[Imu]`: measured IMU state. Raw hardware feedback uses the separate
  `low_level_imu` input. Odometry still comes from simulation or localization.

For example, send `JointState(name=["g1/left_shoulder_pitch"], position=[0.2])`
to the coordinator's `/g1/joint_command` input. Its trajectory task moves the
reference at its configured rate and forwards only claimed arm joints.
`c.task_invoke("sonic_wbc", "set_upper_body", {"positions": [...]})` submits
a complete 14-element arm target in DDS order through that same task.
`set_vr_3point()` retains its existing wrist/head reference
contract; a generic Cartesian command stream is not implemented here.

`arm()`, `disarm()`, `set_dry_run()`, `set_locomotion_mode()`, motion-clip
commands, `reset_runtime_state()` and `status()` are direct module RPCs.
`disarm()` returns to measured-pose hold; it does not keep the balancing policy
running. `set_estop(True)` and `halt_robot()` latch damping. Ordinary zero
walking commands and command expiry keep balancing and do not clear faults.

The former `task_invoke("sonic_wbc", ...)` commands remain available. Walking
and arm targets enter coordinator tasks; gait, height, clip and VR settings
forward to the policy. Coordinator activation/reset discards pending references,
and coordinator E-stop reaches the connection's fault latch before waiting for
task locks. Use coordinator inputs in this blueprint; direct connection commands
bypass task arbitration. The connection can still be composed without a coordinator.
Stopping just the coordinator stops reference delivery; SONIC returns to zero
walking velocity and keeps balancing. Stopping the complete stack also stops the
connection, which latches damping. Use `set_estop(True)` for an immediate damping stop.

This uses the existing `Module` base and the standard port names proposed by
the ControlCoordinator refactor. The unmerged `ConnectionModule` base and its
`describe_control()` discovery contract are not dependencies of this branch.
