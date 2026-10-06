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
c = app.G1SonicConnection
c.status()
c.list_locomotion_modes()
c.set_locomotion_mode("SLOW_WALK")
c.list_motion_clips()
c.play_motion_clip("macarena_001__A545")
c.stop_motion_clip()
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
c = app.G1SonicConnection
c.set_dry_run(False)
c.arm()
```

The standalone controller has not been validated on physical hardware.

Squat and kneeling request zero translation. Centered sticks stop crawling
while retaining its posture. Face-down mode 7 is unavailable, matching NVIDIA's
selectable motion menu. Floor transitions remain timed; crawl stability and
the earlier hardware instability remain open validation issues.

## Module interface

The blueprint runs `G1SonicConnection` without ControlCoordinator. The module
owns the 50 Hz policy loop, while `G1WholeBodyConnection` owns real motor IO
and its independent 500 Hz publisher. MuJoCo uses the existing shared-memory
whole-body adapter. SONIC sends no motor commands before its models load.

- `base_command: In[Twist]`: `linear.x`, `linear.y`, and `angular.z`. The
  blueprint remaps this to `cmd_vel`, so navigation and teleop keep their
  existing connections. After one second without a walking command, requested
  velocity becomes zero while the policy keeps balancing.
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

For example, a publisher can send
`JointState(name=["g1/left_shoulder_pitch"], position=[0.2])` to the arm input.
`set_upper_body([...])` remains available for a complete 14-element reference
in DDS arm order. `set_vr_3point()` retains its existing wrist/head reference
contract; a generic Cartesian command stream is not implemented here.

`arm()`, `disarm()`, `set_dry_run()`, `set_locomotion_mode()`, motion-clip
commands, `reset_runtime_state()` and `status()` are direct module RPCs.
`disarm()` returns to measured-pose hold; it does not keep the balancing policy
running. `set_estop(True)` and `halt_robot()` latch damping. Ordinary zero
walking commands and command expiry keep balancing and do not clear faults.

This uses the existing `Module` base and the standard port names proposed by
the ControlCoordinator refactor. The unmerged `ConnectionModule` base and its
`describe_control()` discovery contract are not dependencies of this branch.
