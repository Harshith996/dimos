# G1 SONIC controller

SONIC runs the planner, encoder and decoder at a 50 Hz policy rate. It accepts
standard Twist walking commands, selectable gaits, motion clips and optional
upper-body reference targets. No headset or teleop service is required.

The default gait is `SLOW_WALK`. Velocity commands set direction and speed
within that gait; they never select walking or running automatically. With a
walking gait selected, zero input idles while preserving that selection. Use
`set_locomotion_mode` on G1SonicPolicyModule to change modes; `None` restores
`SLOW_WALK`.

Install the workstation dependencies and NVIDIA assets:

```bash
uv sync --extra control --extra cuda --extra sim \
  --no-install-package onnxruntime --reinstall-package onnxruntime-gpu
source .venv/bin/activate
dimos-sonic-models
dimos-sonic-models --check
dimos --transport zenoh --simulation mujoco run unitree-g1-sonic-wbc
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

sim2 starts MuJoCo in the SONIC pose and waits for a complete control command
before advancing physics. Simulation arms automatically. Commands from a second shell:

```bash
dimos --transport zenoh shell
```

```python
c = app.G1SonicPolicyModule
c.status()
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

On the Jetson, install the clock service once from the checkout:

```bash
sudo bash bin/hardware/g1/install-jetson-clocks
```

This applies NVIDIA's `jetson_clocks` now and at each boot, after
`nvpmodel.service` restores the selected power mode. It locks CPU, GPU and EMC
clocks to the maximum allowed by that mode. It preserves the selected power mode
and fan policy. The service runs as root; DimOS runs as your normal user. It does
not start or restart the robot stack and needs no passwordless sudo rule.

Power mode and clock locking are separate: selecting MAXN alone still permits
dynamic clock scaling. A manual `sudo jetson_clocks` lasts until reboot (or a
clock-setting change).

In the 2026-10-18 Orin NX G1 deployment trial, MAXN was already selected. A
publisher-free inference harness used synthetic standing feedback for 250 steps,
with ZED HD720 at 15 FPS, NEURAL LIGHT depth and DimOS Desktop running; Rerun was
disabled. The same harness before and after locking clocks measured:

| Metric | Dynamic clocks | Locked clocks |
| --- | ---: | ---: |
| Effective rate | 45.58 Hz | 50.00 Hz |
| Mean inference step | 15.54 ms | 8.29 ms |
| p95 inference step | 28.19 ms | 13.75 ms |
| Maximum inference step | 33.93 ms | 18.65 ms |

GPU clocks were mostly 510 MHz before and fixed at 918 MHz afterward; CPU limits
were locked to 1,984 MHz. The policy's 50 Hz target gives each step 20 ms. These
numbers measure the deployed inference workload, not this PR's complete blueprint
or physical motion. The harness sent no motor commands. After locking, the
maximum standing start-to-start interval was still 27.36 ms, and a separate
synthetic turning workload had a 36.31 ms maximum inference step. Clock locking
does not establish walking stability or guarantee every control deadline.

Check the service after a reboot with:

```bash
systemctl status dimos-jetson-clocks.service --no-pager
```

`active (exited)` is expected for this one-shot service. SONIC independently reads
the CPU/GPU frequency limits before hardware startup, arming, reactivation and
switching off dry-run. Unlocked or unreadable limits reject the operation with
the command to fix it; simulation and non-Jetson hosts skip the check. The check
does not run in the policy loop. It verifies configured CPU/GPU locks, not the
power budget, EMC, thermal throttling or inference deadlines. A lower power mode
can still be too slow with locked clocks; choose and validate the mode for the
specific robot. An `active` service alone does not prove clocks were left locked.

On a configured G1, start the stack with:

```bash
dimos --transport zenoh run unitree-g1-sonic-wbc
```

Use the Controller app in DimOS Desktop for the camera, map, robot state and
policy controls. Desktop receives these streams directly over Zenoh. This
blueprint includes no Rerun bridge, Rerun WebSocket server or legacy web
visualization module, including when an old launch command supplies `--viewer rerun`.
No Rerun viewer flags or viewer tunnels are needed.

After checking startup and feedback, use **Activate policy** in Desktop, or
the equivalent hardware activation RPC:

```python
c = app.G1SonicPolicyModule
c.activate()
```

This standalone wiring has not been validated on physical hardware.

Squat and kneeling request zero translation. Centered sticks stop crawling
while retaining its posture. Face-down mode 7 is unavailable, matching NVIDIA's
selectable motion menu. Floor transitions remain timed; crawl stability and
the earlier hardware instability remain open validation issues.

## Module interface

The blueprint runs `G1SonicPolicyModule` directly, without ControlCoordinator,
coordinator tasks or a task-compatibility wrapper. Command flow is:

```text
navigation / teleop -> MovementManager -> G1SonicPolicyModule -> SonicController -> motor IO
                      selects cmd_vel                        runs SONIC at 50 Hz
```

`MovementManager` combines `nav_cmd_vel` and `tele_cmd_vel`, with teleoperation
cancelling the active navigation goal. Its `cmd_vel` output feeds SONIC's
`base_command`. Applications should use those upstream inputs instead of
competing publishers on `cmd_vel`. There is no general task-priority scheduler
or timed arm-trajectory API in this stack. SONIC owns all 29 motor targets;
arm inputs condition the policy rather than independently controlling motors.

The policy module exposes streams and RPCs and delegates to `SonicController` in
`sonic_controller.py`. That controller owns the 50 Hz loop, pose ramp and fault
latch; `SonicPipeline` handles model inference. `G1WholeBodyConnection` owns real
motor IO and its independent 500 Hz publisher. In simulation, sim2's
`WholeBodyConnection` delivers the same typed feedback and command streams
through a coherent shared-memory device. Its command timeout and controller-fault
input latch damping until restart. SONIC sends no motor commands before its models load.

The driver's
`controller_fault: In[String]` accepts faults from any motor controller;
`G1SonicPolicyModule.controller_fault` supplies that signal for SONIC.

- `base_command: In[Twist]`: `linear.x`, `linear.y`, and `angular.z`, remapped
  to `cmd_vel`. After one second without fresh input, SONIC commands zero
  walking velocity while continuing to balance.
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
to `/g1/position_command`. `c.set_upper_body(positions)` sets a complete
14-element arm reference in DDS order. The controller limits these encoder
reference changes to 1 rad/s per joint, starting from measured arm positions.
`c.clear_upper_body()` returns to the default arm references at the same rate.
`set_vr_3point()` retains its existing wrist/head reference
contract; a generic Cartesian command stream is not implemented here.

Topic paths above are logical names. Typed Zenoh keys include the `dimos/`
prefix and message-type suffix, for example `dimos/cmd_vel/geometry_msgs.Twist`.

`arm()`, `disarm()`, `set_dry_run()`, `set_locomotion_mode()`, motion-clip
commands, `reset_runtime_state()` and `status()` are direct module RPCs.
`disarm()` returns to measured-pose hold; it does not keep the balancing policy
running. `set_estop(True)` and `halt_robot()` latch damping. Ordinary zero
walking commands and command expiry keep balancing and do not clear faults.

`set_velocity_command(vx, vy, yaw_rate)` is also available for direct shell
control. It bypasses `MovementManager`, so do not use it alongside active
navigation/teleoperation. Repeat commands to sustain motion; they expire after
the same one-second timeout. Zero velocity means stop walking and keep balancing,
not disarm. Disarming, resetting, and completing the activation ramp clear old
walking commands; send fresh commands after activation.

Stopping the complete stack stops SONIC and latches damping. This uses the
existing `Module` base and does not depend on the new `ConnectionModule` or
ControlCoordinator refactor. `task_invoke`, `set_activated`, `execute_trajectory`
and `/g1/joint_command` are not part of the SONIC interface.

## Focused validation

```bash
uv run pytest dimos/robot/unitree/g1/test_sonic_policy.py \
  dimos/robot/unitree/g1/test_wholebody_connection.py dimos/control/sonic \
  -m 'not mujoco'
```

These exercise standalone command routing, expiry, re-arm/reset, fault latching,
motor-publisher safeguards, inference semantics and blueprint composition.
Models and hardware IO are mocked. The blueprint tests import optional simulation
dependencies but do not load assets or start hardware. They do not establish
physical walking stability or hardware safety.


## Desktop and sim2 integration

The `pim/feat/sonic-desktop-sim2` branch combines current `main`, the standalone
SONIC stack, the Desktop gateway and Jeff's `jeff/feat/sim2_world_builder` work.
The existing SONIC PR branches remain separate from this integration branch.
The companion Controller changes are published at
[Nabla7/dim-controller, feat/policy-control](https://github.com/Nabla7/dim-controller/tree/feat/policy-control).

Use this checkout as Desktop's DimOS installation. Install the matching Controller
and Jeff's World Builder in Desktop:

```bash
dimos-desktop install github.com/Nabla7/dim-controller --ref feat/policy-control
dimos-desktop install github.com/jeff-hykin/dim-world-builder --ref main
```

In Launcher's G1 section, select **G1 SONIC whole-body control**, then
**Simulator**. The default world is sim2's logistics scene. Controller receives
camera, depth, lidar, joint state and world-frame TF directly. Simulation starts
balancing automatically; **Stop walking** keeps balancing, **Deactivate** holds
the measured pose, and **Damping stop** latches until the stack is restarted.
The simulated robot can fall when balancing is deactivated.
This branch does not open an additional MuJoCo or Rerun viewer window.

For an exported World Builder world, install its Python package into this
checkout's environment and compose its scene publisher with SONIC. For example,
using the repository's navigation course:

```bash
git clone https://github.com/jeff-hykin/dim-world-builder ../dim-world-builder
uv pip install --python .venv/bin/python --no-deps -e ../dim-world-builder/examples/nav_test_course
source .venv/bin/activate
MUJOCO_GL=egl dimos --transport zenoh --simulation mujoco \
  --scene-package ../dim-world-builder/examples/nav_test_course \
  run unitree-g1-sonic-wbc nav-test-course.scene-data
```

`MUJOCO_GL=egl` selects headless GPU rendering on a Linux NVIDIA workstation.
For your own export, use its folder and the `<package>.scene-data` name in its
`pyproject.toml`. The folder supplies `scene.xml` and `scene.json`; the scene-data
component supplies Controller's **Simulated world** layer. Named spawn points,
scripted movers, cameras and lidar use Jeff's sim2 implementation.
Desktop discovers the individual blueprint names in a composed run so policy
controls and velocity routing remain available.

`SimulationModule.status()` reports physics time, the robot's position and
whether it is waiting for the first control command. Its `reset()` changes the
physical world only; it does not clear a latched SONIC fault or reset policy
history. Restart the stack for a complete reset. Pausing physics while the
policy is running causes its feedback watchdog to stop control.

Local integration validation on 2026-10-18 used an RTX 5090, the actual SONIC v1.1
CUDA models and the navigation-course export. The policy maintained a 20.0 ms
mean interval, with 1.38 ms mean inference and 2.07 ms p95 during the sampled
window. RGB, depth, lidar and the 13-body scene description arrived over Zenoh;
Desktop's lifecycle API passed deactivate, reactivate and damping-stop checks.
This is a simulation smoke test, not validation of stairs, slopes or the physical G1.
