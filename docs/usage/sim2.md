# sim2 core

sim2 owns a MuJoCo world and independently scheduled camera/lidar workers.
Robot connections and policy hosts use `SimRobotClient` for internal motor IO.
The core does not choose a locomotion policy or create public robot connections.

This branch extracts the existing emulator onto current main. Robot blueprint
migrations are subsequent work; existing MuJoCo blueprints have not been
switched by this core change.

## Ownership

| Component | Owns |
| --- | --- |
| `SimulationModule` | World lifecycle, paced physics thread, privileged scene RPCs |
| `SimulationRuntime` | One mutable MuJoCo model/data, robot bindings, command application |
| `SimRobotClient` | One robot's internal command/observation channel; not a Module |
| Camera/lidar modules | Worker-local model/data, rendering/raycasting, typed output streams |
| Robot connection or policy host | Controller selection, command refresh, public robot interface |

`simulation(scene=..., robots=...)` returns a normal `Blueprint` containing one
world and the declared sensors. It does not return `HardwareComponent` entries,
register coordinator adapters, or add a second public connection for a robot.
Multiple robots are composed into that world before MuJoCo compilation.

### Connection-owned robot definitions

An opted-in connection declares `default_sim_scene` and the static method
`describe_simulated_robot(resolved_config, scene: Path) -> RobotInstance`.
The method must only describe the model, devices and placement; it must not
instantiate a driver, open hardware, compile physics or start a policy.

With `--simulation mujoco`, `ModuleCoordinator.build` collects these descriptions
after configuration resolution and before deploying any workers. It adds one
world and the declared sensor workers. Robot IDs are the connection's instance
names, including namespaces. Two copies named `left/arm` and `right/arm` share
one world and have separate device channels. Different default scenes require
an explicit shared scene choice; there is no arbitrary winning robot.

An explicitly supplied `SimulationModule` can set the scene, timestep, viewer
and scene-mounted devices. Its settings are preserved while the robot map is
completed. These are ordinary module configuration options when that module is
declared in the input blueprint. Automatically generated devices take settings
from the robot description, not additional CLI flags for undeclared modules.

Collected sensor outputs are always qualified: `left/arm/wrist/color_image`,
for example. The robot blueprint maps its consumers to those names; `.namespace()`
then keeps the robot stack isolated. TF messages still share the `tf` stream.
No real or DimSim connection is changed by collection. Adding another robot to
a running world requires rebuilding the complete blueprint; cross-host shared
memory clients are rejected.

`RobotConfig` binds an MJCF model, joints/actuators and named sensor attachments.
`RobotInstance` adds placement. Internal `ControlInterface` values select the
motor-frame layout (complete PD commands or native position targets), not a
public walking/arm interface or a robot brand.

Scene-mounted devices use the same definitions and workers. A mount names a
body in the scene; `world` is a fixed frame. A named MJCF camera keeps its
authored intrinsics. No robot is required for a sensor-only world.

```python skip
from dimos.sim2.sensors.spec import Camera, Mount

world = simulation(
    scene=scene_path,
    robots=robot_instances,
    sensors=(Camera("overhead", Mount("world", xyz=(1.5, 0, 3))),),
)
```

The external camera publishes on `world/overhead/color_image` with frame
`world/overhead_optical`. When scene sensors or multiple robots are present,
robot sensor topics are likewise qualified by robot and device name.

## Internal device interface

The world's `describe()` RPC supplies the exact channel descriptor for each
robot. It includes joint order, physical timestep and world-generation identity.
Do not reconstruct it from a robot name or reuse it after restarting the world.

```python skip
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.sim2.blueprint import simulation
from dimos.sim2.client import SimRobotClient
from dimos.sim2.ipc.abi import ChannelDescriptor
from dimos.sim2.module import SimulationModule

app = ModuleCoordinator.build(simulation(scene=scene_path, robots=robot_instances))
try:
    world = app.get_instance(SimulationModule)
    descriptor = ChannelDescriptor.from_dict(world.describe()["robots"]["arm"])
    with SimRobotClient(descriptor, writable=True) as device:
        observed = device.read()
        device.write_positions(target_positions, observed=observed)
finally:
    app.stop()
```

This demonstrates the internal IO boundary, not a trajectory controller. A real
connection/controller refreshes commands in its existing control loop.

| Client call | Contract |
| --- | --- |
| `connect()` / `close()` | Attach/release resources; context-manager equivalent |
| `read()` | One copied, coherent joint/IMU/base frame with episode/tick/time metadata |
| `write_motors(MotorCommandArray, observed=...)` | Complete finite q/dq/kp/kd/tau arrays in descriptor order |
| `write_positions(positions, observed=...)` | Native position targets, including declared gripper units |
| `disable(observed=...)` | Zero effort outputs; native position servos target current position |

Readers are read-only by default. Exactly one writable client may own a robot's
actuator set; public feedback readers do not acquire that ownership. Vendor
sentinels must be normalized by the vendor integration before writing commands.

Feedback and command timeouts default to 250 ms and are explicit client settings.
The runtime rejects expired commands independently of the writer. On expiry it
holds the measured joint position using the robot's configured idle gains; it
does not replay the old target. Disabled effort outputs remain disabled until a
new enabling command or reset. None of these operations guarantees balance for
a floating robot, replaces a physical estop, or implements vendor firmware.

## Scene and reset boundary

The world exposes `describe_scene`, `scene_state`, `set_scene_state`, `reset`,
`set_spawn`, `set_paused`, `set_truth_enabled`, and `status`. These are privileged
operator/evaluator APIs, not robot perception or ordinary agent skills.

Scene edits validate the complete update before changing existing named bodies
or joints. Reset restores the authored baseline. Both invalidate old motor
commands and publish the new generation immediately. A policy result computed
before a reset cannot command the new generation, including a reset racing with
command publication. Application-level reset must also stop/reset controllers,
maps and policy history; physics reset alone does not do that.

## Rolling Mid360

Use `Mid360` as a `Lidar.model` and attach its IMU to the same named site:

```python skip
from dimos.sim2.sensors.lidar.models.mid360 import Mid360
from dimos.sim2.sensors.spec import Imu, Lidar

lidar = Lidar(
    "lidar", "mid360_link", Mid360,
    imu=Imu("lidar_imu", "mid360_link"),
    self_occlusion=True,
    truth_outputs=False,
)
```

`raw_pointcloud` carries sensor-frame returns, nanosecond offsets and channel
IDs. `imu_raw` is in SI units, at 200 Hz by default. Wire these to current
`PointLio.lidar_raw` and `PointLio.imu_raw`; use zero extrinsic translation when
the simulated IMU and laser are colocated. Do not connect ground-truth pose or
the corrected `pointcloud` to the estimator. Point-LIO does not yet expose a
filter reset: restart it after resetting, teleporting or pausing its robot.
Resume preserves the physical state and motor-enable state but begins a new
observation generation, discarding pre-pause commands and acquisition history.

The scanner fires 200,000 points/s continuously across scan boundaries.
`downsample` retains complete four-channel groups and their original timestamps.
Range noise and grazing-angle dropout can be disabled independently without
changing the firing pattern. Those returns are an approximation, not a measured
reflectivity/material model. Ray casting uses moving scene/robot geometry at
acquisition time; raw scans retain motion distortion. Missing snapshot intervals
drop a scan rather than replacing it with an instantaneous one.

The Fourier implementation is shared with main's Mid360 model. The small
`mid360_pattern` LFS asset retains the earlier G1-capture frequencies, which
differ from the Go2 fit in `go2_sim`. The Go2 empirical mount-occlusion mask is
not applied to arbitrary robots.

## Transport and tests

Motor IO and world snapshots retain private shared memory. Ordinary image,
point-cloud and scene-truth streams use the configured DimOS transport. Only the
world steps physics; sensing workers restore snapshots into their own model/data.
This extraction does not change process placement or DimOS transport behavior.

```bash
uv run pytest dimos/sim2
uv run pytest -m self_hosted dimos/sim2  # Requires a working native OpenGL renderer.
uv run mypy --follow-imports=silent dimos/sim2
```

Default tests use small generated MJCFs without asset downloads. Coverage includes two
robots, complete PD commands, gripper units, exclusive writers, spawned and
independent-process readers, stale feedback/commands, reset races, scene edits,
RGB-D/raycast output, and a real headless DimOS deployment using Zenoh.
These are core integration checks, not robot-policy or loaded-performance results.

The explicit native canary builds current main's Point-LIO binary and drives a
0.6 m linear stage from the ordinary device client. It checks the estimator
against independently read truth, with and without a 640x480 RGB-D worker:

```bash
cargo build --release --locked --bin pointlio_native \
  --manifest-path dimos/hardware/sensors/lidar/pointlio/rust/Cargo.toml
uv run pytest dimos/sim2/test_mid360.py -m native_e2e \
  -o junit_family=legacy --junitxml=/tmp/sim2-mid360-native.xml
```

This requires the calibration asset and native OpenGL for the camera case.
Set `SIM2_POINTLIO_BINARY` for an externally built executable. The JUnit
`sim2_mid360_evidence` property records rates, frame age and pose errors.
This synthetic-room check is not validation of G1/Go2 locomotion, hardware
sensor equivalence, or a complex-scene performance claim.
