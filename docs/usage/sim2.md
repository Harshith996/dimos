# sim2 core

sim2 owns a MuJoCo world and independently scheduled camera/lidar workers.
Simulated robot modules use `SimRobotClient` for internal motor IO.
The core does not choose a locomotion policy or create public robot connections.

Robot-specific simulated modules and blueprints are follow-up work.

## Ownership

### Source layout

```text
dimos/sim2/
  spec.py                RobotConfig, RobotInstance, WorldConfig
  blueprint.py           World/device blueprint composition
  module.py              Simulation and scene RPCs
  client.py              Robot-side command/feedback client
  world/                 MuJoCo assembly, stepping, scene state, viewer, run evidence
  messages.py            Robot endpoint description, timed commands and samples
  streams.py             DimOS Zenoh pub/sub and bounded local sample history
  sensors/
    spec.py              Shared mounts and acquisition requirements
    composition.py       Supported sensor families and integrated attachments
    device.py            Acquisition clock and delivery envelope
    reader.py            Read-only world snapshots/history
    module.py            Shared worker lifecycle and publication
    camera/              spec.py, device.py, module.py, renderers/
    lidar/               spec.py, device.py, module.py, raycast.py, models/
    imu/                 spec.py, device.py, module.py
```

Put a sensor's configuration beside its measurement implementation and publisher.
Ray patterns belong to `lidar/models/`; image renderers belong to
`camera/renderers/`. Neither belongs in a robot package. A robot package owns
the choice of devices, their mounts and its actuator bindings. Tests live with
their owner; cross-sensor acquisition tests stay in `sensors/test_sensors.py`.

| Component | Owns |
| --- | --- |
| `Simulation` | World lifecycle, paced physics thread, privileged scene RPCs |
| `SimulationRuntime` | One mutable MuJoCo model/data, robot bindings, command application |
| `SimRobotClient` | One robot's internal command/observation channel; not a Module |
| `CameraDevice`, `LidarDevice`, `ImuDevice` | Measurement acquisition, simulation-time scheduling, delayed delivery |
| Sensor modules | Worker lifecycle, a local `WorldReader`, publication of device measurements |
| Simulated robot module | `describe_simulated`, controller, the real connection's streams and RPCs |

`simulation(world=WorldConfig(...))` returns a normal `Blueprint` containing one
world and the declared sensors. It does not return `HardwareComponent` entries,
register coordinator adapters, or add a second public connection for a robot.
Multiple robots are composed into that world before MuJoCo compilation.

### Simulated robots are their own modules

A real robot connection contains no simulation code. A simulated robot is a
separate module that exposes the same streams and RPCs, and a blueprint chooses
one or the other:

```python skip
robot = autoconnect(RobotConnection.blueprint())

robot_sim = Simulation.autoconnect(
    RobotSim.blueprint(robot_id="robot"),
    SimRGBDCameraModule.blueprint(robot_id="robot", sensor=camera),
    LidarModule.blueprint(robot_id="robot", sensor=lidar, root_body="base"),
    scene="path/to/scene.xml",
).remappings([(LidarModule, "pointcloud", "lidar")])
```

MuJoCo compiles every robot together. A simulated module therefore has a static
`describe_simulated(config)` returning a `SimulatedRobot` or `SimulatedSensor`.
`Simulation.autoconnect` reads these from the blueprint's atoms before anything
is deployed and passes them to `Simulation` as its `simulated` config, so the
world compiles in `build()`. Descriptions come from the blueprint's own kwargs:
set spawn and sensor mounts there, not as CLI overrides. Add robots by
rebuilding the complete blueprint.

`Simulation` takes a scene path (default `--scene-package`). An
explicit `world=WorldConfig(...)` can also preset robots, timestep and
scene-mounted devices. A named spawn places a robot on that scene support. The
world timestep is lowered to the smallest robot `max_timestep`. Several robots
share one world under different robot IDs, each with its own device channel.

`RobotConfig` binds an MJCF model, joints/actuators and named sensor attachments.
`RobotInstance` adds placement, initial joint positions, the explicitly selected
control IMU, startup/watchdog settings and controller provenance. A robot has no
whole-body/manipulator packet category. Every joint states whether it is an
effort motor or a native position servo; one robot can contain both. Fixed servo
gains belong to the device; policy PD gains arrive with each motor command.

Scene-mounted devices use the same definitions and workers. A mount names a
body in the scene; `world` is a fixed frame. A named MJCF camera keeps its
authored intrinsics. No robot is required for a sensor-only world.

```python skip
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.sensors.spec import Mount
from dimos.sim2.spec import WorldConfig

world = WorldConfig(
    scene=scene_path,
    robots=robot_instances,
    sensors=(Camera("overhead", Mount("world", xyz=(1.5, 0, 3))),),
)
blueprint = simulation(world=world)
```

The external camera publishes on `world/overhead/color_image` with frame
`world/overhead_optical`. Robot sensor topics are always qualified by robot and
device name, even with one sensor. Application blueprints explicitly select the
device they consume; adding a device cannot rename existing streams.

## Internal device interface

Construct the client with the running world and robot ID. The client discovers
the endpoint through `describe_robot()` and checks the joint order. Transport
descriptors stay inside sim2; robot modules do not construct or decode them.

```python skip
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.sim2.blueprint import simulation
from dimos.sim2.client import SimRobotClient
from dimos.sim2.module import Simulation
from dimos.sim2.spec import WorldConfig

app = ModuleCoordinator.build(simulation(world=WorldConfig(scene_path, robot_instances)))
try:
    world = app.get_instance(Simulation)
    with SimRobotClient(world=world, robot_id="arm") as device:
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
| `read()` | `RobotFeedback`: `JointState`, optional `Imu`, ground-truth `PoseStamped`, base `Twist`, episode/tick/time metadata |
| `start()` / `stop()` | Attach and run typed feedback callbacks; stop the publisher and release the connection |
| `write(MotorCommandArray)` | Forward a timestamped command using current feedback, without restamping it |
| `write_motors(MotorCommandArray, observed=...)` | Complete finite q/dq/kp/kd/tau arrays in descriptor order |
| `write_positions(positions, observed=...)` | Native position targets, including declared gripper units |
| `disable(observed=...)` | Zero effort outputs; native position servos target current position |
| `set_estop(True)` | Latch configured motor damping until world restart; not cleared by a reset |

Use `writable=False` for a feedback-only client. Exactly one writable client may
own a robot's actuator set; read-only clients cannot command or halt it. Vendor
sentinels must be normalized by the vendor integration before writing commands.

Feedback and command timeouts default to 250 ms and are explicit client settings.
The runtime rejects expired commands independently of the writer. On expiry it
holds the measured joint position using the instance's holding gains; it
does not replay the old target. Disabled effort outputs remain disabled until a
new enabling command or reset. None of these operations guarantees balance for
a floating robot, replaces a physical estop, or implements vendor firmware.

## Scene and reset boundary

The world exposes `describe_scene`, `scene_state`, `set_scene_state`, `reset`,
`set_spawn`, `set_paused`, `set_truth_enabled`, and `status`. These are privileged
operator/evaluator APIs, not robot perception or ordinary agent skills.

Scene edits validate the complete update before changing existing named bodies
or joints and preserve each robot's motor-enable state, including teleports.
Reset restores the authored baseline. Both invalidate old motor
commands and publish the new generation immediately. A policy result computed
before a reset cannot command the new generation, including a reset racing with
command publication. Application-level reset must also stop/reset controllers,
maps and policy history; physics reset alone does not do that.

`SceneState.episode` counts physical resets. `SceneState.generation` also
advances for an edit or resume, invalidating pending control and sensor work
without starting a new task episode. Graders must retain both, plus simulation
time, rather than treating every scene edit as another trial. Internal transport
`FrameMetadata.episode_id` carries that invalidation generation.

## Transport and tests

`device.poll()` returns typed `Measurement` values without requiring a Module,
transport or dedicated process. Each value records acquisition start/end,
modeled availability, and epoch. `device.published(value)` separately records
actual host publication and lateness. Device rates follow simulation time, not
worker polling. Instantaneous devices acquire the newest due sample and count
missed samples; rolling lidar requires complete history and reports dropped
scans/IMU samples. Pending output is discarded on an epoch change. Pause stops
acquisition; resume reanchors the clock and invalidates old work.

The publication timestamp is producer-side, not downstream receipt time. The
measurement envelope stays inside the device/module implementation; public
streams still carry Image, PointCloud2 and Imu messages. A configured delay is
an explicit model parameter, not evidence that the hardware delay was measured.

Each sensor definition supplies its required sample rate/history duration.
Physics sizes the snapshot history from these requirements without recognizing
particular lidar models. The selected coherent control IMU travels with joint
feedback; additional IMUs can be independent streams. Delayed coherent control
IMU feedback is not implemented and is rejected explicitly.

The shipped modules still use dedicated workers. That is a deployment default,
not a requirement of the measurement implementations; this change makes no new
claim that processes outperform threads.

### Run evidence

CLI runs retain a small JSON record under the existing run log's `sim2/`
directory. Direct Python users can set `Simulation.evidence_dir` or call
`describe_run()`. It includes resolved device parameters, controller provenance,
initial MuJoCo integration state, model/source hashes and MuJoCo version.
Controller provenance contains the values supplied by the robot instance. It
does not inspect a policy loaded in another process: custom controller weights
and configuration must be supplied explicitly and checked against that run.
`WorldConfig.calibration_evidence` records parameter-to-evidence references;
`sampled_variation` records caller-sampled values, distributions and seeds. These
are separate: randomizing a parameter does not calibrate it. Unlisted parameters
are unvalidated, not implicitly measured. Sensor-model RNG seeds are also
preserved in their resolved model arguments.

No assets or compiled models are duplicated. This record identifies the initial
conditions; it is not a portable bundle or deterministic replay of asynchronous
commands. Record those command streams through the existing DimOS recorder.
The initial record does not include later scene mutations or controller resets.

Motor commands, robot feedback and world snapshots use DimOS `pZenohTransport`.
Startup uses the existing `ZenohRPC` to obtain a first sample and acquire the
robot's command ownership. It is not called for each command or physics tick.
The world accepts one writer per robot; commands and ownership expire if it
stops sending. A newly attached writer has a 30-second startup allowance.
Closing the writer releases it immediately. A killed writer cannot leave a
shared-memory segment or file lock behind.

Every command names its observation epoch and retains its original issue time.
Reset discards held commands and rejects delayed pre-reset messages. A client
reads cached feedback without waiting; stale feedback raises rather than being
restamped. Source timestamps are Unix seconds; expiry uses the same host's
monotonic clock. Remote physics/clients are not yet supported.

Snapshots carry the complete MuJoCo integration state. Each sensor keeps a
bounded local history, clears it on reset, and refuses to interpolate over a
missing interval. Only the world steps physics; sensing workers restore the
snapshots into their own model/data. `mj_forward` refreshes derived poses and
sensor readings after integration, before publishing the post-step sample.
Zenoh's native SHM optimization remains at the repository default. This code
does not require or claim zero-copy delivery.

```bash
uv run pytest dimos/sim2
uv run pytest -m self_hosted dimos/sim2  # Requires a working native OpenGL renderer.
uv run mypy --follow-imports=silent dimos/sim2
```

Tests use small generated MJCFs without asset downloads. Coverage includes two
robots, complete PD commands, gripper units, exclusive writers, spawned and
independent-process readers, stale feedback/commands, reset races, scene edits,
RGB-D/raycast output, and a real headless DimOS deployment using Zenoh.
These are core integration checks, not robot-policy or loaded-performance results.
