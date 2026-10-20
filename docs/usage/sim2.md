# sim2 core

sim2 owns a MuJoCo world and independently scheduled camera/lidar workers.
Robot connections and policy hosts use `SimRobotClient` for internal motor IO.
The core does not choose a locomotion policy or create public robot connections.

This branch extracts the existing emulator onto current main. Robot blueprint
migrations and automatic collection of robot descriptions are subsequent work;
the existing MuJoCo blueprints have not been switched by this change.

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

`RobotConfig` binds an MJCF model, joints/actuators and named sensor attachments.
`RobotInstance` adds placement. Internal `ControlInterface` values select the
motor-frame layout (complete PD commands or native position targets), not a
public walking/arm interface or a robot brand.

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

## Transport and tests

Motor IO and world snapshots retain private shared memory. Ordinary image,
point-cloud and scene-truth streams use the configured DimOS transport. Only the
world steps physics; sensing workers restore snapshots into their own model/data.
This extraction does not change process placement or DimOS transport behavior.

```bash
uv run pytest dimos/sim2
uv run mypy --follow-imports=silent dimos/sim2
```

Tests use small generated MJCFs without asset downloads. Coverage includes two
robots, complete PD commands, gripper units, exclusive writers, spawned and
independent-process readers, stale feedback/commands, reset races, scene edits,
RGB-D/raycast output, and a real headless DimOS deployment using Zenoh.
These are core integration checks, not robot-policy or loaded-performance results.
