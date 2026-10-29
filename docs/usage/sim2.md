# sim2

sim2 runs a MuJoCo world as one DimOS module and talks to the rest of the stack
with the same messages a hardware connection uses. There is no sim-specific
transport, client or message on the wire between the simulated robot and the
coordinator.

## Layout

```text
dimos/sim2/
  spec.py                RobotConfig, RobotInstance, WorldConfig
  blueprint.py           simulation(world, module): one world plus its sensor workers
  module.py              SimulationModule: lifecycle, per-robot ports, scene and reset RPCs
  messages.py            StateFrame: the MuJoCo state snapshot sensor workers restore
  world/
    runtime.py           SimulationRuntime: compile, step, accept commands, build feedback
    scene.py             Compose scene and robot MJCFs; scene_path, scene_robot
    scene_types.py       The scene sidecar schema (entities, joints, regions, spawns)
    spec.py              SimulationSpec: describe() for sensor workers
  sensors/
    spec.py              Mounts and acquisition timing shared by all sensors
    composition.py       Supported sensor families
    device.py            Acquisition clock and delivery envelope
    reader.py            Worker-local model restored from snapshots
    module.py            Sensor worker lifecycle and publication
    camera/ lidar/ imu/  spec, device and module per sensor family
```

## Robot IO

A robot in the world is a `RobotConfig` (MJCF, root body, joints bound by
name to MJCF joints and actuators, sensors mounted on named sites or cameras)
placed by a `RobotInstance` (pose, start pose, hold gains, control IMU,
command timeout). Both are plain data.

The world module publishes and subscribes standard messages through ports named
`<robot_id>_<kind>`. A subclass declares the ports for the robots in its world:

```python skip
class G1World(SimulationModule):
    g1_motor_command: In[MotorCommandArray]
    g1_motor_states: Out[JointState]
    g1_imu: Out[Imu]
    g1_odom: Out[PoseStamped]
    g1_tf: Out[TFMessage]
```

Each physics step publishes the robot's joint state, control IMU and ground
truth odometry with one timestamp, plus the `world` to `<frame_id>` TF edge.
A `MotorCommandArray` on the command port is applied as PD plus feedforward
torque until `command_timeout` passes without a newer one; then the robot holds
where it is, like a hardware deadman. Native position servos accept positions
only. Malformed commands are counted in `status()` and ignored. A port named
for a robot that is not in the world fails at start.

Because the ports carry the hardware messages, the coordinator uses its
existing transport whole-body adapter in simulation. The G1 GR00T blueprint
remaps `g1_motor_states` to the stream `g1/motor_states`, and so on, which is
the topic the adapter subscribes to for hardware id `g1`.

## Sensors

Sensor workers run in their own processes. The world publishes a `StateFrame`
on its `snapshots` port every snapshot stride; each sensor module subscribes,
restores the state into its own copy of the compiled model (fetched through
`describe()`), and renders or raycasts there. Physics never waits on a sensor.
Cameras publish `color_image`, `camera_info` and, with depth, `depth_image`,
`depth_camera_info` and optionally `pointcloud`. Lidars publish `pointcloud`
in the world frame and `raw_pointcloud` in the sensor frame. `simulation()`
names each worker `<robot>_<sensor>` and remaps its outputs to
`<robot>/<sensor>/<port>`; blueprints remap from there to the names their
consumers use.

## Scenes

A scene is `scene.xml`, optionally with a `scene.json` sidecar
(`SceneDescription`) that names bodies, fixture joints, regions and spawn
poses. `scene_robot()` places a robot on a named spawn plus its own root
height. The world applies the sidecar's authored initial poses at load.
`describe_scene()` returns the sidecar; `reset()` restores the authored state
and starts a new episode; `set_paused()` stops physics while RPCs stay live.
Privileged scene state, scene editing and run evidence are not part of this
package.

## Running

```bash
dimos --simulation mujoco run unitree-g1-groot-wbc
dimos --simulation mujoco --scene-package workbench run unitree-g1-groot-wbc
uv run pytest dimos/sim2
uv run pytest -m self_hosted dimos/sim2   # needs a native OpenGL renderer
uv run mypy --follow-imports=silent dimos/sim2
```
