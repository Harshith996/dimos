# sim2

`Simulation` is one simulated world, built from the modules in a blueprint that
describe themselves. It compiles them into a single MuJoCo model; stepping
physics, robot clients and sensor output are later work.

A module opts in with a static `describe_simulated(config)` that returns a
`SimulatedRobot` or `SimulatedSensor`. MuJoCo compiles every robot together, so
`Simulation.autoconnect` reads these descriptions from the blueprint's atoms
before anything is deployed and passes them to `Simulation` as its `simulated`
config:

```python skip
from dimos.sim2.module import Simulation
from dimos.sim2.spec import Mount, SimulatedRobot, SimulatedSensor


class RobotSim(Module):
    config: RobotSimConfig

    @staticmethod
    def describe_simulated(config: RobotSimConfig) -> SimulatedRobot:
        return SimulatedRobot(config.robot_id, config.model, root_body="base")


class CameraSim(Module):
    config: CameraSimConfig

    @staticmethod
    def describe_simulated(config: CameraSimConfig) -> SimulatedSensor:
        return SimulatedSensor("camera", "camera", Mount("head"), robot_id=config.robot_id)


blueprint = Simulation.autoconnect(
    RobotSim.blueprint(robot_id="dog"),
    CameraSim.blueprint(robot_id="dog"),
    scene="path/to/scene.xml",
)
```

Descriptions see the blueprint's own kwargs, not CLI overrides, so put
simulation-relevant options in the blueprint. `scene` defaults to
`--scene-package`.

`Simulation.build()` attaches each robot's MJCF to the scene at its pose, with
its names prefixed `<robot_id>/`. A camera becomes an MJCF camera and any other
sensor a site, on the named body of its robot, or on a scene body (`"world"` for
the world frame, prefixed `world/`). `describe()` returns the compiled model's
path and the robots and sensors in it. Robots cannot be added to a running
world; rebuild the blueprint instead.
