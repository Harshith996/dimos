# SONIC in Desktop

Run the existing SONIC blueprint with Zenoh. macOS simulation selects CPU
inference automatically; real hardware still requires CUDA.
Desktop's Controller app uses the
running module's RPCs, not a second controller or motor publisher.

```bash
uv run dimos --simulation mujoco --transport zenoh --viewer none run unitree-g1-sonic-wbc
```

Use the SONIC-enabled Controller app linked from this PR. It discovers the policy
from `/active/modules`, reads `status()`, and calls the existing gateway's
`/active/modules/{id}/rpcs/{name}` endpoint. Each request includes the `run_id`
that the operator inspected; the gateway rejects a request for a replaced run.
The UI never retries activation automatically.

| Control | Operation |
| --- | --- |
| Activate | `G1SonicPolicyModule.activate()`: fresh joint/IMU feedback, clock checks on hardware, then the measured-pose ramp into live balancing. |
| Stop motion | `MovementManager.stop_motion()` cancels navigation; policy `stop_motion()` clears velocity, clips and operator references. Balancing continues. |
| Deactivate | `disarm()` stops balancing and holds the measured pose. This is not a safe standing mode for an unsupported humanoid. |
| Damping stop | `G1WholeBodyConnection.set_estop(True)` on hardware; policy `halt_robot()` in sim. Latched until restart. This is software damping, not a power disconnect. |

Keyboard/gamepad driving uses the existing teleoperation velocity stream. The UI
blocks driving while policy status is stale, inactive, dry-running or faulted.
Motor watchdogs remain in the robot stack; the browser is not a safety controller.
`dry_run` still holds the prepared robot pose and is not a no-motor-output mode.

Start in simulation. Hardware testing requires the existing SONIC setup, gantry
support, an operator and exclusive ownership of the motor command interface.
No sim2, native Pico service, or ControlCoordinator is required.

## Third-person view

Add `--mujocosimmodule.enable-observer true` after the blueprint name to publish
an operator-only `observer_image` stream. In Controller, select `observer_image`
in a camera panel; keep `color_image` in another panel for the head view.
The observer follows the floating base from outside at 10 Hz and renders RGB
only. It has no camera calibration or TF and must not be used as robot perception.
Disable it when measuring control timing without viewer overhead.

```bash
uv run --no-sync dimos --simulation mujoco --transport zenoh --viewer none \
  run unitree-g1-sonic-wbc \
  --mujocosimmodule.enable-color true \
  --mujocosimmodule.camera-name head_color \
  --mujocosimmodule.fps 10 \
  --mujocosimmodule.enable-observer true
```
