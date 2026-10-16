# OpenYAM quickstart

Step by step from install to teleop and motion planning on an OpenYAM arm.

**More detail**

- [OpenYAM hardware setup](/docs/capabilities/manipulation/openyam_integration.md): CAN wiring, ports, Linux and macOS
- [Installation](/docs/installation/index.md)
- [Manipulation and planning](/docs/capabilities/manipulation/index.md)
- [Controlling the arm from Python](/docs/capabilities/manipulation/python_api.md)
- [VR (WebXR) teleop](/dimos/teleop/webxr/README.md)

> **Safety:** when it starts, the arm switches on its motors and the gripper opens and closes fully once to calibrate, so keep the jaws and workspace clear. Stopping dimos (Ctrl-C) turns the motors off and the arm drops, so hold the arm before you stop it.

## 0. Install

Ubuntu 22.04/24.04 or an Apple Silicon Mac:

```bash
curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash -s -- --mode dev --project-dir ./dimos --extras manipulation,control
cd dimos && source .venv/bin/activate
```

## 1. Try it without the arm

```bash
dimos --simulation run keyboard-teleop-openyam
```

Open the 3D viewer link printed in the terminal (`http://localhost:8095`). Click the keyboard window, then:

| Keys | Action |
|---|---|
| W/S, A/D, Q/E | Move the gripper forward/back, left/right, up/down |
| R/F, T/G, Y/H | Roll, pitch, yaw |
| `[` / `]` | Open / close the gripper |

## 2. Connect the arm

Linux:

```bash
dimos hardware can list
dimos hardware can setup can0
```

macOS: plug in the USB-CAN adapter. `dimos hardware can list` shows its serial number.

Check the arm comes up:

```bash
dimos run coordinator-openyam
# not on can0? add --can-port can1 (macOS: --can-port <USB-SERIAL>)
```

## 3. Keyboard teleop on the real arm

```bash
dimos run keyboard-teleop-openyam
```

## 4. VR teleop (Quest / PICO)

```bash
dimos run teleop-webxr-openyam
```

In the headset browser open `https://<laptop-ip>:8443/teleop`, accept the certificate, tap Connect.

The right controller drives the arm. **Hold the grip (middle finger) to move; let go to stop.** The index trigger works the gripper.

Two arms:

```bash
dimos run teleop-webxr-dual-openyam --left-can-port can0 --right-can-port can1
```

Hold both grips to move.

## 5. Motion planning

```bash
dimos run openyam-planner-coordinator
```

In the viewer link it prints, drag the target, then **Plan**, **Preview**, **Execute**. **Cancel** stops.

Control it from Python in a second terminal:

```bash
dimos shell
```

```python
from dimos.manipulation.sdk import Arm
arm = Arm.from_app(app)
arm.joints(); arm.pose()
q = arm.joints(); q[0] += 0.05
arm.move_joints(q, speed_scale=0.2)
```

Two arms:

```bash
dimos run dual-openyam-planner-coordinator --left-can-port can0 --right-can-port can1
```

## No arm handy?

Add `--simulation` to any single-arm command. For the two-arm ones, leave out the CAN ports and it runs on a simulated arm.
