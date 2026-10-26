# OpenYAM integration

OpenYAM uses the generic Damiao whole-body adapter backed by
`can-motor-control`. The adapter owns one upstream robot containing the six-joint
arm and its gripper; motor topology is independent of the host operating system.

## CAN transport

The generic Damiao layer selects the native transport provided by
`can-motor-control>=0.0.8`:

| Host | Default | `--can-port` override |
|------|---------|-----------------------|
| Linux | `SocketCanBus("can0")` | SocketCAN interface, such as `can1` |
| macOS | first `1d50:606f` gs_usb adapter | USB serial number |

The macOS transport is implemented in Rust with native IOKit access. It does
not require PyUSB, libusb, `python-can`, or the Python `gs_usb` package.

Logical buses are ordered. Without explicit selectors, a future two-bus Damiao
robot maps to `can0`/`can1` on Linux and gs_usb indices 0/1 on macOS. Production
multi-arm deployments should configure USB serial numbers because enumeration
order is not a stable device identity.

List the selectors accepted on the current host:

```bash
dimos hardware can list
```

On Linux this prints SocketCAN network-interface names. Configure each physical
interface before use:

```bash
dimos hardware can setup can0
dimos hardware can setup can1
```

On macOS, connect a `gs_usb`/candleLight-compatible adapter with USB ID
`1d50:606f` for each arm. No SocketCAN interface or extra USB library is needed.
The motor buses still require CAN-H/CAN-L wiring, motor power, and correct bus
termination. To assign sides reliably, connect one adapter at a time, run the
list command, record and label its USB serial, then repeat for the other arm.
Adapters used together must expose unique serial numbers.

## Run

Use the default CAN device:

```bash
dimos run coordinator-openyam
```

Select another Linux SocketCAN interface:

```bash
dimos --can-port can1 run coordinator-openyam
```

Select a macOS adapter by USB serial number:

```bash
dimos --can-port <USB-SERIAL> run coordinator-openyam
```

The dual-arm WebXR blueprint is identical on both operating systems; only the
selector values differ:

```bash
# Linux
dimos run teleop-webxr-dual-openyam --left-can-port can0 --right-can-port can1

# macOS
dimos run teleop-webxr-dual-openyam \
  --left-can-port <LEFT-USB-SERIAL> \
  --right-can-port <RIGHT-USB-SERIAL>
```

Linux interfaces must already be configured for classical CAN at 1 Mbit/s.
The native macOS transport configures the adapter bitrate directly.

### Headset cameras

The headset shows the egocentric camera in a large panel above the workspace
and each wrist camera in a smaller panel beside its arm. Pick the webcams per
panel; a camera left unset only blanks its panel:

```bash
dimos run teleop-webxr-dual-openyam \
  --left-can-port can0 --right-can-port can1 \
  --egocentric-camera.device /dev/v4l/by-id/<scene-camera> \
  --left-wrist-camera.device /dev/v4l/by-id/<left-wrist-camera> \
  --right-wrist-camera.device /dev/v4l/by-id/<right-wrist-camera>
```

See the [WebXR README](/dimos/teleop/webxr/README.md) for the panel layout,
border colors, and uplink limits.

#### RealSense RGB overhead camera on macOS

The `teleop-webxr-dual-openyam-realsense` blueprint reads the overhead camera's
RGB stream through the existing native RealSense module. The wrist cameras still
use their OpenCV devices. Build the native module once as your normal user:

```bash
brew install librealsense
bash dimos/hardware/sensors/camera/realsense/rust/build-macos.sh
```

On macOS 12+, librealsense needs elevated USB access. Start this blueprint from
the repository root with the virtual environment's `dimos` executable under
`sudo`; use the serial reported by `sudo rs-enumerate-devices -s`:

```bash
sudo .venv/bin/dimos run teleop-webxr-dual-openyam-realsense \
  --left-can-port <LEFT-USB-SERIAL> \
  --right-can-port <RIGHT-USB-SERIAL> \
  --egocentric-camera.serial-number <REALSENSE-SDK-SERIAL> \
  --left-wrist-camera.device <LEFT-OPENCV-INDEX> \
  --right-wrist-camera.device <RIGHT-OPENCV-INDEX>
```

The overhead stream defaults to 640×480 RGB8 at 30 fps. Camera indices can
change when devices are reconnected; verify the two wrist feeds before running.

### Arm posture

With body tracking enabled in the headset browser (Meta Quest Browser 32 and
later list WebXR body tracking as an experimental feature), the operator's
forearm direction steers each robot elbow: `left_link3` and `right_link3`
follow the wrist-to-elbow direction of the matching arm as a low-weight Pink
task, so the arm's posture mirrors the operator's instead of being whatever
configuration reaches the hand. Without body tracking the hint is inert and
teleop is hand-pose only. The weighting lives in the teleop blueprint
(`hint_position_cost`); see the [WebXR README](/dimos/teleop/webxr/README.md)
for the mechanism.

### Record episodes

`learning-collect-webxr-dual-openyam` adds the episode monitor, a recorder
that captures all three cameras, measured joint state, post-arbitration joint
targets, and episode markers, plus an exporter that rebuilds a LeRobot dataset
after every saved episode. The LeRobot `observation.state` feature contains
measured positions; `action` contains the absolute position targets sent to the
arm and gripper hardware. Press B
to start a take and B again to save it, or Y to discard it; the zero-pose reset
is on A in this blueprint. The headset HUD shows a blinking `REC` timer while
recording and the export result afterwards.

```bash
dimos run learning-collect-webxr-dual-openyam
# recording:  recordings/session_dual_openyam_<ts>.db
# dataset:    datasets/dual_openyam_<ts>   (rebuilt on every save)
```

For a RealSense overhead camera, use
`learning-collect-webxr-dual-openyam-realsense` with the same CAN selectors,
RealSense serial number, and wrist device indices as the teleop-only blueprint:

```bash
sudo .venv/bin/dimos run learning-collect-webxr-dual-openyam-realsense \
  --left-can-port <LEFT-USB-SERIAL> \
  --right-can-port <RIGHT-USB-SERIAL> \
  --egocentric-camera.serial-number <REALSENSE-SDK-SERIAL> \
  --left-wrist-camera.device <LEFT-OPENCV-INDEX> \
  --right-wrist-camera.device <RIGHT-OPENCV-INDEX>
```

The `teleop-webxr-dual-openyam-realsense` blueprint provides camera viewing and
arm control. Select the `learning-collect-...` blueprint to enable the
collection label, recording HUD, episode controls, and automatic LeRobot export.

Give a collection run a short label with `--dataset-suffix`. Use letters,
numbers, hyphens, or underscores; the timestamp remains in the name so separate
runs never overwrite one another:

```bash
dimos run learning-collect-webxr-dual-openyam --dataset-suffix pick-red-cube
# dataset: datasets/dual_openyam_<ts>_pick-red-cube
```

The Quest `/teleop` page also shows a **Collection label** field before the
Connect button. Enter a label and press Enter or **Set label** before recording;
spaces and punctuation are converted to hyphens. The label locks when recording
begins so every saved take in the session stays in one dataset folder.

See the [imitation learning guide](/dimos/imitation/README.md) for the full
collection workflow.

## Hardware topology

| Motor | Send ID | Reply ID | Type |
|-------|---------|----------|------|
| joint1–3 | `0x01`–`0x03` | send ID + `0x10` | DM4340 |
| joint4–6 | `0x04`–`0x06` | send ID + `0x10` | DM4310 |
| gripper | `0x08` | `0x18` | DM4310 |

The bus uses classical CAN at 1 Mbit/s. Opening the gripper decreases motor
position. The arm and gripper are exposed together as one whole-body hardware
component.

## can-motor-control 0.0.8 feedback decoder

Version 0.0.8 can discard valid motor feedback when the low byte of the
reported position equals `0x33` or `0x55`. It mistakes that position byte for
a parameter response opcode, so an otherwise responsive gripper can fail
startup calibration with "did not receive fresh feedback before starting".

The local fix checks the parameter response's motor-ID header as well as
its opcode. Build and install it into this checkout's virtual environment:

```bash
bash scripts/install_can_motor_control_feedback_fix.sh
```

This builds the pinned v0.0.8 source from
[dimensionalOS/can-motor-control](https://github.com/dimensionalOS/can-motor-control)
with the patch saved under `dimos/hardware/whole_body/damiao/patches/`, runs
the Rust motor-control and codec tests, and installs the wheel into `.venv`.
Rerun it if the environment is rebuilt or the dependency is reinstalled.
The installer does not access the robot. Use `.venv/bin/dimos` to run the stack
after installing the fix.

## Safety

- Keep the workspace clear and the emergency stop reachable during first
  activation.
- Verify motor IDs and types against the physical unit before commanding large
  motions; an incorrect type scales velocity and torque incorrectly.
- Deactivation removes motor torque, so support the arm before stopping it.
