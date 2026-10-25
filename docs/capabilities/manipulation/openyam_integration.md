# OpenYAM integration

OpenYAM uses the generic Damiao whole-body adapter backed by
`can-motor-control`. The adapter owns one upstream robot containing the six-joint
arm and its gripper; motor topology is independent of the host operating system.

## CAN transport

The generic Damiao layer selects the native transport provided by
`can-motor-control>=0.0.8`:

| Host | Default | `runtime_config.bus_devices["openyam"]` override |
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

Select a Linux SocketCAN interface or a macOS adapter serial in a blueprint of your own with `openyam_hardware(bus_devices={"openyam": "can1"})`. The device lives in the adapter's `runtime_config`, not in `controlcoordinator.addresses`; see [Hardware configuration](/docs/usage/configuration.md#hardware-configuration).

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

### Quest controls

To set **forward**, release both grips, point either controller in your intended
operating direction, and click that controller's thumbstick. This sets one
horizontal heading for both arms; up stays up. The headset confirms the change.
Grip again to continue from the held robot poses. Recalibrate after changing your
operating direction or reconnecting the headset. Looking around does not change
the heading, and calibration is ignored while either grip is held.

### Record dual-arm demonstrations

`teleop-webxr-dual-openyam` includes `DualOpenYamCollection`. The blueprint owns
both USB cameras and publishes their images to Desktop. Desktop attaches to the
existing run; it does not open another arm connection or require a station setup.

Engage both arms and their grippers once after startup so the recorder has actual
accepted targets for every joint. In Desktop, enter the task, choose training or
evaluation, and click **Record demonstration**. Teleoperate, then **Save episode**
or **Discard**. In the headset, **A** starts/saves a training take and **Y** discards.
**B** still resets the arms to zero and discards an active take. Reset the objects
between takes while recording is idle.

Each take is stored under `$XDG_STATE_HOME/dimos/recordings/dual-openyam/<id>`
(default `~/.local/state/dimos/recordings/dual-openyam/<id>`):

- `recording.mcap`: both camera streams, 14-joint feedback, driver-accepted position
  updates and start/save/discard boundaries, with source timestamps.
- `schema.json`: camera shapes, joint ordering, action semantics and alignment.
- `metadata.json`: task, split, outcome, camera USB identities and verified counts.

Camera names are `camera1` and `camera2`, ordered by USB identity; these names do
not assert a left/right mount. With more than two USB cameras, explicitly set
`--dualopenyamcollection.camera-ids` to the two intended identities. Capture defaults
to 640×480 at 15 Hz. The module's `directory`, `width`, `height` and `fps` settings
are blueprint configuration fields.

Save drains the native recorder and verifies its finalized MCAP counts before
publishing the manifest. A feedback/camera outage discards the active take;
interrupted files without a manifest are excluded from the saved list. Held
targets retain their last accepted command; measured state is never substituted
for an action. Driver acceptance is not a firmware execution acknowledgement.

The dual-arm collection path is implemented. Dual-arm ACT training and policy
rollout are not yet connected to Desktop; the existing single-arm learning loop
has a different observation/action contract.

Each controller's grip engages its own arm: left grip controls the left arm,
right grip controls the right arm, and both can operate together. Releasing a
grip ends teleoperation for that arm without disengaging the other. Each index
trigger controls its corresponding gripper.

With both grips released, B requests a planned move of both arms to the zero
joint pose. This uses the existing manipulation planner and coordinator
trajectory task. Held teleoperation has higher priority than the reset move.

The Desktop Workcell continues to attach to the same coordinator and camera
feeds. Blueprint code changes take effect on the next blueprint start; reloading
the viewer does not reload the running robot's code.

## Hardware topology

| Motor | Send ID | Reply ID | Type |
|-------|---------|----------|------|
| joint1–3 | `0x01`–`0x03` | send ID + `0x10` | DM4340 |
| joint4–6 | `0x04`–`0x06` | send ID + `0x10` | DM4310 |
| gripper | `0x08` | `0x18` | DM4310 |

The bus uses classical CAN at 1 Mbit/s. Opening the gripper decreases motor
position. The arm and gripper are exposed together as one whole-body hardware
component.

## Safety

- Keep the workspace clear and the emergency stop reachable during first
  activation.
- Verify motor IDs and types against the physical unit before commanding large
  motions; an incorrect type scales velocity and torque incorrectly.
- Deactivation removes motor torque, so support the arm before stopping it.
