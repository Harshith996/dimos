# WebXR Teleop

Spatial teleoperation through browser WebXR input sources. Supports tracked
controllers and hands on compatible headsets, including Meta Quest and PICO.

## Architecture

```
WebXR Browser  ──WebSocket──→  Embedded HTTPS Server  ──→  ArmTeleopModule
(poses + Joy)                    (port 8443)                  (absolute PoseStamped)
                                                                  │ left/right
                                                                  ▼
                                                     TeleopControlCoordinator
                                                                  │ by task name
                                                                  ▼
                                                        TeleopIKTask
                                                        (relative targets + Pink)
```

## Running

```bash
dimos run teleop-webxr-rerun    # WebXR teleop + Rerun viz
dimos run teleop-webxr-xarm7   # XArm7
dimos run teleop-webxr-hand-xarm7  # XArm7 hand tracking; pinch to toggle
dimos run teleop-webxr-piper   # Piper
dimos run teleop-webxr-a1z     # A1Z with mock hardware
dimos run teleop-webxr-dual    # Mixed XArm6 + Piper, one task per arm
dimos run teleop-webxr-openarm # OpenArm, bimanual IK + planner/Viser + mock hardware
dimos run teleop-webxr-dual-openyam # Dual OpenYAM + egocentric and wrist camera panels
```

Select a CAN interface explicitly to control real A1Z hardware:

```bash
dimos --can-port a1zcan run teleop-webxr-a1z
```

Open `https://<host-ip>:8443/teleop` in a WebXR-capable headset browser. Accept
the certificate, then tap Connect.

To watch the live cameras from an ordinary browser, open
`https://<host-ip>:8443/cameras` and accept the certificate. For example, use
the robot computer's Tailscale IP when viewing remotely. This page receives
the same JPEG camera frames as the headset but has its own view-only WebSocket;
it does not take the headset's control connection or send robot commands. The
camera page is available to anyone who can reach the WebXR server port.

For hand teleop, remove the controllers. Pinch the thumb and index finger on
the selected hand to engage it, move the wrist to control the arm, then pinch
again to disengage. Pinch the thumb and middle finger to close the gripper;
release it to open the gripper. Hand tracking must be enabled in the headset
browser.

`teleop-webxr-openarm` is safe by default: it always uses the in-memory
`mock_whole_body` adapter, regardless of the global simulation setting. It does
not select physical OpenArm hardware implicitly. The mock and bimanual model
start at the canonical all-zero pose. Since that pose places both joint-4
coordinates at their lower limits, the OpenArm planner and teleoperation task
share a Pink joint-limit posture margin that supplies a deterministic inward
direction without changing the measured seed. No random retry runs in the
control loop.

Specify both CAN interfaces to select real OpenArm hardware. Supplying only one
is rejected:

```bash
dimos run teleop-webxr-openarm --left-can-port can1 --right-can-port can0
```

The blueprint also includes `ManipulationModule` with the same bimanual model
and Viser visualization. Its coordinator has a joint-trajectory task over both
arms at priority 10. Manual arm and gripper tasks run at priority 20, so
manual takeover aborts any active planner or policy trajectory.

## Arm task bindings

Arm teleoperation uses one `TeleopIKTask` configured with one or two hand
bindings. Each binding names the controller (`left` or `right`), a frame in the
task's `RobotModelConfig`. The task's top-level `joint_names` explicitly select
the joints Pink may update. Gripper triggers publish normalized per-hand streams
to dedicated gripper tasks; gripper joints are not owned by the IK task.

Single-arm and mixed-arm setups use one binding per task. A bimanual robot such
as OpenArm uses one task, two bindings, and one bimanual model, so Pink solves
both frame targets in one control tick.

For a two-binding task, both controller grips must be held. Engagement captures
both controller and robot references together. Releasing either grip,
receiving stale input from either controller, preemption, or E-stop clears the
entire session; both hands must engage again before commands resume.

For controller-based arm teleoperation, the middle-finger grip is the deadman:
hold the relevant grip to engage and release it to disengage. Face buttons are
available for application lifecycle controls; the OpenYAM learning rollout
uses **A** to toggle policy execution. The index-finger trigger remains the
analog gripper command and is forwarded only while that hand's classified grip
bit is held. Arm engagement and manual gripper gating therefore use the same
float-to-digital conversion from the Quest input.

`teleop_buttons` publishes raw button levels. `button_pressed` and
`button_released` publish digital-only edges after 50 ms of stable input;
disconnecting the control client releases held buttons immediately. Lifecycle
consumers should subscribe to the edge streams instead of detecting edges from
raw levels independently.

## Headset camera panels

Video modules push camera frames to the headset over the same `/ws` socket:
one binary message per frame, a leading byte indexing the module's
`camera_panels` followed by the JPEG. `/teleop/config` lists those panels with
a layout slot, and the client floats them in passthrough, placed once per
session relative to the operator's starting position and heading:

| Slot | Camera | Placement |
|------|--------|-----------|
| `center` | scene / egocentric | large, above the workspace |
| `left`, `right` | wrist | smaller, beside that arm, turned toward the operator |

Panel borders show state: a wrist panel turns green while that controller's
grip is held, the center panel turns red while an episode is recording, and a
panel goes gray when its camera has not delivered a frame for two seconds. A
panel without a camera shows a dark blank.

A compact HUD card sits below the center panel whenever a collection stack
publishes episode status or the module is configured with a `hud_hint` (one
line of operator guidance such as the button map). It shows the recording
state with a blinking `REC` timer, the hint, the task label, take counters,
and dataset export progress from `export_status`.

`VideoArmTeleopModule` and `Go2TeleopModule` fill the center panel from
`color_image`. `DualArmVideoTeleopModule` adds `egocentric_image`,
`left_wrist_image`, and `right_wrist_image`; `teleop-webxr-dual-openyam` feeds
them from three `TeleopCamera` webcams picked per instance:

```bash
dimos run teleop-webxr-dual-openyam \
  --left-can-port can0 --right-can-port can1 \
  --egocentric-camera.device /dev/v4l/by-id/<scene-camera> \
  --left-wrist-camera.device /dev/v4l/by-id/<left-wrist-camera> \
  --right-wrist-camera.device /dev/v4l/by-id/<right-wrist-camera>
```

`device` also takes an OpenCV index. `--video-max-width` and `--video-max-fps`
on the teleop module bound the headset uplink; see `DualArmVideoTeleopConfig`
for defaults.

## Subclassing

| Method | Purpose |
|--------|---------|
| `_handle_engage()` | Customize engage/disengage logic |
| `_should_publish()` | Add conditions for publishing |
| `_get_output_pose()` | Customize pose computation (ArmTeleop publishes absolute poses) |
| `_publish_msg()` | Change output format |

`self._lock` is already held — don't acquire it in overrides.

## Joy Message Format

**Axes**: thumbstick X, thumbstick Y, trigger (analog), grip (analog)

**Buttons**: trigger, grip, touchpad, thumbstick, X/A, Y/B, optional menu. WebXR
omits a platform-reserved menu button on devices such as PICO controllers.

## Body Tracking Messages

The WebSocket carries two frame formats. Controller poses and joystick state use
binary LCM messages. When body tracking is enabled, every sampled frame includes
a JSON body-tracking heartbeat. A `null` joint map means the body source is
unavailable; an empty map means the source resolved no joints for that frame.
Body joints are captured in the same `local-floor` space as the controllers so
the two can be combined.

## Arm posture from body tracking

Hand-pose teleop alone has no notion of the operator's arm: the IK picks any
joint configuration that reaches the hand target. With the WebXR body-tracking
module (`body_tracking_mode="optional"` on the teleop module; Meta Quest
Browser 32+ ships it as an experimental feature), `ArmTeleopModule` also
publishes `left_arm_posture` and `right_arm_posture`: the operator's wrist,
elbow, and shoulder positions in the robot frame, taken from the
`<side>-hand-wrist`, `<side>-arm-lower`, and `<side>-arm-upper` body joints.

A `TeleopIKTask` binding with an `elbow_frame` consumes them. At engagement the
task measures the robot's elbow-to-hand distance; every tick it scales the
operator's wrist-to-elbow direction by that distance and attaches it to the
hand target as a position-only Pink task at `hint_position_cost`, below the
hand pose costs. The robot elbow therefore follows the operator's forearm
direction while the hand target keeps priority. When body tracking drops out
the hint's weight falls to zero, and hand-only teleop continues unchanged.
`teleop-webxr-dual-openyam` binds `left_link3` and `right_link3`, the OpenYAM
elbow joints.

The PICO demo requires body tracking. Enable standard DimOS debug logging to
inspect incoming snapshots:

```bash
DIMOS_LOG_LEVEL=DEBUG uv run dimos run demo-pico-body-tracking
```

The WebXR module logs the first resolved body pose at INFO. At DEBUG, it reports
the received snapshot rate, availability, reference space, joint count, and
joint positions every five seconds of incoming messages. Timing starts with the
first snapshot, excluding headset setup time. Null and empty snapshots are valid
and do not generate warnings; malformed messages do. Reports stop when messages
stop arriving, so these diagnostics do not detect a disconnected or silent client.

## File Structure

```
webxr/
├── module.py             # Base module
├── extensions.py         # ArmTeleop, TwistTeleop, video panel modules
├── camera.py             # TeleopCamera: webcam for one headset panel
├── controller_types.py   # WebXRControllerState, Buttons
├── blueprints.py
└── web/static/           # WebXR client (index.html, teleop.js)
```
