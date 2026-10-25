# OpenYAM spatial teleoperation

`openyam-spatial-sim` and `openyam-spatial` use the same WebXR hand interface
and registered RGB-D observation renderer. The first uses sim2 devices; the
second composes the existing physical dual-YAM coordinator, wrist-camera
collection module, RealSense and a calibrated camera mount. The viewer is
served by the existing WebXR HTTPS server at `/spatial`.

Both simulated grippers start fully open. The station overrides the active and
passive finger startup positions without modifying the shared robot artifact.
Native initialization and reset were checked with the exported YAM model;
zero means closed, and positive finger displacement opens the jaws. Full opening
is a measured 71 mm mesh gap. Physical connections retain measured state.

## Run

Install the repository dependencies and provide the validated `yam-sim2-model`
artifact (`contract.json`, `robot.xml`, and `assets/`). Set `DIMOS_YAM_ARTIFACT`
to that directory. The default is `~/.local/share/dimos/yam-sim2-model`.
The existing artifact loader verifies the contract and model on startup.

```bash
uv run dimos run openyam-spatial-sim --server-port 8445
```

Open `https://<this-computer-LAN-address>:8445/spatial` in the headset on the
same network. Use the existing WebXR server's HTTPS certificate setup; the
browser must trust the connection before WebXR is available. The address must
be reachable from the headset. Port 8445 here keeps the prototype apart from
a physical station's usual 8443 server. A desktop browser can open the same
page to preview observations without acquiring the control connection.

Select **Enter immersive view** and allow hand tracking. The workspace is
placed at metre scale ahead of the current head direction. The same transform
maps both displayed geometry and hand commands. Exit and re-enter to place it
ahead again; the desktop **Place workspace ahead** button also requests this.

In immersive view, pinch the rail beneath the workspace to move the entire view.
Add the other hand to resize and turn it, keeping it upright. Release either
hand to continue dragging with the remaining hand; release both to leave it
in place. Scale is bounded to 25–200%. The rail shows the current percentage,
**Actual size** restores 100% around the rail, and **Place ahead** recenters
without changing the chosen scale. These controls remain available with stale
robot observations because they only arrange the local view.

Arrangement releases both robot gestures. Open your hand and make a fresh tool
grab afterward to resume operation. Calibration and robot coordinates stay
unchanged; all displayed observations and controls share the same transform.
At 200%, a 2 cm hand translation becomes a 1 cm robot translation. Hand
orientation uses only the transform's rotation, independent of display scale.
Tracking loss freezes the view and requires release before re-grabbing the rail.

- Open your hand, then pinch index and thumb near the corresponding coloured
  gripper handle. Move and rotate that hand to move that arm independently.
- Release the pinch to hold. Re-grabbing captures new hand and robot references.
- Pinch the small bead on the tool's opening rail, then slide toward OPEN or
  CLOSE. Opening starts at measured feedback, so grabbing the bead does not
  jump the jaws. This action holds the arm; moving the arm leaves opening alone.
- The solid bead and percentage show measured opening. A white outline shows
  the requested opening while dragging. Release holds the last command.
- Each pinch captures one action until release. Crossing another handle while
  pinched cannot switch actions. Middle-finger position has no control effect.
- Tracking loss, stale observations, disconnection, or leaving the immersive
  session releases control. Open the hand before grabbing again.
- Point at **Hold control** on the headset panel and pinch to pause both hands.
  **Resume** enables fresh grabs; it does not resume an earlier movement.
- Pinching while pointing away from a handle selects a surface in the observed
  cloud and publishes `perception_focus`. Recognition needs a connected consumer.

Use Ctrl-C in the launching terminal to stop. Restart the blueprint after a
simulation reset: the device adapter rejects commands from a changed generation.

## Cameras and episodes

The spatial panel shows labeled left-wrist, workspace and right-wrist images,
per-arm movement/opening/hold status, and recording state. Stale camera frames
are replaced by an unavailable indicator. The timer and saved/discarded counters
come from the blueprint's recorder.

Set the task before entering the immersive session. Engage each arm and its
opening control once after startup, establishing the actual accepted command
history. The panel explains this requirement until recording becomes ready.
Point at **Record** and pinch; demonstrate the task, then **Save** or **Discard**.
An episode-button pinch cannot become an arm grab without a release. Save shows
SAVING until the native recorder has finalized the take; failures stay visible.
The same recorder is available through Desktop's existing episode controls.

`DualOpenYamCollection` owns MCAP recording in a dedicated worker. In sim it
subscribes to the two existing wrist streams; on hardware it retains ownership
of the two configured cameras. Both record fourteen measured joints, accepted
joint commands, task/status and two wrist images. The central image remains an
operator view. Data goes to `dualopenyamcollection.directory`, defaulting to
`~/.local/state/dimos/recordings/dual-openyam`. The viewer calls the shared
`EpisodeCollectionSpec`; it does not create another recorder or exporter.
Dual-arm ACT training and rollout are still a separate integration step.

## DimOS Desktop

Open the installed Workcell app in DimOS Desktop. Its worker must use this
DimOS checkout and follow running blueprints. It attaches automatically to
`openyam-spatial-sim`, shows the same observed scene inside Desktop, subscribes
to all three RGB camera streams and exposes the coordinator's joint controls.
The preview does not acquire a teleoperation connection. Open the advertised
HTTPS headset link to enter the immersive view.

Attachment uses the run registry and module/stream introspection, not a
blueprint-name list. `ControlCoordinator.describe_controller()` advertises the
run identity, joint limits and command owners. `get_web_views()` advertises
optional HTTP views; the app proxies embeddable views through its existing
backend to the local server. The embedded scene uses relative asset/API URLs.
The blueprint keeps ownership of its devices and control loop. Missing limits
or command owners disable manual controls; missing recorder RPCs hide recording.
If several runs exist, select one. Replacement runs reconnect automatically
when following the sole run, and old run commands and callbacks are rejected.

## Physical station

Run the physical blueprint with a station JSON file through the normal CLI:

```bash
uv run dimos run openyam-spatial --config /path/to/station.json
```

The file uses ordinary module settings; no second viewer or robot-control
process is required. Supply these deployment values:

| Module / field | Meaning |
| --- | --- |
| `ControlCoordinator.left_can_port`, `right_can_port` | Distinct, explicitly identified left and right CAN interfaces |
| `dualopenyamcollection.camera_ids` | Two stable USB camera IDs, ordered left then right; the recorder remains their sole owner |
| `workspace_camera.serial_number` | The central RealSense serial number |
| `workspace_mount.frames` | Calibrated fixed edges, each `[child, parent, [x,y,z], [roll,pitch,yaw]]`, in metres/radians; provide the `world` → `camera_link` mount |
| `openyamspatialteleop.depth_unit_m` | Metres per raw uint16 depth unit, from the device's depth-units setting; default 0.001. Float depth is already metres |
| `openyamspatialteleop.server_port` | Existing WebXR HTTPS port setting |

RealSense supplies the factory-calibrated `camera_link` → optical-frame subtree.
The mount publisher supplies the measured station edge at 20 Hz. An absent
mount is an error; the viewer never substitutes an identity transform. Registered
color/depth use one optical frame and resolution, with timestamps within 30 ms.
The default coordinator still supports its existing mock mode if neither CAN
port is set; that does not connect physical arms.

The default physical model is `dual_openyam_abc_box_v2`. Verify its arm-base
placements against the actual rig. The simulation artifact has different grasp
frames and joint limits and is not substituted for this model. For a calibrated
station URDF, compose `build_openyam_spatial(model=...)` with a `RobotModelConfig`;
that exact model is passed to IK, hardware limits, manipulation and display.
Do not independently override only the viewer's model.

The existing physical URDF fixes its fingertip visuals and supplies no moving
finger kinematics. Those fixed visuals are omitted rather than animated with
guessed axes. The measured opening gauge is live; observed fingertips come from
RGB-D. Simulated finger meshes follow measured finger state. Physical mesh
animation needs a calibrated moving-finger description.

Physical deployment is wired and configuration/model checks pass locally.
Real camera acquisition, calibration and arm operation still require station
acceptance.

## What is in the scene

Two arm bases are 0.62 m apart. Each arm has an RGB wrist camera (320 × 240,
requested 15 Hz). A central RGB-D camera (480 × 360, requested 10 Hz) looks
forward and down at 45 degrees from a pole. This is an example arrangement,
not a calibration of the office station or a RealSense sensor-noise model.

The immersive scene contains the latest observed depth surface, the known robot meshes,
measured gripper handles, three camera panels and episode controls. The table and task objects enter
only through camera pixels and depth. The viewer does not consume `sim_truth`
or load the simulation scene. Occluded surfaces remain absent. Each observation
replaces the previous one; this is not a persistent reconstructed map.

| Owner | Inputs | Outputs / responsibility |
| --- | --- | --- |
| `OpenYamSpatialSimulation` | Native servo targets | Physics and sim2 device observations |
| `OpenYamSpatialCameras` | Read-only sim2 sensor snapshots | Central RGB/depth/intrinsics/cloud/TF; two wrist RGB feeds |
| `OpenYamSpatialTeleop` | Registered RGB/depth/intrinsics + timestamped TF; measured joints; tracked hands | Sensor-derived surface; independent pose/opening actions through existing teleop streams; optional focus request |
| `OpenYamSpatialCoordinator` | Per-arm teleop streams | Existing per-arm IK tasks; one writer for both sim2 arm devices |
| `DualOpenYamCollection` | Measured joints, accepted commands, two wrist images | Existing native MCAP recorder and episode RPCs shared with Desktop |

All three existing CameraDevices run sequentially in one dedicated sensor
worker. Physics and control run separately. Concurrent native camera workers
aborted in the macOS graphics driver during development; this composition
avoids that observed failure without changing sim2's renderer or worker system.
Each device keeps its acquisition clock and reports skipped acquisitions.

The simulated mesh model and temporary kinematics-only URDF derive from the
same robot artifact, including its grasp site. Thus display, IK and simulated
joints use consistent frames. Robot poses always follow measured joint state,
not the requested hand target. The renderer retains an organized depth grid
bounded by `max_pixels` (60,000 by default) and textures it with the original
RGB resolution. It rectifies supported distortion, preserves invalid samples,
and rejects triangles with edges longer than `max_edge_m` (35 mm). It does not
smooth across depth holes or invent hidden surfaces. The transport carries
millimetre uint16 depth and JPEG in a versioned binary frame; GPU buffers are
reused, and each client requests only acquisitions newer than its last frame.

The viewer requires camera calibration at the acquisition timestamp and rejects
control when feedback or surface age exceeds 0.5 seconds. Input expires after
0.3 seconds. It does not accumulate queued hand commands.

This station adds **ideal gravity compensation on robot links**, matching the
physical driver's enabled gravity-compensation mode at an idealized level.
The artifact's native servo gains remain unchanged; workspace objects remain
under gravity. The imported RL artifact alone does not supply this compensation:
without it, measured-pose re-grabs cause repeated gravity sag. This is an
explicit interactive device-model assumption, not calibrated hardware fidelity
or the unmodified RL policy deployment. The source artifact is not edited.

## Perception and manipulation

RGB-D projection plus timestamped TF already supplies geometric perception.
Object recognition is a separate stage. Existing
`ObjectSceneRegistrationModule` consumes RGB, depth, intrinsics and TF and emits
`objects`; configure its target frame as `world` before wiring that stream to
this viewer. The viewer displays those world-frame boxes when a detector is
connected. Its optional `perception_focus: PointStamped` output identifies a
selected observed surface; adapting that point into a detector request is a
separate consumer, not implemented by the default blueprint.

`ManipulationModule` can use detected objects and suitably filtered scene
geometry for planning. The raw cloud includes the robot itself, so this
prototype does not feed it directly into a collision planner. Teleoperation
currently uses IK with joint/tracking limits, not collision-avoiding planning.
The default blueprint loads no semantic model and says so in the interface.
The spatial blueprint attaches the existing collection recorder. It does not
yet supply bimanual ACT training or policy rollout.

## Verification

Workspace arrangement, later 2026-10-25: sixteen JavaScript tests cover existing
pinch/surface behavior and eight new workspace cases, including one/two-hand
transitions, scale limits, coincident/vertical hands and tracking loss. A private
browser harness runs the real hand-input path against live sim observations,
using synthetic tracked-hand samples and a captured command socket. It checks
same-frame release of an active arm when the other hand grabs the rail, empty
arm commands throughout arrangement, fresh-grab gating, 2:1 translation mapping,
unit orientation quaternions, Actual size/Place ahead callbacks and local
arrangement during stale observations. Miniature rail and tool targets remain
distinct. Browser errors and actual control WebSockets: zero. No robot is
commanded by this browser test. Evidence and the reproducible harness are at
`/Users/pimvandenbosch/Desktop/openyam-learning-artifacts/apple-vision-pro-workspace-20261025/`.
Actual Vision Pro tracking and gesture comfort still require operator testing.

Spatial UX integration and gripper correction, 2026-10-25:

- Corrected the erroneous `1-opening` sim conversion in commands, feedback and
  startup. Actual asset jaw meshes meet at zero and separate with positive
  displacement; body origins alone had led to the wrong interpretation.
- Native HTTPS/WebSocket input moved only the requested arm and exercised the
  opening control separately. Measured opening/gap changed from approximately
  1.0 / 71 mm to 0.503 / 35.7 mm, then back to 0.997 / 70.8 mm. The other
  gripper stayed open. Gap measurement uses the actual mesh vertices and live
  display transforms, independently of the percentage label.
- The five-module sim recorded and finalized a two-camera MCAP episode through
  the same episode RPCs used by Desktop. A subsequent discard did not add a
  saved episode. Native recorder tests also read back images and fourteen-joint
  accepted targets and exercise failed finalization.
- Focused validation: 39 Python tests, eight JavaScript tests, five blueprint
  registry checks, and strict mypy on seven production files. Private browser
  checks cover live cameras and a view-only preview with no control WebSocket.
- The installed Desktop shell displays all three live images and discovers the
  recorder. A browser harness exercises the actual spatial UI's ray/button hit,
  Record/Save/Discard requests and Hold/Resume callbacks. It supplies connection
  state for those UI checks; it does not emulate actual Vision Pro hand tracking.
  Both browser runs report zero JavaScript errors. Native MCAP readback finds
  state, accepted actions, both camera channels and two episode boundaries.

Evidence and reproducible scripts are outside the engine repository at
`/Users/pimvandenbosch/Desktop/openyam-learning-artifacts/apple-vision-pro-ux-20261025/`.
Actual headset ergonomics and physical deployment remain unverified by these
checks. Earlier opening percentages below used the inverted convention and
must not be treated as proof of actual jaw direction.

Scene clarity and tool controls verified in the native simulation, 2026-10-25:

| Check | Observed result |
| --- | --- |
| One central RGB-D frame | 23,004 valid depth samples, 43,148 triangles, 106,314 bytes including full-resolution RGB JPEG |
| Left opening drag | Measured opening 0.994 → 0.685 during the sampled closing interval; arm displacement below 0.001 mm; right arm stationary |
| Reopen | Opening increased toward the fully-open target; arm displacement below 0.001 mm; right arm stationary |
| Left pose gesture | 21.1 mm maximum measured displacement; right arm stationary; measured opening stayed within 0.000001 |
| Surface age during opening exercise | Maximum below 158 ms across the sampled commands |
| Physical model inspection | Display grasp positions agreed with physical-model Pinocchio FK across 16 poses, maximum numerical difference below 1e-14 m |
| Browser and installed DimOS Desktop | Live textured surface and tool controls; zero JavaScript errors and zero preview control WebSockets |

These are short local exercises, not a throughput benchmark or physical-arm/
Vision Pro acceptance. Depth tests cover float metres, non-default uint16 units,
invalid samples, timestamp/frame mismatch, distortion and missing transforms.
Gesture/control tests cover no-jump opening capture, action ownership, independent
hands, release, missing tracking, stale observations and replayed messages.
The existing physical blueprint's mock tests also retain independent grip behavior.
The old measurements below describe the earlier point-cloud/gesture version.

Python tests cover independent grabs, tracking loss and release-before-regrab,
stale/replayed input, observed-surface focus, timestamped cloud transforms,
arm/gripper device ordering, reset rejection, and agreement between robot FK
and the generated IK frames. Node tests exercise pinch hysteresis and tracking
loss. A private headless browser renders captured native sensor observations
and checks that desktop preview opens no control socket.

The full native blueprint is exercised separately through HTTPS/WebSocket into
sim2. See the dated result below. Actual Vision Pro hand tracking, gesture
comfort, headset rendering performance and LAN/TLS setup still require a
headset test; a browser preview does not establish those results.

Verified on macOS, 2026-10-25, with the existing `yam-sim2-model-v1` artifact:

| Native HTTPS/WebSocket exercise | Observed result |
| --- | --- |
| Left hand requests 25 mm forward + 25 mm up | 35.09 mm measured gripper displacement; right drift 0 mm |
| Release and re-grab with a different hand orientation | 0.009 mm measured displacement |
| Right hand requests the same translation | 35.10 mm measured displacement; left drift below 0.001 mm |
| Right hand closes gripper | Measured normalized opening 0.991 → 0.230 |
| Observation age across 374 command samples | Cloud p95 176 ms; maximum 290 ms |
| Camera endpoints | Both wrist JPEGs present; approximately 10,300 observed cloud points |

The run built all four real modules, used the actual native robot asset,
rendered all three cameras, streamed commands through `/ws`, read back measured
poses and joint opening, and shut down its workers. No physical arm was
connected by the test. This short exercise is not a latency benchmark or
hardware-transfer acceptance. Component validation: 74 Python tests, two Node
gesture tests, strict mypy on the four implementation files, and a headless
preview with zero JavaScript errors and zero desktop control connections.

Desktop attachment verified on macOS, 2026-10-25: the installed Workcell app
discovers the four-module native run, receives fourteen measured joints and
three RGB streams, and embeds this renderer through its existing backend.
A headless browser opens the actual Desktop shell and nested preview, reports
zero JavaScript errors and no control WebSocket, and uses a joint increment
button through the coordinator. A 0.025 rad request reaches 0.0241 rad during
the sampled settling interval; the other arm remains stationary. Stale run
preview URLs return HTTP 410. The existing Desktop service stays running.
Component checks: 179 Python tests, 17 app tests, strict Python/TypeScript
checks, Desktop API-contract validation and the packaged macOS Nix build.
This does not verify headset ergonomics or add recording to this blueprint.
