# OpenYAM spatial teleoperation

`openyam-spatial-sim` is an experimental bimanual workstation for WebXR hand
teleoperation. It uses sim2 devices through the control and sensor interfaces;
it does not open physical CAN devices. The viewer is served by the existing
WebXR HTTPS server at `/spatial`.

Both simulated grippers start fully open. The station overrides the active and
passive finger startup positions without modifying the shared robot artifact.
Native initialization and reset were checked with the exported YAM model;
the six spatial-model tests pass. Hand-control gestures are unchanged.

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

- Open your hand, then pinch index and thumb near the corresponding coloured
  gripper handle. Move and rotate that hand to move that arm independently.
- Release the pinch to hold. Re-grabbing captures new hand and robot references.
- While grabbed, bring the middle finger toward the thumb to close the gripper;
  separate them to open it. These gesture distances are prototype defaults.
- Tracking loss, stale observations, disconnection, or leaving the immersive
  session releases control. Open the hand before grabbing again.
- Pinching while pointing away from a handle selects a surface in the observed
  cloud and publishes `perception_focus`. Recognition needs a connected consumer.

Use Ctrl-C in the launching terminal to stop. Restart the blueprint after a
simulation reset: the device adapter rejects commands from a changed generation.

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

## What is in the scene

Two arm bases are 0.62 m apart. Each arm has an RGB wrist camera (320 × 240,
requested 15 Hz). A central RGB-D camera (480 × 360, requested 10 Hz) looks
forward and down at 45 degrees from a pole. This is an example arrangement,
not a calibration of the office station or a RealSense sensor-noise model.

The immersive scene contains the latest observed cloud, the known robot meshes,
measured gripper handles, and wrist images. The table and task objects enter
only through camera pixels and depth. The viewer does not consume `sim_truth`
or load the simulation scene. Occluded surfaces remain absent. Clouds replace
one another; this is not a persistent reconstructed map.

| Owner | Inputs | Outputs / responsibility |
| --- | --- | --- |
| `OpenYamSpatialSimulation` | Native servo targets | Physics and sim2 device observations |
| `OpenYamSpatialCameras` | Read-only sim2 sensor snapshots | Central RGB/depth/intrinsics/cloud/TF; two wrist RGB feeds |
| `OpenYamSpatialTeleop` | Cloud + timestamped camera TF; measured joints; tracked hands | Sensor-derived view; existing teleop streams; optional focus request |
| `OpenYamSpatialCoordinator` | Per-arm teleop streams | Existing per-arm IK tasks; one writer for both sim2 arm devices |

All three existing CameraDevices run sequentially in one dedicated sensor
worker. Physics and control run separately. Concurrent native camera workers
aborted in the macOS graphics driver during development; this composition
avoids that observed failure without changing sim2's renderer or worker system.
Each device keeps its acquisition clock and reports skipped acquisitions.

The known mesh model and the temporary kinematics-only URDF derive from the
same robot artifact, including its grasp site. Thus display, IK and simulated
joints use consistent frames. Robot poses always follow measured joint state,
not the requested hand target. The viewer accepts at most 12,000 observed
points, requires camera calibration at the acquisition timestamp, and rejects
control when feedback or cloud age exceeds 0.5 seconds. Input expires after
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
It also does not attach the collection recorder or ACT training loop yet; those
remain on the base OpenYAM workflow.

## Verification

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
