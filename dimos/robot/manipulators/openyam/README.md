# YAM in sim2

`simulation.py` consumes an externally supplied, resolved YAM artifact containing
`robot.xml`, its `assets/`, and a `contract.json` with schema
`humanoid-control-lab.yam-simulated.v1`. It verifies the manifest hashes, referenced
resources, joint order, position servos, limits and coupled finger initialization.
The exporter revisions are provenance; the original training checkout is not a
runtime dependency. Keep the artifact immutable while using it.

Use a normal DimOS installation with its MuJoCo and Zenoh dependencies. No mjlab,
Torch, CUDA, hardware SDK or CAN interface is needed. Set `YAM_ARTIFACT` to the
resolved artifact directory, then run
`python -m dimos.robot.manipulators.openyam.demo_sim2 --artifact "$YAM_ARTIFACT" --steps 100`.
Add `--contract-sha256` with the independently recorded manifest hash to pin the
artifact, and `--report` with a new JSON output path to retain the result. The demo
owns a unique loopback world, applies 100 control ticks, checks reset and command
freshness, and closes its client and world. Run it in a separate process from any
existing stack because its transport settings are process scoped.

For integration, `load_yam_artifact(Path(...)).robot_instance()` supplies the
`RobotInstance` for a caller-owned `WorldConfig`. It uses the model's native
position servos without adding another PD controller. Six arm coordinates are
radians; the public gripper coordinate 0..1 maps to left-finger 0..0.0355 metres.
This is a simulation convention, not a hardware calibration. The passive right
finger is initialized in native coordinates and restored by ordinary world reset.

`YamSimAdapter.write_model_targets(targets, observed=frame)` converts native targets
and requires the actual `StateFrame` used by the controller. It does not reread
feedback and make an old policy result appear fresh. Policy observations,
action decoding, control cadence and checkpoint provenance remain the caller's
responsibility; this adapter does not load or train a policy.

`test_simulation.py` uses a miniature generated MJCF and includes an actual local
`SimRobotClient` round trip. Run it with the repository's normal pytest setup.
The bounded demo validates motor transport and reset behavior; it does not claim
trajectory parity, a successful manipulation task or physical-robot readiness.
