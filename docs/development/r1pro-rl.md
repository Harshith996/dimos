# R1 Pro simulation preparation

This baseline runs wrist target references through a pinned MuJoCo model, bounded arm position targets, RSL-RL PPO, checkpoint reload, and independent held-out evaluation. Its first curriculum fixes the torso and chassis. It establishes executable preparation for whole-body tracking. It does not establish controller quality, teleoperation performance, or hardware readiness.

The implementation lives in `dimos/robot/galaxea/r1pro/rl`. It uses the existing DimOS joint naming and official asset revision, and reuses the K1 project's installed MuJoCo, PyTorch and RSL-RL packages. It does not import the hardware connection module or start ROS. SEW is not a prerequisite. No vision, grasp learning, robot operation or online robot RL is included.

## Reproduce a bounded smoke

Use a dedicated Python 3.11 environment with `configs/r1pro_rl_requirements.txt`. A CPU PyTorch installation is sufficient. These pins record the local tested package versions; they are not a complete dependency lock. Set `R1PRO_ASSET_ROOT` to the official `R1Pro/urdf_r1pro_g1z_2026` directory at commit `2e5d31e1784481a34d178006c0d0e18e0a84a82a`. The loader verifies the URDF and mesh hashes and never downloads assets.

From the repository root, run:

```bash
R1PRO_RL_PYTHON=/path/to/isolated/bin/python \
R1PRO_ASSET_ROOT=/path/to/URDF/R1Pro/urdf_r1pro_g1z_2026 \
bash scripts/r1pro_rl_smoke.sh /path/to/new/output-directory
```

The output directory must not already exist. `R1PRO_RL_DEPS` optionally names a local dependency overlay. The local integration reused the K1 GPU Python with a separate overlay containing MuJoCo and its lightweight dependencies. The existing environments remain unchanged. The local `rl/pyproject.toml` gives this optional experiment its own test configuration. DimOS's existing parent collection rule excludes nested projects, so its default suite does not acquire undeclared RSL-RL or asset requirements. The script runs focused simulation tests without the repository's live-system pytest configuration, one PPO update on two sequential CPU environments, eight inference steps after checkpoint reload, held-out debug-hold evaluation with dropout and timeout cases, and held-out checkpoint evaluation. No larger training is exposed by this CLI. Its hard cap is two updates with at most four CPU environments.

The run writes tests and process logs, TensorBoard events, checkpoint and replay files, target manifests, and independent evaluation traces. Training and evaluation record configuration, model inputs, generated XML, targets and checkpoint hashes as applicable. Generated binaries and run outputs belong outside the source tree.

## Reproduce preparation checks without training

```bash
R1PRO_RL_PYTHON=/path/to/existing/bin/python \
R1PRO_RL_DEPS=/path/to/existing/dependency-overlay \
bash scripts/r1pro_rl_prepare.sh /path/to/new/preparation-output
```

This command runs focused tests, the per-link static import audit, eight frozen-target hold/IK evaluations, and a synthetic recording EOF watchdog check. It performs no policy updates. The report verifies identical target, model and constraint hashes between controllers. Normal cases must complete the requested short duration; injected dropout and stale cases must retain failure status. These assertions establish regression behavior, not a tracking-performance threshold.

The independent static checks compare source full inertia tensors, masses, centers of mass and joint transforms with MuJoCo, and verify collision masks, mesh associations, exclusions and actuator settings. Deliberately corrupted model imports fail the checks. Matching the CAD import does not establish that CAD matches the installed robot.

The small frozen suite stores realized knots, sampled targets and evidence before further tuning. It remains a regression suite. A final teleoperation test distribution still requires separately reviewed targets or recordings.

## Targets, observations and reward

The initial targets are small, deterministic continuous joint excursions converted through FK. Training and held-out scenarios use separate seed domains and different trajectory shapes. Sampled collision and limit checks establish only sampled kinematic admissibility. They do not prove dynamic feasibility or that these targets represent live teleoperation.

Targets carry timestamps, world-frame wrist positions and wxyz orientations, validity masks and provenance. Optional elbow and chest preferences remain interface fields. Missing orientation is masked out of observations, reward and evaluation. The local JSON recording loader preserves timestamps, masks, confidence, calibration identifiers, session provenance, and optional preferences. Its tests use synthetic fixtures. No human recording is available, and no AMASS or LAFAN1 data is downloaded or bundled. Pose targets are not robot action labels.

The actor sees measured joint and wrist state, the current reference and its masks, the previous bounded position command, and reference age. It receives no future reference, clip identifier or phase. PPO uses the current target that the actor observed. Target dropout, stale timestamps, joint bounds and meaningful collision checks are enforced independently of reward. Tracking error alone does not end an episode.

The replaceable smoke reward uses normalized position and shortest SO(3) orientation errors with provisional broad and tight kernels. Terms are logged individually. Command-rate and joint-velocity terms use physical units and currently have zero penalty weights. Limit-proximity and collision reward costs are deferred; hard constraints remain active independently. This baseline does not settle reward tuning or the final human-motion distribution. Elbow and chest preference rewards, richer regularization and controller identification require separate experiments.

## Independent evaluation

The evaluator has its own fixed small case manifest and does not use reward as an acceptance metric. It reports both wrists' position and orientation distributions, physical smoothness, bounded lag separately from unshifted error, collisions, joint limits, target faults, requested duration and completion. It logs raw targets and actions alongside limited joint commands. A failed episode remains failed for the requested duration.

The zero-rate hold controller is a debugging baseline. Comparing a one-update checkpoint against it establishes matching evaluation plumbing, not an RL improvement. The same-model differential IK baseline reuses existing DimOS numerical functions. Both controllers now receive identical frozen targets, update rates, limits and preprocessing. The full production Pink controller comparison remains open because its Pinocchio dependency is unavailable in this runtime. Step overshoot, settling, robust clearance, hardware latency, unreachable/conflicting command suites, and full dropout recovery are not established by this milestone. Draft 3 cm and 10 degree RMSE goals are not achieved results or p95 guarantees.

## Gate before larger training

1. Identify the actual robot variant and authorized SDK build. Resolve joint order, signs, units, reference frames, permitted targets, limits, tracking-speed behavior, command and feedback age, and robot-side watchdog ownership. Keep torso targets unavailable for deployment until the selected authorized configuration permits them. Do not bypass a vendor restriction.
2. Accept the simulation geometry and dynamics for the intended experiment. Check calibrated TCPs and tool masses, collision exclusions and convex approximation, actuator response, timing, and conservative bounds. Loading, positive inertias and a finite short rollout do not validate dynamics.
3. Freeze a small meaningful trajectory manifest, independent metrics and production controller comparison before reward tuning. Separate feasible tracking from deliberately unreachable/conflicting safety tests. Keep world-fixed and body-relative semantics explicit when the chassis is introduced.
4. Review checkpoint provenance, fault behavior and held-out results before increasing environment count, training length or scenario complexity. Then add the torso curriculum, and later a chassis velocity adapter with appropriate footprint evaluation. This code currently implements neither released torso nor mobile-base dynamics.

See the [model audit](r1pro-rl-model-audit.md), [control contract](r1pro-rl-control-contract.md), [target/evaluation notes](r1pro-rl-targets-evaluation.md), [controller baseline](r1pro-rl-controller-baseline.md), and [recorded stream schema](r1pro-rl-recorded-targets.md) for source evidence and remaining limitations.
