# R1 Pro RL model audit

The baseline uses the existing DimOS planning asset, the vendor's 2026 G1Z R1 Pro description at [URDF commit 2e5d31e1784481a34d178006c0d0e18e0a84a82a](https://github.com/userguide-galaxea/URDF/tree/2e5d31e1784481a34d178006c0d0e18e0a84a82a/R1Pro/urdf_r1pro_g1z_2026). This identifies the selected simulation asset. It does not establish the installed robot's build, gripper, camera arrangement, firmware, or SDK version.

The converter in `dimos/robot/galaxea/r1pro/rl/model.py` verifies all input hashes against `configs/r1pro_rl_assets.json` before it builds MJCF. It reads the existing local asset cache by default. Call `load_model(path)` or set `R1PRO_ASSET_ROOT` to the `urdf_r1pro_g1z_2026` package directory. It does not download assets, initialize a robot SDK, or connect to hardware. To prepare another machine, obtain the repository at the exact commit above and supply that package directory. No large meshes are duplicated in this change.

## Mapping and limits

`configs/r1pro_rl_model_audit.json` records every joint's parent, child, axis, transform and raw URDF limit, and every link's mass, inertial transform, full inertia tensor and collision mesh. The source has 43 links and 42 joints. Four torso joints and two seven-joint arms describe the articulated upper body. Those 18 coordinates describe model structure, not permission or support for an 18-joint hardware action interface.

The debugging model fixes the base, all four torso joints, wheel and steering joints, and gripper joints at their URDF zero. Its only movable joints are `left_arm_joint1` through `left_arm_joint7`, followed by the corresponding right arm joints. `joint_addresses(model)` returns their qpos and qvel indices. The right second joint has the opposite limit interval to the left second joint; callers must use the per-joint limits rather than one shared limit vector.

The URDF gives arm effort limits of 55, 55, 25, 25, 18, 18 and 18 N m, and velocity limits of 7.1209, 7.1209, 8.3776, 8.3776, 10.4720, 10.4720 and 10.4720 rad/s for each arm. These are source metadata, not recommended training speeds or verified hardware limits. Position actuators preserve joint position bounds and use the URDF effort caps. MuJoCo does not enforce URDF velocity metadata as a hard speed constraint; the environment must implement its own more conservative action rate bound.

`left_wrist` and `right_wrist` sites sit at the corresponding `left_gripper_link` and `right_gripper_link` origins, matching the existing planner's tip links. They do not represent a measured tool center or a fingertip. The vendor transform from arm link 7 to each gripper origin is `[-0.0295, 0, -0.1637]` m in the link 7 frame. Tool calibration and any teleoperation frame conversion remain integration requirements.

## Geometry and dynamics

The MuJoCo import preserves vendor inertials, all links, and both visual and collision meshes. The fixed model has 44 MuJoCo bodies including the world, 86 geometries, and total imported body mass 92.83112138 kg. All 43 source inertia matrices have positive eigenvalues and satisfy the principal-moment triangle inequality. This checks mathematical consistency, not agreement with physical measurements.

MuJoCo uses convex mesh collision here. It does not reproduce arbitrary concave STL geometry exactly. The converter copies the existing DimOS planning collision exclusions for structural mesh overlaps and gripper linkages. It keeps visual geometry non-colliding. At the zero pose, `mj_forward` reports five very shallow contacts at the arm-base interfaces, with penetration less than 0.3 micrometers. `collision_pairs` ignores contacts shallower than an explicit 10 micrometer numerical tolerance. This tolerance is not a real-world clearance margin. No meaningful nonexcluded penetration appears at this pose under that check. Targets still need collision checks throughout their trajectories.

The debugging dynamics use a fixed base and torso, 2 ms timestep, implicitfast integration, joint damping 2, and provisional position actuator gains kp 80 and kv 8. The gravity setting is MuJoCo's default. These parameters are not fitted to the SDK's controller or a physical robot. The full imported mass includes fixed structure, so total mass alone says little about arm tracking fidelity. Wheels are welded and there is no mobile-base contact model, floor model or velocity adapter in this first curriculum.

## Checks and integration gate

`test_model.py` checks the 14 movable coordinates and actuators, ordered mappings, limit transfer, literal zero-pose wrist positions, imported mass and positive inertias. It also steps two identical simulations for 100 steps, checks finite states, and compares the resulting states exactly. The short run is 0.2 s of simulation. It is a smoke check, not a dynamics test or a tracking evaluation.

The direct import and stepping smoke ran against MuJoCo 3.14.0 in the existing local Wuji simulation runtime. Both pytest tests passed with MuJoCo 3.10.0 in an existing DimOS runtime, using `python -m pytest dimos/robot/galaxea/r1pro/rl/test_model.py -m mujoco -q -o addopts=''`. The integration owner runs the tests again in the baseline's own pinned runtime. A larger training run must wait for agreement on the installed hardware variant and permitted control mode, calibrated target frames, collision-model acceptance, conservative operational bounds, and validation of the provisional actuator response. Torso and mobile-base curricula require explicit model and adapter work after that gate. There is no hardware clearance or online robot RL authorization in these results.
