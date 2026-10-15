# R1 Pro differential IK comparison baseline

`dimos.robot.galaxea.r1pro.rl.controllers.DifferentialIKController` provides a small deterministic controller for comparison against the PPO plumbing smoke. It uses the same pinned MuJoCo model, current wrist targets, pose masks, environment command adapter, joint margins and watchdogs. It has no hardware interface.

Instantiate `DifferentialIKController(env, gain=2.0, damping=0.05)` and call it with the current NumPy observation. The returned 14-element action has the same normalized arm target-rate meaning as the PPO action. The observation must follow `env.set_target(current_frame)`, so the controller and actor see the same current target. The controller reads `env.current_target` and the actual MuJoCo state. It never steps physics or mutates the current joint state.

The controller calls the existing `compute_error_twist` and `damped_pseudoinverse` functions from `dimos/manipulation/planning/utils/kinematics_utils.py`. MuJoCo `mj_jacSite` supplies the actual wrist site Jacobians. Both the error twist and Jacobian use the world frame. Position and orientation validity masks select the rows of the combined two-wrist least-squares problem. Missing orientation contributes no angular command. Invalid or stale targets return zero action; the environment independently terminates on dropout or timeout.

The output is clipped to the environment's normalized action range. With the default adapter, each joint command changes by at most 0.25 rad/s multiplied by the 0.02 s control period. Joint position margins remain enforced in `TrackingEnv.step`, exactly as they are for PPO. The controller does not introduce a second collision or safety policy.

This is a differential IK baseline using existing DimOS numerical functions. It is not the full production Pink controller. Production Pink and its Pinocchio dependency were not available in this isolated runtime. The gain and damping are provisional. Error reduction in a small kinematic step does not establish dynamic tracking, collision avoidance, robustness, real TCP calibration or hardware safety. Large rotation errors inherit the existing `compute_error_twist` implementation's behavior.

## Local verification

From the integration checkout, with the isolated runtime overlay and existing K1 GPU virtual environment Python:

```bash
PYTHONPATH=/home/cc/Documents/Codex/2026-10-15/task/runtime-deps:/home/cc/Documents/Codex/2026-10-15/task/dimos-r1pro-rl \
/home/cc/Documents/Codex/2026-09-30/task-2/setup/venv-gpu/bin/python -m pytest \
  --confcutdir=dimos/robot/galaxea/r1pro/rl -o addopts='' \
  dimos/robot/galaxea/r1pro/rl/test_controllers.py -q
```

Seven tests passed against the actual pinned model. They check zero action at zero error, no state mutation during action computation, reduced position and orientation errors after small kinematic steps, orientation masking, finite deterministic bounded actions at the zero pose, shared environment rate limiting, joint-margin clipping near a limit, and invalid-target termination. Tests use no hardware, training or additional dependencies.
