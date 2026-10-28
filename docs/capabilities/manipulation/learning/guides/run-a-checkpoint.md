# Run a compatible checkpoint

Start with a local pretrained-model directory and the single-arm, single-camera
OpenYAM setup expected by the policy.

Stop the collection stack before launching rollout. Replace `CHECKPOINT_DIR`
with a compatible pretrained-model directory, such as
`outputs/openyam-act/checkpoints/last/pretrained_model`.

```bash skip
dimos --can-port follower_l run openyam-lerobot-rollout --daemon \
  --policy.policy-path CHECKPOINT_DIR \
  --policy.task "pick up the red block" \
  --policy.device cuda \
  --wristcamera.hardware.camera-index 0
dimos imitation rollout
```

Use `openyam-lerobot-quest-rollout` for the graph with Quest takeover.
The optional rollout panel discovers `RolloutControlSpec` and shows policy
state and errors. Space explicitly starts/stops policy execution. Before
starting, preflight loads the checkpoint and checks the control task and fresh
observations without sending a trajectory. A failed check leaves the policy
stopped and reports the error.

Q only detaches, even while the policy is active. Run `dimos imitation rollout`
again to reattach. To finish, stop the policy with Space, detach, support the
arm, and use `dimos stop` to shut down the runtime. Neither UI is a deadman
switch: lost connectivity does not guarantee stopping motion.

See the [LeRobot module contract](/dimos/imitation/policy/lerobot/README.md)
for checkpoint and control requirements. Rollout uses the existing single-arm,
single-camera contract with absolute joint targets in the hardware's native
coordinates. A prepared dataset does not establish checkpoint compatibility
with another robot. ABC integration, dual-arm policy rollout, and policy
backend generalization are deferred.
