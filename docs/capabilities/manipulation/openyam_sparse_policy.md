# OpenYAM sparse policy boundary

`SparseSimPolicyAdapter` connects an external deterministic sparse49 policy to an
existing `SimRobotClient`. It owns the seven incremental servo targets and checks
observation freshness, reset generations, command acknowledgements and control
timing. It does not open a world or robot connection, load weights, advance physics
or evaluate task success. This interface requires privileged simulator state; it
does not establish camera perception, hardware calibration or policy quality.

The pure NumPy contract and state assembly are in
[`sparse_policy.py`](/dimos/robot/manipulators/openyam/sparse_policy.py). The client
binding is in [`sparse_sim2.py`](/dimos/robot/manipulators/openyam/sparse_sim2.py).
The contract is exported as ordinary JSON using `SparsePolicyContract.to_dict()`;
DimOS has no dependency on the training package or Torch. An external actor owns
its saved observation normalization and returns seven finite actions in `[-1,1]`.

## Verify the artifacts

Before constructing the adapter, load the robot through `load_yam_artifact` in
[`openyam/simulation.py`](/dimos/robot/manipulators/openyam/simulation.py), pinning its
`expected_contract_sha256` to the policy contract's `robot_contract_sha256`.
That loader verifies the model resources and compiled robot parameters. Keep the
verified artifact directory immutable for the world's lifetime.

The application must separately hash the actual policy weight file and compare
it with `policy_sha256` **before loading it**. A hash string in the policy contract
does not verify any file. Authenticate or pin the policy contract itself against
the selected export receipt. Verify the actor architecture and normalization
metadata against that receipt too. Task-source and reset-bank hashes identify the
training task and pool; the adapter does not fetch or verify those files.

The v1 robot artifact supplies the fixed base translation and creates an identity
base orientation. The binding checks both against the policy contract. A rotated
or relocated base requires a compatible export and world configuration. Matching
the robot contract alone does not verify the cube, floor, contacts or task logic;
full task playback parity remains a separate simulation qualification.

## State and reset requirements

The unnormalized state contains eight native joint positions, eight velocities,
cube position, cube WXYZ quaternion, world-axis linear and angular velocity,
seven servo targets, seven previous bounded actions, goal position, normalized
episode clock, normalized dwell count and the lifted latch. The canonical order is
six arm joints, the controlled left finger, then the passive right finger. Read
the passive finger's actual simulator position and velocity; do not infer them
from the controlled finger or the coupling constraint.

Cube, goal and fixed-base positions use world axes translated by the environment
origin. They are not rotated into the robot or cube frame. Quaternions remain
world WXYZ and twists use world axes. All robot feedback, passive joint feedback,
object state and task counters must come from the same physics tick. The
`PrivilegedTaskState.stamp` must equal `policy_stamp(observed)` for the actual
`StateFrame` returned by that client's `read()`.

After a simulator reset, call `reset(observed=frame, task=task,
servo_targets=preload)`. Supply the exact native holding targets from the reset
transaction, independently of measured joint positions. Contact and gravity can
make those values differ. Reset the task clock and dwell count to zero; initialize
the lifted latch from the reset state using the task's declared predicate. The
adapter resets previous action to zero and sends no command. Each further reset
must use a newer simulator reset generation.

The sparse lift-hold export uses a 5 ms physics step, four physics steps per policy
action, a 400-action horizon and 25 consecutive qualifying hold steps. These
values are explicit contract fields. Task predicates, including the hold count,
terminal status and lifted latch, remain the caller's responsibility.
Normalized counters use `float32(count) * float32(1 / limit)`, matching the
recorded CUDA scalar-division arithmetic. Dividing in Python float64 before casting
can differ by one float32 ULP and is not the exported observation convention.

## One inference transaction

1. Obtain one coherent simulator snapshot and its actual client `StateFrame`.
   Construct the privileged task state with that frame's exact stamp.
2. Call `observation(observed=frame, task=task)` to receive a read-only float32
   vector of length 49. Pass it to the external actor with its frozen normalizer.
3. Call `write_action(action, observed=frame, task=task)` with the **same frame and
   task objects** used for inference. Do not refresh the observation to make an
   old action appear fresh. The adapter forwards that frame to `SimRobotClient`.
4. Advance exactly the declared control period, then obtain the next coherent
   snapshot. Its applied-action sequence must acknowledge the preceding write.

Each accepted action updates the float32 accumulator once:
`targets = clip(targets + action * action_scales, target_limits)`. The sparse
contract uses 0.02 radians per arm coordinate and 0.001 metres for the controlled
finger. Zero action holds the reset preload or last targets. The passive finger
is never independently commanded. The robot adapter converts the controlled
finger target to the public normalized gripper coordinate.

An action sample can be consumed only once. Skipped ticks, stale feedback,
unacknowledged commands, changed generations and malformed states/actions raise
errors. Any exception during command transmission faults the episode because the
command may already have been applied. Reset the simulator and the adapter before
continuing; do not retry the same increment. The caller must serialize physics
and snapshot access, own the single command writer and close its world/client
resources. The adapter's lock protects its own memory, not the simulator.

CPU unit tests cover exact state layout and recurrence, passive feedback, frame
identity, transport uncertainty and rejection behavior in
[`test_sparse_policy.py`](/dimos/robot/manipulators/openyam/test_sparse_policy.py).
They do not step physics or demonstrate successful learned behavior.

## Recorded native simulation result

On October 24, 2026, the exported historical A2 sparse PPO policy completed one
predeclared `validation-000:reaching` episode through the actual local-only
`SimRobotClient` and this adapter. It grasped the grounded cube, lifted it and
completed the required 25-step hold, with no drop or constraint violation. The
episode ran 400 policy commands and 1,600 CPU MuJoCo physics steps. This starts
with the gripper aligned near the cube; it does not demonstrate reaching from the
distant home pose. The actor uses privileged simulator state and no scripted
motion or imitation controller.

The independent direct-CPU comparison also succeeded, and all Boolean task
flags agreed. **Strict numerical/contact parity failed and remains unqualified:**
contact sets differed on 46 of 1,600 substeps; maximum cube-position difference
was 6.45 micrometres, joint-position difference 0.000105 radians and contact-force
difference 0.0456 newtons. Physical model parameters matched exactly in the named
model audit. These facts establish one successful native policy playback, not
exact cross-composition dynamics, CPU/Warp equivalence, a performance estimate,
SGS superiority or hardware readiness.

The [complete eight-second replay](https://github.com/Nabla7/humanoid-control-lab/releases/download/sparse-policy-a2-numpy-v1/a2-native-sim2-validation-000.mp4)
and [measured result](https://github.com/Nabla7/humanoid-control-lab/blob/7e27868ceb0c33fd792549e4de253eb03407979b/results/sparse-policy-native-sim2-playback-002.json)
require access to the private research repository. The replay contains every
recorded state through termination; software rendering added no physics or
policy inference. The [policy export release](https://github.com/Nabla7/humanoid-control-lab/releases/tag/sparse-policy-a2-numpy-v1)
retains its original archive unchanged.

Exact recorded playback and rendering pin DimOS source revision
`d28399a91ae5a3368d6a2f342353f25e2b5f9f99`, MuJoCo 3.11.0 and the exported
model/policy hashes. The frozen research runners reject a different checkout
revision, including a later documentation-only commit. Use a separate checkout
at that recorded revision when repeating them. The actual run is
`20261024T020236868316Z-yam-sparse-sim2-parity-v2-seed0`; its failed strict gate,
complete trajectories and successful task outcome are retained together.
