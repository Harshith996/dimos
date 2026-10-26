# Rehearse communication delay

`LinkConditions` is an optional application-delivery experiment on
`RobotWebInterface`. WebXR modules expose it through their ordinary module config.
It delays incoming commands/requests and outgoing observations while physics,
device acquisition and robot-local control continue normally. It is disabled
unless configured. The same option works with any WebXR robot; it has no sim2 or
OpenYAM dependency.

```python
from dimos.protocol.testing.link import Delay, LinkConditions, Stall

conditions = LinkConditions(
    seed=7,
    to_robot=Delay(base_s=0.100, jitter_std_s=0.020, correlation_s=0.250),
    to_operator=Delay(base_s=0.100, jitter_std_s=0.030, correlation_s=0.250),
    stalls=(Stall(at_s=30, duration_s=0.4, direction="both"),),
)
```

Pass `link_conditions=conditions` to a WebXR module's `.blueprint()` or to
`RobotWebInterface`. `build_openyam_spatial_sim(link_conditions=conditions)`
forwards this option to its teleop module. The blueprint selects settings; it
does not implement queues or change sim2's motor device.

## Run an experiment

The repository includes [a jitter profile](/docs/usage/examples/link-conditions.json).
From the checkout, with the usual OpenYAM artifact and environment configured:

```bash
dimos run openyam-spatial-sim --server-port 8445 \
  --config docs/usage/examples/link-conditions.json
```

Use the same Desktop view and `/spatial` headset page as the ordinary sim.
The profile adds 100 ms per direction, plus correlated jitter. It includes no
stall by default. The camera/processing delay and actual network delay add to
these configured delays. This is not a claim that the profile matches Anvil.

Each nested field is also a CLI option. For a fixed-delay comparison:

```bash
dimos run openyam-spatial-sim --server-port 8445 \
  --link-conditions.to-robot.base-s 0.1 \
  --link-conditions.to-operator.base-s 0.1
```

Add a scripted gap to either command:

```text
--link-conditions.stalls '[{"at_s":30,"duration_s":0.4,"direction":"both"}]'
```

The timeline starts with the first shaped request, including page/assets loading.
Stalls happen once at the specified elapsed time. Through `dimos shell`, call the
teleop module's `reset_link_conditions()` to disconnect all current experiment
clients, cancel pending deliveries and restart the same seeded timeline. The
browser reconnects and requires a fresh grab. `get_link_conditions()` returns
status without resetting. Resetting link conditions does not reset physics;
a raw physics reset is not an application/session reset.

The OpenYAM spatial viewer treats delayed observations separately from control
ownership. A slow state/RGB-D response keeps the existing pinch and visible tools.
Commands still require current robot-local joint feedback and a live input stream.
A real input gap or joint-feedback failure holds the robot and invalidates the
control epoch; queued commands carrying the old epoch cannot re-grab it. Once the
viewer receives the new epoch, the operator releases and pinches again. Actual
hand-tracking loss and disconnect also require a fresh grab. These rules belong
to the spatial teleop application, not the link emulator or sim2 physics.

## Timing contract

`base_s` is nominal additional one-way delay. `jitter_std_s` is Gaussian standard
deviation, not a hard maximum; zero selects fixed delay. `correlation_s` is the
exponential correlation time constant. Jitter is sampled on a 10 ms grid using
a stationary Gaussian recurrence, with independent seeds derived for each
direction. Random draws follow elapsed time, not message counts. A different
camera rate cannot consume random draws intended for control traffic.

Candidate delays are clipped at zero. A message whose candidate delivery lands
inside a stall waits until the stall ends, including messages queued before the
stall started. Overlapping stalls combine. Within each WebSocket or HTTP response
stream, the final deadline is the maximum of the candidate and preceding deadline.
Later messages can therefore bunch behind an earlier message. Ordering, stalls
and clipping change the final distribution; the profile's nominal parameters
are not a promise about delivered percentiles.

HTTP scope/header arrival is delayed, including GETs whose handlers never read
a body. Responses are captured before their outbound delay; original bytes and
observation timestamps are retained. Streaming chunks are supported. WebSocket
setup/accept is unshaped; data is shaped. A disconnect cancels the session's
queues immediately when observed by the server. Shutdown cancels delayed work.

The message/byte budgets cover all pending work on this endpoint, including a
send in progress. Overflow closes a WebSocket with code 1013 or returns HTTP 503
before response headers; an already-started HTTP response fails. No control
message is silently dropped to make room. Existing application watchdogs and
command validation still run; the adapter never renews timestamps or deadlines.

## Inspect and extend

`GET /link-conditions` returns the profile, elapsed time, connection/queue counts,
overflow/cancellation counts and the last 128 delivery records. This diagnostics
route is unshaped. Records distinguish requested delay, ordering wait, scheduled
deadline and actual server-boundary forwarding time. They contain sizes and
message types, not payloads. Forwarding time is not browser receipt time.

The pure model and bounded deadline queue live in
`dimos/protocol/testing/link.py`. The asynchronous ASGI adapter and its lifecycle
live in `dimos/web/link_conditions.py`. The queue takes explicit times; the live
adapter supplies monotonic time, and tests supply a clock directly. Another
transport can reuse the model/queue with its own delivery and ordering adapter.

This models application-visible delay, jitter and stalls. It does not model
packet loss/retransmission, link bandwidth, or the shared TCP congestion of an
SSH tunnel. Randomly dropping WebSocket messages would not reproduce TCP loss.
The injected condition timeline is repeatable; human actions, OS scheduling and
the underlying connection are not made deterministic.

## Verification

Focused checks cover timing/spread/correlation, message-rate independence,
ordering, preserved observation timestamps, streaming delivery, bounded buffers,
disconnect/reconnect, shutdown and real OpenYAM handler release-before-grab after
reset. Run them with:

```bash
uv run pytest dimos/protocol/testing/test_link.py dimos/web/test_link_conditions.py \
  dimos/robot/manipulators/dual_openyam/test_spatial_viewer.py \
  dimos/robot/manipulators/dual_openyam/blueprints/test_spatial.py \
  dimos/teleop/webxr/test_module.py
```
