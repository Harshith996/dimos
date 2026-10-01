# Add EpisodeStatus as a built-in CDR message

This sanity check adds one message to the CDR proposal. It does not migrate the
robot-learning collector or change its JSON transport. The source is the
[internal model in PR4343](https://github.com/dimensionalOS/dimos/blob/ef5e5f2710c482fb1d43d43ec50fd73b0fbe1dde/dimos/imitation/collection/episode.py).

## Add the definition, generate, import

Prepare the existing checkout once (Python 3.12, uv, Ruff 0.14.3 and Rust 1.92.0
with rustfmt; native checks additionally need their toolchains/Fast CDR):

```sh
uv sync --group tests --group message-codegen --frozen
```

Add `dimos/message_codegen/schemas/dimos_msgs/msg/EpisodeStatus.msg`:

```text
float64 ts
string state
int64 episodes_saved
int64 episodes_discarded
string last_event "init"
string[<=1] task_label
```

Run the same two commands as the built-in tutorial, without type/output flags:

```sh
uv run python -m scripts.generate_builtin_messages
uv run python -m scripts.generate_builtin_messages --check
```

Commit the definition **and** generated files in `packages/dimos-generated/src`.
The editable checkout already consumes this source; no message wheel/native
extension rebuild is needed before a new Python process imports the type:

```python
from dimos_generated.dimos_msgs.msg import EpisodeStatus

status = EpisodeStatus(ts=17.25, state="recording", episodes_saved=2,
                       episodes_discarded=1, last_event="start", task_label=[""])
assert EpisodeStatus.decode(status.encode()) == status
```

A real typed module uses the normal stream API. This is the actual example's
inspector, with domain validation kept outside the generated value class:

```python
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos_generated.dimos_msgs.msg import EpisodeStatus
from examples.episode_status.demo_episode_status import validate_status

class EpisodeInspector(Module):
    status: In[EpisodeStatus]
    observed: Out[EpisodeStatus]

    async def handle_status(self, status: EpisodeStatus) -> None:
        validate_status(status)
        self.observed.publish(status)
```

Run the finite two-worker exchange over an ephemeral loopback Zenoh router:

```sh
uv run python examples/episode_status/demo_episode_status.py
```

It prints `PASS: typed EpisodeStatus state=recording, empty label preserved`
and writes `build/episode-status/status.mcap`. The recording embeds the complete
`ros2msg` schema and raw CDR; source and log timestamps are recorded separately.
The example reports tuning needs without applying host/network configuration.
There is no robot, controller or collection state machine in this example.

## Semantic mapping

| Source model | CDR value / application behavior |
|---|---|
| Required `ts: FiniteFloat` | `float64 ts`; the example validator rejects NaN/infinity. Seconds remain a float, without adding a Header or quantizing the payload. MCAP metadata uses rounded nonnegative uint64 nanoseconds. |
| Required `state: Literal["idle", "recording"]` | String spelling retained; the application validator enforces the two values. `.msg` strings do not encode a Python Literal constraint. |
| Required Python integer counters | Signed `int64`, including negative values allowed by the source model. Wire range is −2^63…2^63−1; arbitrary-size Python integers outside it cannot be represented. |
| `last_event`, default `"init"` | Same default and string spellings `start/save/discard/init`; application validation enforces the set. |
| `task_label: str \| None = None` | Bounded string sequence: `[]` means None, `[""]` means an explicitly empty label, `[label]` means a nonempty label. Construction/validation/codecs enforce at most one element; mutation can be temporarily invalid until validation. |
| Required-field presence / Pydantic coercion | The example-only `episode_status(...)` factory requires the four required inputs. Plain generated values have normal `.msg` zero/empty defaults and do not reproduce Pydantic input coercion. `EpisodeStatus()` therefore needs valid domain fields before application use. |
| JSON document `schema_version: 1` | This is added by `to_json()`, not an internal-model field. It is not added to the CDR value and no JSON recording compatibility is claimed. |

The factory/validator are example application code, not new framework APIs.
Native users must also apply domain rules where needed; generated codecs enforce
wire types/bounds, not this model's finite/enum semantics.

## Native package consumers and codec checks

With Fast CDR 2.4.0 prepared as in the existing message toolchain setup, build the
normal native artifacts, then compile consumers through their exported package:

```sh
bash scripts/package_messages.sh
cmake -S examples/episode_status/cpp -B build/episode-status/cpp \
  -DCMAKE_PREFIX_PATH="$PWD/build/message-codegen/release/install;$PWD/build/message-codegen/install"
cmake --build build/episode-status/cpp --parallel 2
cargo build --locked --manifest-path examples/episode_status/rust/Cargo.toml
DIMOS_EPISODE_NATIVE_REQUIRED=1 uv run python -m pytest -c /dev/null \
  --rootdir . --noconftest -q examples/episode_status/test_episode_status.py
```

The C++ consumer uses `find_package(dimos_generated CONFIG REQUIRED)`, links
`dimos_generated::messages`, and includes `<dimos_generated/messages.hpp>`.
The Rust consumer's ordinary Cargo path dependency selects the committed
`dimos-generated-messages` crate; it does not run another generator. Both read,
write and construct `EpisodeStatus` using the generated types.

The 42 focused checks cover all nine Python/C++/Rust encoder/decoder pairs in
both byte orders; optional None/empty/nonempty labels; signed counters and
out-of-range/bounded-sequence rejection; domain constraints; and independent
MCAP decoding from its embedded schema. CI compiles these consumers, requires
all native cases, runs the actual worker exchange and retains its MCAP.
No generator/build implementation was changed for this message.
