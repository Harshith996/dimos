# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import re
import subprocess
from typing import get_args

from dimos_generated.dimos_msgs.msg import EpisodeStatus
from mcap.reader import make_reader
import pytest
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from examples.episode_status.demo_episode_status import (
    episode_status,
    record_status,
    validate_status,
)


def sample(label=""):
    return episode_status(
        ts=17.25,
        state="recording",
        episodes_saved=2,
        episodes_discarded=1,
        last_event="start",
        task_label=label,
    )


def test_readme_definition_and_python_blocks_are_copyable():
    root = Path(__file__).resolve().parents[2]
    readme = Path(__file__).with_name("README.md").read_text()
    definition = (
        root / "dimos/message_codegen/schemas/dimos_msgs/msg/EpisodeStatus.msg"
    ).read_text()
    fields = "\n".join(
        line for line in definition.splitlines() if line and not line.startswith("#")
    )
    assert re.search(r"```text\n(.*?)\n```", readme, re.S).group(1) == fields
    namespace = {"__name__": "episode_status_documentation"}
    for block in re.findall(r"```python\n(.*?)\n```", readme, re.S):
        exec(compile(block, "episode-status-README", "exec"), namespace)
    assert namespace["status"] == sample()
    assert get_args(namespace["EpisodeInspector"].__annotations__["status"])[0] is EpisodeStatus


@pytest.fixture
def reference():
    store = get_typestore(Stores.EMPTY)
    store.register(get_types_from_msg(EpisodeStatus.schema, EpisodeStatus.msg_name))
    return store


@pytest.mark.parametrize("label", [None, "", "pick the cup"])
@pytest.mark.parametrize("little", [True, False])
def test_nullable_label_survives_cdr_and_independent_decode(label, little, reference):
    status = sample(label)
    payload = status.encode(little_endian=little)
    decoded = reference.deserialize_cdr(payload, EpisodeStatus.msg_name)
    assert (decoded.ts, decoded.state, decoded.episodes_saved, decoded.episodes_discarded) == (
        17.25,
        "recording",
        2,
        1,
    )
    assert decoded.last_event == "start"
    assert decoded.task_label == ([] if label is None else [label])
    assert (
        bytes(reference.serialize_cdr(decoded, EpisodeStatus.msg_name, little_endian=little))
        == payload
    )
    assert EpisodeStatus.decode(payload) == status


@pytest.mark.parametrize(
    "field,value",
    [("ts", float("nan")), ("ts", float("inf")), ("state", "paused"), ("last_event", "stop")],
)
def test_source_domain_validation_rejects_invalid_values(field, value):
    status = sample()
    setattr(status, field, value)
    with pytest.raises(ValueError, match=f"EpisodeStatus.{field}"):
        validate_status(status)


def test_source_required_fields_and_defaults_are_explicit():
    with pytest.raises(TypeError, match="required keyword-only"):
        episode_status(ts=1.0, state="idle")
    status = episode_status(ts=1.0, state="idle", episodes_saved=0, episodes_discarded=0)
    assert status.last_event == "init"
    assert list(status.task_label) == []
    with pytest.raises(ValueError, match="state"):
        validate_status(EpisodeStatus())


def test_signed_counter_range_and_optional_bound(reference):
    status = sample(None)
    status.episodes_saved = -(2**63)
    status.episodes_discarded = 2**63 - 1
    decoded = reference.deserialize_cdr(status.encode(), EpisodeStatus.msg_name)
    assert (decoded.episodes_saved, decoded.episodes_discarded) == (-(2**63), 2**63 - 1)
    with pytest.raises(ValueError):
        status.episodes_saved = 2**63
    status.task_label = ["one", "two"]
    with pytest.raises(ValueError):
        status.encode()


@pytest.mark.parametrize("label", [None, "", "pick the cup"])
def test_mcap_embeds_complete_schema_and_preserves_fields_and_source_timestamp(tmp_path, label):
    path = tmp_path / "status.mcap"
    status = sample(label)
    record_status(path, status)
    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        assert reader.get_header().profile == "ros2"
        rows = list(reader.iter_messages())
    assert len(rows) == 1
    schema, channel, message = rows[0]
    assert (schema.name, schema.encoding, channel.message_encoding) == (
        "dimos_msgs/msg/EpisodeStatus",
        "ros2msg",
        "cdr",
    )
    assert (message.publish_time, message.log_time) == (17_250_000_000, 17_250_000_001)
    independent = get_typestore(Stores.EMPTY)
    independent.register(get_types_from_msg(schema.data.decode(), schema.name))
    decoded = independent.deserialize_cdr(message.data, schema.name)
    assert decoded.task_label == ([] if label is None else [label])
    assert decoded.state == "recording" and decoded.last_event == "start"
    assert message.data == status.encode()


@pytest.fixture(scope="module")
def native_consumers():
    root = Path(__file__).resolve().parents[2]
    binaries = {
        "cpp": root / "build/episode-status/cpp/episode-status",
        "rust": root / "examples/episode_status/rust/target/debug/episode-status-consumer",
    }
    missing = [name for name, path in binaries.items() if not path.is_file()]
    if missing:
        reason = f"Build the documented EpisodeStatus native consumers: {missing}"
        if os.environ.get("DIMOS_EPISODE_NATIVE_REQUIRED") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    return binaries


@pytest.mark.parametrize("encoder", ["python", "cpp", "rust"])
@pytest.mark.parametrize("decoder", ["python", "cpp", "rust"])
@pytest.mark.parametrize("little", [True, False])
def test_all_nine_codec_pairs_and_both_endians(
    tmp_path, native_consumers, encoder, decoder, little
):
    source, echoed = tmp_path / "source.cdr", tmp_path / "echo.cdr"
    expected = sample().encode(little_endian=little)
    if encoder == "python":
        source.write_bytes(expected)
    else:
        subprocess.run(
            [
                str(native_consumers[encoder]),
                "seed" if little else "seed-be",
                str(source),
                str(source),
            ],
            check=True,
        )
    assert source.read_bytes() == expected
    if decoder == "python":
        assert EpisodeStatus.decode(source.read_bytes()) == sample()
    else:
        subprocess.run(
            [
                str(native_consumers[decoder]),
                "echo" if little else "echo-be",
                str(source),
                str(echoed),
            ],
            check=True,
        )
        assert echoed.read_bytes() == expected


@pytest.mark.parametrize("decoder", ["cpp", "rust"])
@pytest.mark.parametrize("label", [None, "", "pick the cup"])
def test_native_consumers_preserve_optional_labels_and_signed_counters(
    tmp_path, native_consumers, decoder, label
):
    status = sample(label)
    status.episodes_saved = -1
    source, target = tmp_path / "source.cdr", tmp_path / "target.cdr"
    source.write_bytes(status.encode())
    subprocess.run([str(native_consumers[decoder]), "echo", str(source), str(target)], check=True)
    assert EpisodeStatus.decode(target.read_bytes()) == status


@pytest.mark.parametrize("decoder", ["cpp", "rust"])
def test_native_consumers_reject_optional_sequence_overflow(
    tmp_path, native_consumers, reference, decoder
):
    value = reference.deserialize_cdr(sample().encode(), EpisodeStatus.msg_name)
    value.task_label = ["one", "two"]
    source, target = tmp_path / "invalid.cdr", tmp_path / "target.cdr"
    source.write_bytes(bytes(reference.serialize_cdr(value, EpisodeStatus.msg_name)))
    result = subprocess.run(
        [str(native_consumers[decoder]), "echo", str(source), str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not target.exists()
