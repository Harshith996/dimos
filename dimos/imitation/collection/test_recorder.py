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

from pathlib import Path
import sys

import pytest

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.module import Module
from dimos.core.native_module import NativeModule
from dimos.core.stream import Out
from dimos.imitation.collection.profile import CollectionProfile
from dimos.imitation.collection.recorder import CollectionRecorderConfig, collect
from dimos.imitation.dataprep.schema import FeatureSpec, OutputConfig, RecordingSchema
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String


class _CameraPublisher(Module):
    color_image: Out[Image]


def _profile(camera_count):
    return CollectionProfile(
        metadata={"name": "test", "robot_type": "test"},
        observations={
            **{
                f"images.{i}": FeatureSpec(
                    stream=f"camera_{i}",
                    message_type=Image,
                    field="data",
                    dtype="video",
                    shape=(8, 8, 3),
                    names=["height", "width", "channels"],
                )
                for i in range(camera_count)
            },
            "state": FeatureSpec(
                stream="measured",
                message_type=JointState,
                field="position",
                dtype="float32",
                shape=(1,),
                names=["joint"],
            ),
        },
        actions={
            "action": FeatureSpec(
                stream="measured",
                message_type=JointState,
                field="position",
                dtype="float32",
                shape=(1,),
                names=["joint"],
            )
        },
    )


@pytest.fixture
def recorder(tmp_path):
    profile = _profile(2)
    profile.actions["action"].stream = "commanded"
    atom = collect(
        profile=profile,
        recording=tmp_path / "collection.mcap",
    ).active_blueprints[0]
    instance = atom.module(**atom.kwargs)
    yield instance
    instance.stop()


@pytest.mark.parametrize("count", [1, 2, 4])
def test_factory_exposes_all_typed_inputs_before_autoconnect(count, tmp_path):
    blueprint = collect(profile=_profile(count), recording=tmp_path / "test.mcap")
    stack = autoconnect(
        blueprint,
        *[_CameraPublisher.blueprint(instance_name=f"publisher_{i}") for i in range(count)],
    ).remappings([(f"publisher_{i}", "color_image", f"camera_{i}") for i in range(count)])
    atom = stack.active_blueprints[0]
    assert {s.name: s.type for s in atom.streams if s.name != "tf"} == {
        **{f"camera_{i}": Image for i in range(count)},
        "measured": JointState,
        "status": String,
    }
    assert len([s for s in atom.streams if s.name == "measured"]) == 1
    # Autoconnect groups remapped ports by (name, message type).
    for i in range(count):
        peers = [
            (peer.name, port.direction)
            for peer in stack.active_blueprints
            for port in peer.streams
            if stack.remapping_map.get((peer.name, port.name), port.name) == f"camera_{i}"
            and port.type is Image
        ]
        assert peers == [(atom.name, "in"), (f"publisher_{i}", "out")]


def test_recorder_resolves_every_connected_stream(recorder, mocker):
    for name in (
        "camera_0",
        "camera_1",
        "measured",
        "commanded",
        "status",
    ):
        getattr(recorder, name).transport = mocker.MagicMock(channel=f"dimos/{name}")
    assert {s.name: s.codec for s in recorder._stream_specs()} == {
        "camera_0": "jpeg",
        "camera_1": "jpeg",
        "measured": "lcm",
        "commanded": "lcm",
        "status": "json",
    }


def test_missing_input_fails_before_native_process_starts(recorder, mocker):
    start = mocker.patch("dimos.experimental.memory.rust_recorder.NativeModule.start")
    with pytest.raises(ValueError, match="Missing required collection inputs"):
        recorder.start()
    start.assert_not_called()


@pytest.mark.parametrize(
    "stream",
    [
        "status",
        "tf",
        "start",
        "config",
        "rpc",
        "ref",
        "input_types",
        "class",
        "bad-name",
        "_private",
    ],
)
def test_invalid_ports_fail_at_factory_boundary(stream, tmp_path):
    profile = _profile(1)
    profile.observations["images.0"].stream = stream
    with pytest.raises(ValueError, match="reserved"):
        collect(profile=profile, recording=tmp_path / "invalid.mcap")


def test_unsupported_message_type_is_rejected(tmp_path):
    profile = _profile(1)
    profile.observations["images.0"].message_type = str
    with pytest.raises(TypeError, match="native recording"):
        collect(profile=profile, recording=tmp_path / "invalid.mcap")


def test_local_message_type_is_rejected(tmp_path):
    class LocalMessage:
        pass

    profile = _profile(1)
    profile.observations["images.0"].message_type = LocalMessage
    with pytest.raises(ValueError, match="importable at module level"):
        collect(profile=profile, recording=tmp_path / "invalid.mcap")


def test_run_config_resolves_collection_destination(tmp_path):
    blueprint = collect(profile=_profile(1))
    parser = BlueprintConfigParser(blueprint)
    help_text = parser.format_help()
    assert "--recorder.recording" in help_text
    assert "--recorder.format" in help_text
    assert "recording-schema" not in help_text
    assert "store.path" in help_text
    parsed = parser.parse(
        ["--recorder.recording", str(tmp_path / "session"), "--recorder.format", "sqlite"],
        environ={},
    )
    atom = blueprint.active_blueprints[0]
    recorder = atom.module(**{**atom.kwargs, **parsed.module_kwargs(atom.name)})
    try:
        assert recorder.config.store.path == str(tmp_path / "session" / "recording.db")
        assert recorder._recording_schema.observation["images.0"].stream == "camera_0"
    finally:
        recorder.stop()


@pytest.fixture
def connected_recorder(tmp_path, mocker):
    def make(format="mcap", **kwargs):
        atom = collect(
            profile=_profile(2), recording=tmp_path / "session", format=format
        ).active_blueprints[0]
        instance = atom.module(
            **atom.kwargs,
            **{"source_dir": None, "build_command": None, "executable": sys.executable, **kwargs},
        )
        for port, _ in instance.recording_inputs:
            getattr(instance, port).transport = mocker.MagicMock(channel=f"dimos/{port}")
        recorders.append(instance)
        return instance

    recorders = []
    yield make
    for instance in recorders:
        instance.stop()


@pytest.mark.parametrize(
    ("format", "payload"), [("mcap", "recording.mcap"), ("sqlite", "recording.db")]
)
def test_build_saves_portable_schema_before_native_capture(
    connected_recorder, format, payload, mocker
):
    recorder = connected_recorder(
        format,
        stream_remapping={"camera_0": "wrist", "status": "episodes"},
    )
    start = mocker.patch.object(NativeModule, "start")
    recorder.build()
    directory = recorder.config.recording
    schema = RecordingSchema.model_validate_json((directory / "schema.json").read_text())
    assert schema.payload == payload
    assert schema.observation["images.0"].stream == "wrist"
    assert schema.episodes.status_stream == "episodes"
    assert schema.action["action"].names == ["joint"]
    config = schema.dataprep_config(directory, OutputConfig(path=directory.parent / "dataset"))
    assert config.source == str(directory / payload)
    start.assert_not_called()
    recorder.start()
    start.assert_called_once_with()
    native_config = recorder.config.to_config_dict()
    assert set(native_config) == {"store", "encoding_threads", "streams"}
    assert native_config["store"] == {"kind": format, "path": str(directory / payload)}
    assert native_config["streams"] == [spec.model_dump() for spec in recorder._stream_specs()]


def test_existing_directory_is_never_overwritten(connected_recorder, mocker):
    recorder = connected_recorder()
    recorder.config.recording.mkdir()
    marker = recorder.config.recording / "schema.json"
    marker.write_text("existing")
    start = mocker.patch.object(NativeModule, "start")
    with pytest.raises(FileExistsError):
        recorder.build()
    assert marker.read_text() == "existing"
    start.assert_not_called()


def test_missing_connections_fail_before_build_or_directory_creation(recorder, mocker):
    native_build = mocker.patch.object(NativeModule, "build")
    with pytest.raises(ValueError, match="Missing required collection inputs"):
        recorder.build()
    native_build.assert_not_called()
    assert not recorder.config.recording.exists()


def test_schema_write_failure_prevents_capture(connected_recorder, mocker):
    recorder = connected_recorder()
    start = mocker.patch.object(NativeModule, "start")
    mocker.patch.object(Path, "open", side_effect=PermissionError("not writable"))
    with pytest.raises(PermissionError, match="not writable"):
        recorder.build()
    start.assert_not_called()
    assert not (recorder.config.recording / "schema.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"store": {"kind": "sqlite", "path": "other.db"}},
        {"store": {"kind": "mcap", "path": "other.mcap"}},
        {"on_existing": "overwrite"},
        {"on_existing": "backup"},
        {"on_existing": "append"},
        {"backup_keep_last": 10},
    ],
)
def test_collection_rejects_conflicting_file_settings_before_creating_directory(tmp_path, kwargs):
    directory = tmp_path / "session"
    with pytest.raises(ValueError):
        CollectionRecorderConfig(recording=directory, **kwargs)
    assert not directory.exists()
