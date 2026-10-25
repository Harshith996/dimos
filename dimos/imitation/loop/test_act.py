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

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from dimos.imitation.loop.checkpoint import export_policy, import_policy, read_checkpoint
from dimos.imitation.loop.recording import digest


def test_changed_processor_invalidates_checkpoint_even_when_weights_match(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"test weights")
    processor = tmp_path / "policy_preprocessor.json"
    processor.write_text('{"normalization": 1}')
    files = {path.name: digest(path) for path in tmp_path.iterdir()}
    (tmp_path / "loop.json").write_text(
        json.dumps(
            {
                "model_sha256": files["model.safetensors"],
                "files": files,
            }
        )
    )
    assert read_checkpoint(tmp_path)["files"] == files
    processor.write_text('{"normalization": 2}')
    with pytest.raises(ValueError, match="policy_preprocessor"):
        read_checkpoint(tmp_path)


def test_bundle_import_is_portable_and_rejects_modified_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "processor.json").write_text("{}")
    manifest = {
        "model_sha256": digest(checkpoint / "model.safetensors"),
        "files": {
            name: digest(checkpoint / name) for name in ["model.safetensors", "processor.json"]
        },
    }
    (checkpoint / "loop.json").write_text(json.dumps(manifest))
    bundle = export_policy(checkpoint, tmp_path / "bundle.zip")
    imported = import_policy(bundle, tmp_path / "models")
    assert read_checkpoint(imported) == manifest
    assert import_policy(bundle, tmp_path / "models") == imported
    with ZipFile(bundle, "w") as archive:
        for name in ["loop.json", "model.safetensors", "processor.json"]:
            archive.writestr(
                name,
                b"tampered" if name == "model.safetensors" else (checkpoint / name).read_bytes(),
            )
    with pytest.raises(ValueError, match="checkpoint file changed"):
        import_policy(bundle, tmp_path / "other-models")
    assert not list((tmp_path / "other-models").iterdir())
