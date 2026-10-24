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

import pytest

from dimos.imitation.loop.act import read_checkpoint
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
