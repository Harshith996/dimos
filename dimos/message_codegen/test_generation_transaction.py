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

import pytest

from dimos.message_codegen import generate as generator


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_failed_generation_preserves_previous_output_and_success_removes_stale_files(
    tmp_path, monkeypatch
):
    source = tmp_path / "inputs/probe_msgs/msg/Reading.msg"
    source.parent.mkdir(parents=True)
    source.write_text("string value\n")
    root = tmp_path / "output"
    generator.generate([tmp_path / "inputs"], root, module="probe")
    before = snapshot(root)
    source.write_text("string<=4 value\n")
    with pytest.raises(NotImplementedError):
        generator.generate([tmp_path / "inputs"], root, module="probe")
    assert snapshot(root) == before
    source.write_text("string value\n")
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("emitter failure")

        patch.setattr(generator.rust, "write_crate", fail)
        with pytest.raises(RuntimeError, match="emitter failure"):
            generator.generate([tmp_path / "inputs"], root, module="probe")
    assert snapshot(root) == before
    with monkeypatch.context() as patch:
        rename = Path.rename

        def fail_install(self, target):
            if self.name == "output" and self.parent.name.startswith(".message-generation-"):
                raise OSError("install failure")
            return rename(self, target)

        patch.setattr(Path, "rename", fail_install)
        with pytest.raises(OSError, match="install failure"):
            generator.generate([tmp_path / "inputs"], root, module="probe")
    assert snapshot(root) == before
    source.rename(source.with_name("Changed.msg"))
    generator.generate([tmp_path / "inputs"], root, module="probe", languages=("python",))
    assert not (root / "rust").exists()
    assert not (root / "cpp").exists()
    assert not (root / "schemas/probe_msgs/msg/Reading.msg").exists()
    assert (root / "schemas/probe_msgs/msg/Changed.msg").exists()
    assert not list(tmp_path.glob(".message-generation-*"))
