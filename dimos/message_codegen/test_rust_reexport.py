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
import subprocess

import pytest

from dimos.message_codegen import generate as generator
from dimos.message_codegen.ownership import Dependency


def test_rust_dependency_only_package_builds_and_preserves_identity(tmp_path):
    if os.environ.get("DIMOS_NATIVE_ACCEPTANCE") != "1":
        pytest.skip("Set DIMOS_NATIVE_ACCEPTANCE=1 for compiled Rust acceptance")
    source = tmp_path / "inputs/probe_msgs/msg/Reading.msg"
    source.parent.mkdir(parents=True)
    source.write_text("int32 value\n")
    base, alias = tmp_path / "base", tmp_path / "alias"
    generator.generate([tmp_path / "inputs"], base, module="base", shared=True, languages=("rust",))
    generator.generate(
        [],
        alias,
        ["probe_msgs/msg/Reading"],
        module="alias",
        dependencies=(Dependency.load(base),),
        languages=("rust",),
    )
    (alias / "rust/interfaces").rmdir()  # Package archives do not retain empty directories.
    tests = alias / "rust/tests"
    tests.mkdir()
    (tests / "identity.rs").write_text(
        "#[test] fn shared_identity_and_codec() {\n"
        "let value = base_messages::probe_msgs::msg::reading::Reading { value: 42 };\n"
        "let alias: alias_messages::probe_msgs::msg::reading::Reading = value;\n"
        "let wire = re_cdr::to_vec::<_, re_cdr::LittleEndian>(&alias).unwrap();\n"
        "let (decoded, used) = re_cdr::from_bytes::<base_messages::probe_msgs::msg::reading::Reading, re_cdr::LittleEndian>(&wire).unwrap();\n"
        "assert_eq!(decoded.value, 42); assert_eq!(used, wire.len());\n"
        "}\n"
    )
    result = subprocess.run(
        ["cargo", "test", "--offline", "--manifest-path", str(alias / "rust/Cargo.toml")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ".message-generation-" not in (alias / "rust/Cargo.toml").read_text()
