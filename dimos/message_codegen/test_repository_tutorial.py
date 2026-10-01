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
import shutil
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_builtin_tutorial_builds_new_definition_without_runtime(tmp_path):
    wheelhouse = os.environ.get("DIMOS_MESSAGE_WHEELHOUSE")
    fastcdr = os.environ.get("DIMOS_FASTCDR_PREFIX")
    if not wheelhouse or not fastcdr:
        pytest.skip("Set message wheelhouse and Fast CDR prefix for clean source acceptance")
    root = Path(__file__).resolve().parents[2]
    tutorial = (root / "docs/development/messages-in-repository.md").read_text()
    definition = re.search(r"<!-- builtin-message: [^>]+ -->\n```text\n(.*?)```", tutorial, re.S)
    check = re.search(r"<!-- builtin-value-check -->\n```python skip\n(.*?)```", tutorial, re.S)
    assert definition is not None and check is not None
    checkout = tmp_path / "checkout"
    ignore = shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", "target")
    shutil.copytree(
        root / "dimos/message_codegen", checkout / "dimos/message_codegen", ignore=ignore
    )
    (checkout / "dimos/__init__.py").write_text("")
    package = checkout / "packages/dimos-generated"
    shutil.copytree(root / "packages/dimos-generated", package, ignore=ignore)
    source = checkout / "dimos/message_codegen/schemas/dimos_msgs/msg/DeviceReading.msg"
    assert not source.exists()
    source.write_text(definition.group(1))
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(
        PATH=os.pathsep.join(
            directory
            for directory in os.environ["PATH"].split(os.pathsep)
            if Path(directory) != Path(sys.executable).parent
        ),
        PIP_NO_INDEX="1",
        PIP_FIND_LINKS=str(Path(wheelhouse).resolve()),
        CMAKE_PREFIX_PATH=str(Path(fastcdr).resolve()),
    )
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = str(venv / "bin/python")
    dist = tmp_path / "dist"
    uv = shutil.which("uv")
    assert uv is not None, "Prepare uv before running the repository tutorial"
    subprocess.run(
        [
            uv,
            "build",
            str(package),
            "--python",
            python,
            "--out-dir",
            str(dist),
            "--no-index",
            "--find-links",
            str(Path(wheelhouse).resolve()),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    wheel = next(dist.glob("dimos_generated-*.whl"))
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    validation = (
        check.group(1)
        + """
from importlib.metadata import distributions
assert 'dimos' not in {dist.metadata['Name'] for dist in distributions()}
assert 'MSG: std_msgs/Header' in DeviceReading.schema
"""
    )
    subprocess.run([python, "-I", "-c", validation], cwd=tmp_path, env=environment, check=True)


def test_checkout_sync_sees_generated_source_changes_without_reinstall(tmp_path):
    wheelhouse = os.environ.get("DIMOS_MESSAGE_WHEELHOUSE")
    if not wheelhouse:
        pytest.skip("Set message wheelhouse for clean checkout acceptance")
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    source = config["tool"]["uv"]["sources"]["dimos-generated"]
    checkout = tmp_path / "checkout"
    package = checkout / source["path"]
    shutil.copytree(
        root / source["path"],
        package,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "checkout-probe"\nversion = "0.1.0"\n'
        f'requires-python = "=={sys.version_info.major}.{sys.version_info.minor}.*"\n'
        'dependencies = ["dimos-generated==0.1.0"]\n[tool.uv.sources]\n'
        f'dimos-generated = {{ path = "{source["path"]}", editable = {str(source.get("editable", False)).lower()} }}\n'
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    environment.update(CC="/bin/false", CXX="/bin/false", UV_PYTHON_DOWNLOADS="never")
    uv = shutil.which("uv")
    assert uv is not None, "Prepare uv before running checkout acceptance"
    subprocess.run(
        [
            uv,
            "sync",
            "--python",
            sys.executable,
            "--no-index",
            "--find-links",
            str(Path(wheelhouse).resolve()),
        ],
        cwd=checkout,
        env=environment,
        check=True,
    )
    python = str(checkout / ".venv/bin/python")
    validation = (
        "from dimos_generated.geometry_msgs.msg import Point; "
        "assert Point.decode(Point(x=1.25).encode()).x == 1.25; "
        "assert getattr(Point, 'CHECKOUT_PROBE', None) == 17"
    )
    # Stand in for a regenerated constant in a committed source value class.
    # The isolated wheel tutorial separately verifies .msg -> all-language generation.
    values = package / "src/dimos_generated/_types.py"
    values.write_text(
        values.read_text().replace(
            "class geometry_msgs__msg__Point(Message):",
            "class geometry_msgs__msg__Point(Message):\n    CHECKOUT_PROBE = 17",
        )
    )
    subprocess.run([uv, "sync", "--frozen", "--offline"], cwd=checkout, env=environment, check=True)
    subprocess.run([python, "-I", "-c", validation], cwd=checkout, env=environment, check=True)
