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

"""Explicit authenticated installation of Lesh's private Go2 policy, never in Git.

python -m dimos.control.go2_freewalk.models --install
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from dimos.constants import DIMOS_PROJECT_ROOT

MODEL_SHA256 = "406e735c32a8e5501c9122ef83fa61d67c70aa9bfc9a4dae266b321db1458330"
MODEL_BLOB = "20769582f37cc4dda34b88260270c1e4858b5df3"
MODEL_REPO = "dimensionalOS/go2web"


def model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "dimos" / "policies" / "go2-freewalk" / MODEL_SHA256 / "freewalk_mcf.bin"


def _external(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_relative_to(DIMOS_PROJECT_ROOT.resolve()):
        raise ValueError("private FREE policy weights must stay outside the DimOS repository")
    return path


def _verify(data: bytes) -> bytes:
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Go2 FREE policy SHA-256 mismatch")
    return data


def load_weights(path: Path | None = None) -> bytes:
    path = _external(path or model_path())
    if not path.is_file():
        raise FileNotFoundError(
            f"Go2 FREE policy not installed at {path}; authenticate gh for {MODEL_REPO}, "
            "then run python -m dimos.control.go2_freewalk.models --install"
        )
    return _verify(path.read_bytes())


def install() -> Path:
    path = _external(model_path())
    if path.is_file():
        load_weights(path)
        return path
    result = subprocess.run(
        ["gh", "api", f"repos/{MODEL_REPO}/git/blobs/{MODEL_BLOB}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    data = _verify(base64.b64decode(json.loads(result.stdout)["content"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install", action="store_true", help="download using existing gh authentication"
    )
    args = parser.parse_args()
    if args.install:
        print(install())
    else:
        load_weights()
        print(model_path())


if __name__ == "__main__":
    main()
