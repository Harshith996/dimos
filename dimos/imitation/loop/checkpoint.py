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

"""Portable, integrity-checked policy artifacts for local deployment."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZIP_STORED, ZipFile

from dimos.imitation.loop.recording import digest


def read_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Verify weights, processors and configuration before loading either runtime."""
    manifest: dict[str, Any] = json.loads((checkpoint / "loop.json").read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise ValueError("checkpoint has no immutable runtime file manifest")
    for name, expected in files.items():
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError(f"checkpoint file changed: {name}")
    if files["model.safetensors"] != manifest["model_sha256"]:
        raise ValueError("checkpoint model identity differs from its file manifest")
    return manifest


def export_policy(checkpoint: Path, destination: Path) -> Path:
    manifest = read_checkpoint(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    with ZipFile(partial, "w", compression=ZIP_STORED) as archive:
        for name in ["loop.json", *manifest["files"]]:
            archive.write(checkpoint / name, name)
    partial.replace(destination)
    return destination


def import_policy(bundle: Path, models: Path) -> Path:
    """Validate a complete artifact before adding it to the model catalog."""
    models.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=models, prefix=".import-") as directory:
        staged = Path(directory)
        with ZipFile(bundle) as archive:
            names = archive.namelist()
            if any(Path(name).name != name or name in (".", "..") for name in names) or len(
                set(names)
            ) != len(names):
                raise ValueError("Policy bundle must contain unique, flat file names")
            if sum(info.file_size for info in archive.infolist()) > 4 * 1024**3:
                raise ValueError("Policy bundle exceeds 4 GiB")
            archive.extractall(staged)
        manifest = read_checkpoint(staged)
        if set(names) != {"loop.json", *manifest["files"]}:
            raise ValueError("Policy bundle contents differ from its manifest")
        destination = models / ("imported-" + digest(staged / "loop.json")[:16])
        if destination.exists():
            read_checkpoint(destination)
            return destination
        staged.rename(destination)
        return destination
