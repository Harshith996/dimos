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

"""Check collection API migrations in a checkout or a candidate Git tree.

Run with the project Python: python scripts/check_collection_stack.py [--ref REF].
This reads source only and never starts hardware or changes Git references.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Inspect a Git tree instead of checkout files")
    args = parser.parse_args()
    command = ["git", "ls-tree", "-r", "--name-only", args.ref] if args.ref else ["git", "ls-files"]
    names = subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
    errors: list[str] = []
    checked = 0
    for name in names:
        if not name.endswith(".py") or not name.startswith(("dimos/", "native/python/lerobot/")):
            continue
        if args.ref:
            source = subprocess.check_output(
                ["git", "show", f"{args.ref}:{name}"], cwd=ROOT, text=True
            )
        else:
            path = ROOT / name
            if not path.exists():
                continue
            source = path.read_text()
        # Parse only files participating in collection to keep the audit fast.
        if "dimos.imitation.collection" not in source and "class CollectionProfile" not in source:
            continue
        checked += 1
        for node in ast.walk(ast.parse(source, filename=name)):
            if isinstance(node, ast.ImportFrom):
                if node.module == "dimos.imitation.collection.composition" or any(
                    alias.name in {"collection_recorder", "joint_rgb_profile", "ActionSource"}
                    for alias in node.names
                ):
                    errors.append(f"{name}:{node.lineno}: obsolete collection import")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "CollectionProfile"
            ):
                removed = {kw.arg for kw in node.keywords} & {
                    "name",
                    "robot_type",
                    "sync",
                    "quality",
                }
                if removed:
                    errors.append(f"{name}:{node.lineno}: removed profile fields {sorted(removed)}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Collection API audit passed: {checked} source files ({args.ref or 'checkout'})")


if __name__ == "__main__":
    main()
