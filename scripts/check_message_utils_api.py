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

"""Check message helper callers use domain namespaces and current function names.

Run ``python scripts/check_message_utils_api.py`` from any checkout in the CDR
stack after migrating callers. This check does not import optional runtimes.
"""

import ast
from pathlib import Path

DOMAINS = {"image", "camera_info", "geometry", "pointcloud", "time", "occupancy"}
ROOT = Path(__file__).resolve().parents[1]


def check() -> list[str]:
    exports: dict[str, set[str]] = {}
    for domain in DOMAINS:
        module = ast.parse((ROOT / "dimos/message_utils" / f"{domain}.py").read_text())
        exports[domain] = {
            node.name for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        exports[domain].update(
            target.id
            for node in module.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        )
    errors = []
    for folder in ("dimos", "examples", "scripts", "native"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            source = path.read_text()
            if "dimos.message_utils" not in source or path == Path(__file__).resolve():
                continue
            tree = ast.parse(source)
            aliases: dict[str, str] = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                if node.module == "dimos.message_utils":
                    aliases.update(
                        (name.asname or name.name, name.name)
                        for name in node.names
                        if name.name in DOMAINS
                    )
                elif node.module.startswith("dimos.message_utils."):
                    domain = node.module.rsplit(".", 1)[-1]
                    for name in node.names:
                        if domain in DOMAINS and not name.name[0].isupper():
                            errors.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}: "
                                f"import the {domain} module instead of {name.name}"
                            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    target_domain = aliases.get(node.value.id)
                    if target_domain is not None and node.attr not in exports[target_domain]:
                        errors.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: "
                            f"unknown helper {target_domain}.{node.attr}"
                        )
    return errors


if __name__ == "__main__":
    failures = check()
    for failure in failures:
        print(failure)
    raise SystemExit(bool(failures))
