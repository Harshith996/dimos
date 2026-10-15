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

import ast
import importlib.metadata
from pathlib import Path
import re
import site
import sys

from packaging.requirements import Requirement

from dimos.constants import DIMOS_PROJECT_ROOT

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DIMOS_DIR = DIMOS_PROJECT_ROOT / "dimos"
PYPROJECT = DIMOS_PROJECT_ROOT / "pyproject.toml"

# Installed from outside pyproject.toml: the ZED SDK wheel and the in-tree
# maturin extensions, both set up by bin/build-test-natives.
UNDECLARABLE_MODULES = frozenset(
    {
        "pyzed",
        "dimos_local_planner",
        "dimos_mls_planner",
        "dimos_trajectory_follower",
        "dimos_voxel_ray_tracing",
    }
)

# Distributions whose installed metadata does not list their top-level module,
# so importlib.metadata cannot map the module back to them.
MODULE_TO_DISTRIBUTION = {
    "rerun": "rerun-sdk",
    "pinocchio": "pin",
}


def _normalize(name: str) -> str:
    """PEP 503 distribution name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_distributions() -> set[str]:
    """Every distribution named in [project] dependencies, extras and dependency groups.

    Build requirements are deliberately left out: they are not available at runtime.
    """
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs: list[str] = list(pyproject["project"]["dependencies"])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    for group in pyproject.get("dependency-groups", {}).values():
        # Non-string entries are `{include-group = "..."}` tables.
        specs.extend(spec for spec in group if isinstance(spec, str))
    return {_normalize(Requirement(spec).name) for spec in specs}


def imported_modules() -> dict[str, list[str]]:
    """Top-level third-party module -> ["dimos/path.py:line", ...], one entry per importing file."""
    first_lines: dict[str, dict[Path, int]] = {}
    for path in sorted(DIMOS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                module = name.split(".")[0]
                if module in sys.stdlib_module_names or module in ("__future__", "dimos"):
                    continue
                files = first_lines.setdefault(module, {})
                files[path] = min(files.get(path, node.lineno), node.lineno)
    return {
        module: [f"{path.relative_to(DIMOS_PROJECT_ROOT)}:{line}" for path, line in files.items()]
        for module, files in first_lines.items()
    }


def test_imports_are_declared_in_pyproject() -> None:
    """Fail if dimos imports a package that only arrives as a dependency of a dependency."""
    declared = declared_distributions()
    assert "numpy" in declared, "pyproject.toml parsing is broken"

    installed = importlib.metadata.packages_distributions()
    # Only distributions inside this venv are pyproject.toml's business. Anything
    # else on sys.path, such as a sourced ROS 2 install, is ignored.
    in_venv = {
        _normalize(dist.metadata["Name"])
        for dist in importlib.metadata.distributions(path=site.getsitepackages())
    }
    undeclared: dict[str, list[str]] = {}
    for module, locations in imported_modules().items():
        if module in UNDECLARABLE_MODULES:
            continue
        if module in MODULE_TO_DISTRIBUTION:
            candidates = [MODULE_TO_DISTRIBUTION[module]]
        else:
            candidates = installed.get(module, [])
        names = sorted({_normalize(name) for name in candidates} & in_venv)
        if not names:
            # Not installed in this venv, so it cannot be mapped to a distribution.
            # A transitive dependency is always installed, so this only skips
            # vendor SDKs and the packages of extras that were not synced here.
            continue
        if not any(name in declared for name in names):
            undeclared[f"{module} (distribution: {', '.join(names)})"] = locations

    if undeclared:
        listing = "\n".join(
            f"  - {module}\n" + "\n".join(f"      {location}" for location in locations)
            for module, locations in sorted(undeclared.items())
        )
        raise AssertionError(
            f"Imports of packages that pyproject.toml does not declare:\n{listing}\n\n"
            "These packages are only installed because another dependency happens to "
            "pull them in. Add each one to [project.dependencies], to the extra that "
            "covers the importing code, or to a dependency group for test-only use."
        )
