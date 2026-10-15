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
import sys

from packaging.requirements import Requirement

from dimos.constants import DIMOS_PROJECT_ROOT

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DIMOS_DIR = DIMOS_PROJECT_ROOT / "dimos"
PYPROJECT = DIMOS_PROJECT_ROOT / "pyproject.toml"

# Modules that pyproject.toml cannot declare because something other than
# `uv sync` installs them.
EXTERNAL_MODULES = frozenset(
    {
        # ROS 2 system install (dimos/protocol/pubsub/impl/rospubsub.py and friends).
        "ament_index_python",
        "builtin_interfaces",
        "geometry_msgs",
        "nav_msgs",
        "rclpy",
        "sensor_msgs",
        "std_msgs",
        # GStreamer bindings from the system PyGObject package.
        "gi",
        # Habitat-Sim is conda-only.
        "habitat_sim",
        # Vendor SDKs. bin/hardware/a1z/setup installs a1z from git and, on macOS,
        # gs-usb and pyusb. a750_control ships its own binding, and the ZED SDK
        # wheel (pyzed) comes from bin/build-test-natives.
        "a1z",
        "a750_control",
        "gs_usb",
        "pyzed",
        "usb",
        # In-tree maturin extensions built by bin/build-test-natives.
        "dimos_local_planner",
        "dimos_mls_planner",
        "dimos_trajectory_follower",
        "dimos_voxel_ray_tracing",
    }
)

# Module name -> distribution name for every import whose distribution is named
# differently. Consulted when the module is not installed, since only installed
# metadata can answer otherwise. rerun-sdk and pin ship no usable metadata at all.
MODULE_TO_DISTRIBUTION = {
    "PIL": "pillow",
    "bosdyn": "bosdyn-client",
    "can": "python-can",
    "cv2": "opencv-contrib-python",
    "dotenv": "python-dotenv",
    "ffmpeg": "ffmpeg-python",
    "git": "gitpython",
    "gtsam": "gtsam-extended",
    "hydra": "hydra-core",
    "lcm": "lcm-dimos-fork",
    "mujoco_playground": "playground",
    "open_clip": "open-clip-torch",
    "pink": "pin-pink",
    "pinocchio": "pin",
    "plum": "plum-dispatch",
    "pxr": "usd-core",
    "pydrake": "drake",
    "rerun": "rerun-sdk",
    "sam2": "edgetam-dimos",
    "socketio": "python-socketio",
    "turbojpeg": "pyturbojpeg",
    "unitree_sdk2py": "unitree-sdk2py-dimos",
    "warp": "warp-lang",
    "websocket": "websocket-client",
    "whisper": "openai-whisper",
    "xarm": "xarm-python-sdk",
    "yaml": "pyyaml",
    "zenoh": "eclipse-zenoh",
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
    """Fail if dimos imports a package that pyproject.toml does not declare."""
    declared = declared_distributions()
    assert "numpy" in declared, "pyproject.toml parsing is broken"

    installed = importlib.metadata.packages_distributions()
    undeclared: dict[str, list[str]] = {}
    for module, locations in imported_modules().items():
        if module in EXTERNAL_MODULES:
            continue
        names = sorted({_normalize(name) for name in installed.get(module, [])})
        if names:
            label = f"{module} (distribution: {', '.join(names)})"
        else:
            names = [_normalize(MODULE_TO_DISTRIBUTION.get(module, module))]
            label = f"{module} (not installed, expected distribution: {names[0]})"
        if not any(name in declared for name in names):
            undeclared[label] = locations

    if undeclared:
        listing = "\n".join(
            f"  - {label}\n" + "\n".join(f"      {location}" for location in locations)
            for label, locations in sorted(undeclared.items())
        )
        raise AssertionError(
            f"Imports of packages that pyproject.toml does not declare:\n{listing}\n\n"
            "Add each one to [project.dependencies], to the extra that covers the "
            "importing code, or to a dependency group for test-only use. If the "
            "distribution is named differently from the module, add it to "
            "MODULE_TO_DISTRIBUTION. If nothing in pyproject.toml can install it, "
            "add it to EXTERNAL_MODULES."
        )
