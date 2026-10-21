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

"""The /catalog answers: the scan cache joined with annotations.yaml, in one shape per kind of thing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experimental.gateway import robots

SCAN_FIELDS = ("importable", "import_error", "missing_module", "optional_dependency")


def notes_for(dimos_dir: Path, names: list[str]) -> dict[str, Any]:
    """annotations.yaml (the checkout's own when it has one), resolved against `names`."""
    checkout = dimos_dir / "experimental" / "gateway" / robots.SOURCE_FILE.name
    return robots.resolved(
        robots.load(checkout if checkout.exists() else robots.SOURCE_FILE), names
    )


def blueprints(
    listed: list[dict[str, Any]], cache: dict[str, Any] | None, notes: dict[str, Any]
) -> list[dict[str, Any]]:
    from dimos.robot.all_blueprints import all_blueprints

    results = (cache or {}).get("results", {})
    annotated = {
        name: blueprint
        for robot in notes["robots"].values()
        for name, blueprint in robot["blueprints"].items()
    }
    found = []
    for entry in listed:
        name = entry["name"]
        record, note = results.get(name) or {}, annotated.get(name) or {}
        ref = record.get("ref") or all_blueprints.get(name)
        found.append(
            {
                "name": name,
                "kind": entry.get("kind", "builtin"),
                "ref": ref,
                "robot": note.get("robot") or record.get("robot"),
                "doc": record.get("doc"),
                "title": note.get("title"),
                "description": note.get("description"),
                "starter": note.get("starter"),
                "hidden": note.get("hidden", robots.HIDDEN_NAME.search(name) is not None),
                "recommended_config": note.get("recommended_config", []),
                "modules": [m["module"] for m in record.get("modules") or []],
                **{key: record.get(key) for key in SCAN_FIELDS},
                "suggested_extras": record.get("suggested_extras") or [],
                "scanned_at": record.get("scanned_at"),
            }
        )
    return found


def modules(cache: dict[str, Any] | None) -> list[dict[str, Any]]:
    cache = cache or {}
    robots_of: dict[str, set[str]] = {}
    blueprints_of: dict[str, set[str]] = {}
    for record in cache.get("results", {}).values():
        for module in record.get("modules") or []:
            blueprints_of.setdefault(module["class"], set()).add(record["name"])
            if record.get("robot"):
                robots_of.setdefault(module["class"], set()).add(record["robot"])
    return [
        {
            **{k: v for k, v in record.items() if k not in ("kind", "rpcs")},
            "robots": sorted(robots_of.get(record["class"], ())),
            "rpcs": [r["name"] for r in record["rpcs"]],
            "blueprints": sorted(blueprints_of.get(record["class"], ())),
        }
        for record in sorted(cache.get("modules", {}).values(), key=lambda m: m["name"])
    ]


def rpcs(cache: dict[str, Any] | None, namespace: str | None = None) -> list[dict[str, Any]]:
    by_class = {m["class"]: m for m in modules(cache)}
    found = []
    for record in sorted((cache or {}).get("modules", {}).values(), key=lambda m: m["name"]):
        module = f"{namespace}/{record['name']}" if namespace else record["name"]
        found += [
            {
                **r,
                "module": module,
                "robots": by_class[record["class"]]["robots"],
                "rpc": f"{module}/{r['name']}",
            }
            for r in record["rpcs"]
        ]
    return found


def robot_list(notes: dict[str, Any]) -> dict[str, Any]:
    """Robots with their blueprints by name: the blueprints' own details live on the blueprint."""
    return {
        robot_id: {**robot, "blueprints": list(robot["blueprints"])}
        for robot_id, robot in notes["robots"].items()
    }


def errors(cache: dict[str, Any] | None) -> list[str]:
    cache = cache or {}
    found = [
        f"blueprint {r['name']}: {r['import_error']}"
        for r in cache.get("results", {}).values()
        if not r["importable"]
    ]
    return found + [e for e in cache.get("errors", []) if e.startswith("module ")]


class FieldError(ValueError):
    pass


def pick(answer: dict[str, Any], fields: str | None) -> dict[str, Any]:
    """Keep only the dot paths in `fields` (comma-separated), and `scan`; a path through a list or map applies to each element."""
    if not fields:
        return answer
    tree: dict[str, Any] = {"scan": True}
    for path in filter(None, (f.strip() for f in fields.split(","))):
        node = tree
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {}) if node.get(part) is not True else {}
        node[parts[-1]] = True
    picked: dict[str, Any] = _pick(answer, tree, "")
    return picked


def _pick(value: Any, tree: dict[str, Any] | bool, at: str) -> Any:
    if isinstance(tree, bool):
        return value
    if isinstance(value, list):
        return [_pick(item, tree, at) for item in value]
    if not isinstance(value, dict):
        raise FieldError(f"{at} has no fields")
    if (
        not any(key in value for key in tree)
        and value
        and all(isinstance(v, dict) for v in value.values())
    ):
        return {key: _pick(item, tree, at) for key, item in value.items()}
    unknown = [key for key in tree if key not in value and key != "scan"]
    if unknown:
        raise FieldError(f"no field {', '.join(f'{at}{k}' for k in unknown)}")
    return {key: _pick(value[key], sub, f"{at}{key}.") for key, sub in tree.items() if key in value}
