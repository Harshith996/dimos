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

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from experimental.gateway import catalog, introspect, responses
from experimental.gateway.state import ApiError, GatewayState, State

router = APIRouter(prefix="/catalog")
# imports dimos in a child process: seconds
SLOW = {"x-slow": True}
MAX_ERRORS = 200


def check_name(name: str) -> None:
    if not name or name.startswith("-") or any(c.isspace() for c in name):
        raise ApiError(400, f"bad name: {name}")


def picked(answer: dict[str, Any], fields: str | None) -> dict[str, Any]:
    try:
        return catalog.pick(answer, fields)
    except catalog.FieldError as error:
        raise ApiError(400, str(error))


async def listed(state: State) -> list[dict[str, Any]]:
    return state.listed or (await asyncio.to_thread(introspect.blueprint_list))["blueprints"]


async def blueprint_list(state: State) -> list[dict[str, Any]]:
    entries = await listed(state)
    notes = await asyncio.to_thread(
        catalog.notes_for, state.dimos_dir, [e["name"] for e in entries]
    )
    return catalog.blueprints(entries, state.scanner.cache(), notes)


async def find_blueprint(state: State, name: str) -> dict[str, Any]:
    check_name(name)
    found = next((b for b in await blueprint_list(state) if b["name"] == name), None)
    if found is None:
        raise ApiError(404, f"no blueprint {name}")
    return found


async def scan(state: State) -> dict[str, Any]:
    return await asyncio.to_thread(state.scanner.snapshot)


@router.get("", response_model=responses.Catalog)
async def everything(
    state: GatewayState,
    module: str | None = None,
    blueprint: str | None = None,
    fields: str | None = None,
) -> dict[str, Any]:
    """Every blueprint, module, skill and robot from the last scan, with how stale it is.

    ?module= (a name or class) keeps that module, its skills and the blueprints using it; ?blueprint= keeps that
    blueprint, its modules and their skills.
    """
    entries = await listed(state)
    notes = await asyncio.to_thread(
        catalog.notes_for, state.dimos_dir, [e["name"] for e in entries]
    )
    cache = state.scanner.cache()
    blueprints, modules = catalog.blueprints(entries, cache, notes), catalog.modules(cache)
    if blueprint is not None:
        blueprints = [b for b in blueprints if b["name"] == blueprint]
        names = {name for b in blueprints for name in b["modules"]}
        modules = [m for m in modules if m["name"] in names]
    if module is not None:
        modules = [m for m in modules if module in (m["name"], m["class"])]
        names = {m["name"] for m in modules}
        blueprints = [b for b in blueprints if names & set(b["modules"])]
    kept = {m["name"] for m in modules}
    answer = {
        "blueprints": blueprints,
        "modules": modules,
        "skills": [s for s in catalog.skills(cache) if s["module"] in kept],
        "robots": catalog.robot_list(notes),
        "types": notes.get("types", {}),
        "errors": catalog.errors(cache, MAX_ERRORS),
        "scan": await scan(state),
    }
    return picked(answer, fields)


@router.post("/scan", status_code=202, response_model=responses.ScanStarted)
async def scan_all(state: GatewayState) -> dict[str, Any]:
    """Scan every blueprint and module again (or join the running scan); the cache changes when it finishes. Follow it at /jobs/{job}."""
    return {"job": state.scanner.start("requested").id}


@router.post("/scan/blueprints/{name}", status_code=202, response_model=responses.ScanStarted)
async def scan_blueprint(state: GatewayState, name: str) -> dict[str, Any]:
    """Scan just this blueprint and its modules, and update their entries in the cache (or join the running full scan)."""
    await find_blueprint(state, name)
    return {"job": state.scanner.start_one(name).id}


@router.get("/blueprints", response_model=responses.BlueprintList)
async def blueprints(state: GatewayState, fields: str | None = None) -> dict[str, Any]:
    """Every blueprint, including external ones and ones that don't import, with what the last scan found."""
    return picked({"blueprints": await blueprint_list(state), "scan": await scan(state)}, fields)


@router.get("/blueprints/{name}", response_model=responses.BlueprintDetail, openapi_extra=SLOW)
async def blueprint(state: GatewayState, name: str, fields: str | None = None) -> dict[str, Any]:
    """A blueprint, plus its file and line and its modules' streams, RPCs and skills."""
    found = await find_blueprint(state, name)
    wanted = {f.strip().split(".")[0] for f in fields.split(",")} if fields else None
    if wanted is None or wanted & {"file", "line", "modules"}:
        found.update(await state.introspected(["blueprint", name]))
        found.pop("config", None)
    return picked(found, fields)


@router.get(
    "/blueprints/{name}/config", response_model=responses.BlueprintSchema, openapi_extra=SLOW
)
async def blueprint_config(state: GatewayState, name: str) -> Any:
    """The blueprint's config schema: GlobalConfig's fields and each module's, with types, defaults and choices."""
    await find_blueprint(state, name)
    return await state.introspected(["config", name])


async def find_module(state: State, name: str) -> dict[str, Any]:
    check_name(name)
    found = next(
        (m for m in catalog.modules(state.scanner.cache()) if name in (m["name"], m["class"])), None
    )
    if found is None:
        raise ApiError(404, f"no module {name} in the last scan")
    return found


async def importable_module(state: State, name: str) -> dict[str, Any]:
    found = await find_module(state, name)
    if "<locals>" in found["class"]:
        raise ApiError(
            404, f"{found['class']} is defined inside a function, so it can't be loaded on its own"
        )
    return found


@router.get("/modules", response_model=responses.ModuleList)
async def modules(state: GatewayState, fields: str | None = None) -> dict[str, Any]:
    return picked(
        {"modules": catalog.modules(state.scanner.cache()), "scan": await scan(state)}, fields
    )


@router.get("/modules/{name}", response_model=responses.ModuleDetail, openapi_extra=SLOW)
async def module(state: GatewayState, name: str) -> dict[str, Any]:
    """A module, the blueprints using it, and its RPCs and skills (`name` is its registry name or class)."""
    found = await importable_module(state, name)
    detail = await state.introspected(["module", found["class"]])
    return {**found, **{k: detail[k] for k in ("rpcs", "skill_methods", "file", "line")}}


@router.get("/modules/{name}/config", response_model=responses.ModuleSchema, openapi_extra=SLOW)
async def module_config(state: GatewayState, name: str) -> Any:
    """The module's config fields with the code's defaults; saved values are per blueprint (/config/blueprints/{name})."""
    found = await importable_module(state, name)
    return await state.introspected(["module_config", found["class"]])


@router.get("/skills", response_model=responses.CatalogSkills)
async def skills(
    state: GatewayState, namespace: str | None = None, fields: str | None = None
) -> dict[str, Any]:
    """Every skill the last scan found; ?namespace= names them as that namespace would. The running blueprint's are at /runs/skills."""
    found = catalog.skills(state.scanner.cache(), namespace)
    return picked({"skills": found, "scan": await scan(state)}, fields)


@router.get("/skills/{module}/{skill}", response_model=responses.CatalogSkill)
async def skill(
    state: GatewayState, module: str, skill: str, namespace: str | None = None
) -> dict[str, Any]:
    wanted = f"{namespace}/{module}" if namespace else module
    for found in catalog.skills(state.scanner.cache(), namespace):
        if found["name"] == skill and found["module"] in (module, wanted):
            return found
    raise ApiError(404, f"no skill {module}/{skill} in the last scan")


@router.get("/robots", response_model=responses.Robots)
async def robots(state: GatewayState, fields: str | None = None) -> dict[str, Any]:
    """Every robot dimos supports, by type, with its blueprints by name."""
    entries = await listed(state)
    notes = await asyncio.to_thread(
        catalog.notes_for, state.dimos_dir, [e["name"] for e in entries]
    )
    return picked({"types": notes.get("types", {}), "robots": catalog.robot_list(notes)}, fields)


@router.get("/msgs/{name}", response_model=responses.MessageType)
async def message_type(name: str) -> dict[str, Any]:
    """A message type's definition, by the type name topics use (e.g. sensor_msgs.Image). Not filled in yet."""
    check_name(name)
    return {"name": name, "definition": None}
