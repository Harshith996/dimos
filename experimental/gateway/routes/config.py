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
import json
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, JsonValue

from experimental.gateway import overrides as ov, responses, store
from experimental.gateway.routes.catalog import find_blueprint
from experimental.gateway.state import ApiError, GatewayState

router = APIRouter(prefix="/config")


class GlobalUpdate(BaseModel):
    overrides: dict[str, JsonValue]


class BlueprintUpdate(BaseModel):
    global_: dict[str, JsonValue] = Field(default_factory=dict, alias="global")
    modules: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)


async def global_config() -> dict[str, Any]:
    from dimos.core.global_config import GlobalConfig

    defaults = {
        key: json.loads(json.dumps(field.default, default=str))
        for key, field in GlobalConfig.model_fields.items()
        if field.default_factory is None
    }
    schema = GlobalConfig.model_json_schema()
    names = [*GlobalConfig.model_fields, *schema.get("properties", {})]
    secrets = sorted({key for key in names if ov.is_secret_name(key)})
    shown, _ = ov.redact(store.global_config_overrides(), {}, secrets)
    return {
        "schema": schema,
        "defaults": defaults,
        "overrides": shown,
        "secrets": secrets,
    }


def check_keys(values: dict[str, Any]) -> None:
    for key in values:
        if not re.fullmatch("[A-Za-z0-9_]+", key):
            raise ApiError(400, f"bad config key: {key}")


def shown_blueprint(name: str) -> dict[str, Any]:
    saved = store.blueprint_config(name)
    global_, modules = ov.redact(
        saved["global"], saved["modules"], ov.secret_paths(saved["global"], saved["modules"])
    )
    return {"name": name, "global": global_, "modules": modules}


def problems(global_: dict[str, Any]) -> list[dict[str, str]]:
    """The global values dimos would refuse, as {key, error}; module values are dimos run's to check at launch."""
    try:
        ov.check_global(ov.merge(store.global_config_overrides(), global_))
    except ValueError as error:
        return [{"key": "global", "error": str(error)}]
    return []


@router.get("/global", response_model=responses.GlobalConfig)
async def get_global() -> dict[str, Any]:
    """GlobalConfig's schema and defaults, and the values saved for every launch."""
    return await global_config()


@router.put("/global", response_model=responses.GlobalConfig)
async def put_global(update: GlobalUpdate) -> dict[str, Any]:
    check_keys(update.overrides)
    values = ov.keep_hidden(dict(update.overrides), store.global_config_overrides())
    ov.check_global(values)
    await asyncio.to_thread(store.set_global_config_overrides, values)
    return await global_config()


@router.get("/blueprints/{name}", response_model=responses.BlueprintConfigValues)
async def get_blueprint(state: GatewayState, name: str) -> dict[str, Any]:
    """The values saved for this blueprint: `global` (over /config/global) and each module's; the schema is at /catalog/blueprints/{name}/config."""
    await find_blueprint(state, name)
    return shown_blueprint(name)


@router.put("/blueprints/{name}", response_model=responses.BlueprintConfigValues)
async def put_blueprint(state: GatewayState, name: str, update: BlueprintUpdate) -> Any:
    """Replace the values saved for this blueprint; a 400 if a global value isn't one dimos takes (module values are checked by dimos run at launch)."""
    await find_blueprint(state, name)
    check_keys(update.global_)
    saved = store.blueprint_config(name)
    global_ = ov.keep_hidden(dict(update.global_), saved["global"])
    modules = {
        m: ov.keep_hidden(dict(f), saved["modules"].get(m, {})) for m, f in update.modules.items()
    }
    found = problems(global_)
    if found:
        message = "; ".join(f"{p['key']}: {p['error']}" for p in found)
        return JSONResponse({"error": message, "problems": found}, status_code=400)
    await asyncio.to_thread(
        store.set_blueprint_config, name, global_, ov.merge_modules({}, modules)
    )
    return shown_blueprint(name)
