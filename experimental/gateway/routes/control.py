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

from fastapi import APIRouter

from experimental.gateway.control import ControlError, ControlRequest, ControlStatus
from experimental.gateway.state import ApiError, GatewayState

router = APIRouter()


@router.get("/control", response_model=ControlStatus)
async def control_status(state: GatewayState) -> ControlStatus:
    """Read the active motion controller's lifecycle and feedback health."""
    try:
        return await asyncio.to_thread(state.controls.status)
    except ControlError as error:
        raise ApiError(error.status, str(error)) from error
    except TimeoutError as error:
        raise ApiError(503, "Robot status timed out; its control state is unknown") from error


@router.post("/control/actions", response_model=ControlStatus)
async def control_action(state: GatewayState, request: ControlRequest) -> ControlStatus:
    """Request a lifecycle action for an explicitly identified run and controller."""
    try:
        return await asyncio.to_thread(state.controls.action, request)
    except ControlError as error:
        raise ApiError(error.status, str(error)) from error
    except TimeoutError as error:
        raise ApiError(
            503, "Action acknowledgement timed out. Check robot state before retrying"
        ) from error
    except (RuntimeError, ValueError) as error:
        raise ApiError(409, str(error)) from error
