# Copyright 2025-2026 Dimensional Inc.
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

import pytest

from dimos.core.coordination.module_coordinator import _verify_no_name_conflicts
from dimos.core.global_config import global_config
from dimos.robot.get_all_blueprints import get_blueprint_by_name


@pytest.mark.parametrize("viewer", ["none", "rerun"])
def test_go2_streams_share_wire_types(monkeypatch, viewer):
    monkeypatch.setattr(global_config, "viewer", viewer)
    _verify_no_name_conflicts(get_blueprint_by_name("unitree-go2"))
