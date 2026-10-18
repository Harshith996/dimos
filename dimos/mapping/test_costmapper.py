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

from dataclasses import replace
import pickle

import pytest

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.mapping.costmapper import Config, CostMapper
from dimos.mapping.pointclouds.occupancy import (
    GeneralOccupancyConfig,
    HeightCostConfig,
    SimpleOccupancyConfig,
)


def test_default_blueprint_reconstructs_height_cost_settings():
    parsed = BlueprintConfigParser(CostMapper.blueprint()).parse(environ={})

    config = Config(**parsed.module_kwargs("costmapper"))

    assert config.config == HeightCostConfig()


@pytest.mark.parametrize(
    "algo,settings",
    [
        ("height_cost", HeightCostConfig(can_pass_under=1.4, can_climb=0.1)),
        ("general", GeneralOccupancyConfig(mark_free_radius=0.0)),
        ("simple", SimpleOccupancyConfig(closing_iterations=3)),
    ],
)
def test_algorithm_settings_survive_blueprint_worker_serialization(algo, settings):
    blueprint = CostMapper.blueprint(algo=algo, config=settings)
    parsed = BlueprintConfigParser(blueprint).parse(
        environ={}, overrides={"costmapper": {"config": {"resolution": 0.1}}}
    )

    kwargs = pickle.loads(pickle.dumps(parsed.module_kwargs("costmapper")))
    config = Config(**kwargs)

    assert config.algo == algo
    assert type(config.config) is type(settings)
    assert config.config == replace(settings, resolution=0.1)
