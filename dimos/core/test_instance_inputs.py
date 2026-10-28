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

"""Instance inputs share the same contract during wiring and construction."""

import pickle

import pytest

from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.String import String


class ConfiguredInputs(Module):
    result: Out[String]


@pytest.fixture
def configured_module():
    instance = ConfiguredInputs(input_types={"image": Image, "label": String})
    yield instance
    instance.stop()


def test_instance_input_types_match_pickled_blueprint(configured_module):
    blueprint = ConfiguredInputs.blueprint(input_types={"image": Image, "label": String})
    restored = pickle.loads(pickle.dumps(blueprint))
    assert [
        (p.name, p.type, p.direction)
        for p in restored.active_blueprints[0].streams
        if p.name != "tf"
    ] == [
        ("result", String, "out"),
        ("image", Image, "in"),
        ("label", String, "in"),
    ]
    assert {name: port.type for name, port in configured_module.inputs.items() if name != "tf"} == {
        "image": Image,
        "label": String,
    }
    assert {port.name for port in configured_module.module_info().inputs} >= {"image", "label"}


@pytest.mark.parametrize("name", ["result", "rpc", "ref", "config", "_hidden", "bad-name", "class"])
@pytest.mark.parametrize("build", [ConfiguredInputs.blueprint, ConfiguredInputs])
def test_instance_inputs_cannot_shadow_module_attributes(name, build):
    with pytest.raises(ValueError, match="reserved"):
        build(input_types={name: String})


def test_instance_input_message_class_is_required():
    with pytest.raises(TypeError, match="message class"):
        ConfiguredInputs.blueprint(input_types={"image": "Image"})
