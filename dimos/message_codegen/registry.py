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

"""Compatibility imports backed by the published dimos-cdrgen package."""

from dimos_cdrgen.registry import (
    ABI as ABI,
    decode as decode,
    encode as encode,
    initialize as initialize,
    message_types as message_types,
    providers as providers,
    schema as schema,
    schema_roots as schema_roots,
)
