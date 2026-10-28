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

"""Fixed-rate loop waits that survive macOS timer coalescing."""

from threading import Event
import time

# macOS stretches a background process's timer by up to ~7x (wait(0.02) takes ~0.14 s), so
# wait a fraction of what remains and re-check; only the last tiny chunk overshoots.
_CHUNK_FRACTION = 1 / 8
_MIN_CHUNK = 0.0005


def wait_until(stop: Event, deadline: float) -> bool:
    """Wait until monotonic `deadline` or until `stop` is set; returns whether it was set."""
    while (remaining := deadline - time.monotonic()) > 0:
        if stop.wait(max(remaining * _CHUNK_FRACTION, _MIN_CHUNK)):
            return True
    return stop.is_set()
