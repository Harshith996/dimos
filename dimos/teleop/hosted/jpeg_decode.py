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

"""A JPEG camera stream as raw frames for ``CameraMuxModule``, decoded scaled down and rate-capped."""

from __future__ import annotations

import threading
import time
from typing import Any

from reactivex.disposable import Disposable
from turbojpeg import TJPF_RGB, TurboJPEG

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.CompressedImage import CompressedImage
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat


class JpegDecodeConfig(ModuleConfig):
    max_fps: float = 15.0
    # TurboJPEG decodes straight to 1/scale size (1, 2, 4 or 8), far cheaper than full size.
    scale: int = 4


class JpegDecodeModule(Module):
    """``jpeg_in`` → ``image_out``; frames arriving sooner than 1 / max_fps are dropped undecoded."""

    config: JpegDecodeConfig

    jpeg_in: In[CompressedImage]
    image_out: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._jpeg = TurboJPEG()
        self._next_t = 0.0
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.jpeg_in.subscribe(self._on_jpeg)))

    def _on_jpeg(self, msg: CompressedImage) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_t:
                return
            self._next_t = now + 1.0 / self.config.max_fps
        rgb = self._jpeg.decode(
            msg.data, pixel_format=TJPF_RGB, scaling_factor=(1, self.config.scale)
        )
        self.image_out.publish(
            Image(data=rgb, format=ImageFormat.RGB, frame_id=msg.frame_id, ts=msg.ts)
        )
