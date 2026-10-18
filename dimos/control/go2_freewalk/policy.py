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

"""Go2 FREE/HIMLoco inference, independent of physics, transport and scheduling.

Adapted from Andrew's #4441 FreePolicy port; numerical contract from go2web
policy/src/policies/unitree/himloco.rs at 1ffe3ee6. Weights are not distributed here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import struct

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Layer = tuple[Array, Array]
JOINT_NAMES = tuple(
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")
)


@dataclass(frozen=True)
class Proprioception:
    angular_velocity: Array
    gravity: Array
    joint_position: Array
    joint_velocity: Array


def _mlp(layers: list[Layer], x: Array) -> Array:
    for w, b in layers[:-1]:
        x = x @ w + b
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    w, b = layers[-1]
    return x @ w + b


class _Reader:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.pos = 4

    def u32(self) -> int:
        if self.pos + 4 > len(self.blob):
            raise ValueError("truncated FREE header")
        value: int = struct.unpack_from("<I", self.blob, self.pos)[0]
        self.pos += 4
        return value

    def f32(self, n: int) -> Array:
        if self.pos + 4 * n > len(self.blob):
            raise ValueError("truncated FREE weights")
        values = np.frombuffer(self.blob, "<f4", n, self.pos).astype(np.float64)
        self.pos += 4 * n
        if not np.isfinite(values).all():
            raise ValueError("FREE weights contain non-finite values")
        return values

    def block(self, nin: int, output: int) -> list[Layer]:
        count = self.u32()
        if not 1 <= count <= 16:
            raise ValueError("invalid FREE layer count")
        layers = []
        for _ in range(count):
            inputs, outputs = self.u32(), self.u32()
            if inputs != nin or not 1 <= outputs <= 4096:
                raise ValueError("incompatible FREE layer dimensions")
            layers.append((self.f32(inputs * outputs).reshape(inputs, outputs), self.f32(outputs)))
            nin = outputs
        if nin != output:
            raise ValueError("incompatible FREE output dimension")
        return layers


@dataclass(frozen=True)
class _Band:
    encoder: list[Layer]
    actor: list[Layer]


class FreePolicy:
    """Pinned 6 x 45 observation, 12-action FREE v1 model (FL, FR, RL, RR)."""

    joint_names = JOINT_NAMES

    def __init__(self, blob: bytes) -> None:
        r = _Reader(blob)
        if blob[:4] != b"FREE" or r.u32() != 1:
            raise ValueError("expected a FREE v1 policy")
        dimensions = tuple(r.u32() for _ in range(5))
        if dimensions != (6, 45, 12, 3, 16):
            raise ValueError(f"unsupported FREE dimensions: {dimensions}")
        self.clip_obs, self.clip_act = (float(v) for v in r.f32(2))
        if min(self.clip_obs, self.clip_act) <= 0:
            raise ValueError("FREE clipping limits must be positive")
        self.ob_mean, self.ob_scale = r.f32(45), r.f32(45)
        self.act_mean, self.act_scale = r.f32(12), r.f32(12)
        self.default_pose, self.kp, self.kd = r.f32(12), r.f32(12), r.f32(12)
        if (self.kp < 0).any() or (self.kd < 0).any():
            raise ValueError("FREE motor gains must be nonnegative")
        self.band_limits = r.f32(6)
        count = r.u32()
        if not 1 <= count <= 4:
            raise ValueError("invalid FREE expert count")
        self.bands: dict[int, _Band] = {}
        for _ in range(count):
            kind = r.u32()
            if kind > 3 or kind in self.bands:
                raise ValueError("invalid or duplicate FREE expert")
            self.bands[kind] = _Band(r.block(270, 19), r.block(64, 12))
        if r.pos != len(blob):
            raise ValueError("unexpected data after FREE weights")
        self._history: deque[Array] = deque(maxlen=6)
        self._last_action = np.zeros(12)

    def reset(self) -> None:
        self._history.clear()
        self._last_action.fill(0)

    def band_for(self, command: Array) -> int:
        speed = max(abs(command[0]), abs(command[1]))
        want = (
            3
            if speed < 0.05 and abs(command[2]) > 0.05
            else 0
            if speed < 1
            else 1
            if speed < 5
            else 2
        )
        # This is go2web's expert selection, including its absent sprint -> fast rule.
        for kind in (want, max(want - 1, 0), 0):
            if kind in self.bands:
                return kind
        return next(iter(self.bands))

    def forward_for_kind(self, kind: int, history: Array) -> Array:
        """Raw actor output for upstream normalized-history reference vectors."""
        if history.shape != (270,) or not np.isfinite(history).all():
            raise ValueError("FREE history must contain 270 finite values")
        band = self.bands[kind]
        encoded = _mlp(band.encoder, history)
        velocity, latent = encoded[:3], encoded[3:]
        latent = latent / max(float(np.linalg.norm(latent)), 1e-12)
        return _mlp(band.actor, np.concatenate([history[:45], velocity, latent]))

    def act(self, obs: Proprioception, command: Array) -> Array:
        vectors = (
            command,
            obs.angular_velocity,
            obs.gravity,
            obs.joint_position,
            obs.joint_velocity,
        )
        if any(
            v.shape != (n,) or not np.isfinite(v).all()
            for v, n in zip(vectors, (3, 3, 3, 12, 12), strict=True)
        ):
            raise ValueError("invalid FREE observation or command")
        raw = np.concatenate([*vectors, self._last_action])
        frame = np.clip((raw - self.ob_mean) * self.ob_scale, -self.clip_obs, self.clip_obs)
        if not self._history:
            self._history.extend([frame] * 6)
        else:
            self._history.append(frame)
        action = self.forward_for_kind(
            self.band_for(command), np.concatenate(list(self._history)[::-1])
        )
        if not np.isfinite(action).all():
            raise ValueError("FREE inference produced non-finite motor targets")
        self._last_action = np.clip(action, -self.clip_act, self.clip_act)
        return self._last_action * self.act_scale + self.act_mean
