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

"""Native MCAP capture and immutable selection of accepted human segments."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from mcap.reader import make_reader
import numpy as np
from numpy.typing import NDArray

from dimos.imitation.collection.capture import EpisodeCapture
from dimos.imitation.collection.episode import EpisodeEvent, EpisodeStatus
from dimos.imitation.collection.profile import CollectionProfile
from dimos.imitation.collection.recording import RecordingSchema
from dimos.imitation.dataprep.core import inspect_episode_quality, iter_episode_samples
from dimos.imitation.dataprep.schema import (
    Episode as DataEpisode,
    FeatureSpec,
    QualityConfig,
    Sample,
    SyncConfig,
)
from dimos.imitation.dataprep.sources.mcap import open_recording
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.std_msgs.String import String

CONTRACT_KEYS = (
    "joint_names",
    "fps",
    "image_shape",
    "robot_type",
    "action_semantics",
    "observation_profile",
    "artifact_contract_sha256",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def encode_image(image: NDArray[np.uint8]) -> str:
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode observation image")
    return base64.b64encode(data).decode("ascii")


def decode_image(value: str) -> NDArray[np.uint8]:
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("invalid observation image")
    return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def collection_profile(contract: dict[str, Any]) -> CollectionProfile:
    names = contract["joint_names"]
    return CollectionProfile(
        name="openyam-learning-v2",
        robot_type=contract["robot_type"],
        observations={
            "observation.state": FeatureSpec(
                stream="state",
                field="position",
                dtype="float32",
                shape=(len(names),),
                names=names,
                message_type=JointState,
            ),
            "observation.images.wrist": FeatureSpec(
                stream="wrist",
                dtype="video",
                shape=tuple(contract["image_shape"]),
                names=["height", "width", "channels"],
                message_type=Image,
            ),
        },
        actions={
            "action": FeatureSpec(
                stream="action",
                field="position",
                dtype="float32",
                shape=(len(names),),
                names=names,
                message_type=JointState,
                source_kind="joint_position_updates",
            )
        },
        sync=SyncConfig(
            anchor="observation.images.wrist", rate_hz=contract["fps"], tolerance_ms=50
        ),
        quality=QualityConfig(
            mode="strict",
            min_source_rate_ratio=0.8,
            max_camera_gap_ms=100,
            max_alignment_error_ms=50,
        ),
    )


class Episode:
    """A finalized manifest is the commit point; partial files are never selected."""

    def __init__(self, root: Path, metadata: dict[str, Any]) -> None:
        self.directory = root / "episodes" / uuid4().hex
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = {"schema_version": 2, **metadata}
        self._capture = EpisodeCapture(
            self.directory,
            collection_profile(metadata),
            extra_inputs={"provenance": String},
            stream_codecs={"provenance": "json"},
            stream_timestamp_fields={"provenance": "state_ts"},
            stream_json_schemas={"provenance": {"type": "object"}},
        )
        self.frames = 0
        self.training_frames = 0
        self._last_timestamp = 0.0
        self._closed = False

    def _write(self, channel: str, message: Any) -> None:
        self._capture.write(channel, message)

    def _status(self, event: EpisodeEvent, timestamp: float, keep: bool = False) -> None:
        self._write(
            "status",
            String(
                EpisodeStatus(
                    ts=timestamp,
                    state="recording" if event == "start" else "idle",
                    episodes_saved=int(keep),
                    episodes_discarded=int(event == "discard"),
                    last_event=event,
                    task_label=self.metadata["task"],
                ).to_json()
            ),
        )

    def append(self, frame: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("episode is closed")
        names = self.metadata["joint_names"]
        # Use the timestamps preserved by the LCM headers. Their nanosecond
        # quantization must also define segment bounds in the provenance;
        # otherwise a boundary image can fall just outside its own segment.
        state_ts = Header(float(frame["state_ts"])).timestamp
        image_ts = Header(float(frame["image_ts"])).timestamp
        if self.frames == 0:
            self._status("start", min(state_ts, image_ts))
        self._last_timestamp = max(state_ts, image_ts)
        self._write("state", JointState(name=names, position=frame["state"], ts=frame["state_ts"]))
        self._write(
            "wrist",
            Image.from_numpy(
                decode_image(frame["image_png"]), format=ImageFormat.RGB, ts=frame["image_ts"]
            ),
        )
        if frame["applied"]:
            action_ts = frame.get("applied_ts", frame["command_ts"])
            self._write(
                "action", JointState(name=names, position=frame["accepted_action"], ts=action_ts)
            )
        provenance = {
            key: value
            for key, value in frame.items()
            if key not in ("state", "image_png", "accepted_action")
        }
        provenance.update(state_ts=state_ts, image_ts=image_ts)
        if frame["applied"]:
            provenance["applied_ts"] = Header(float(action_ts)).timestamp
        self._write("provenance", String(json.dumps(provenance, allow_nan=False)))
        self.frames += 1
        if frame["applied"] and frame["authority"] in ("human", "correction"):
            self.training_frames += 1

    def finish(self, *, keep: bool, outcome: str = "ungraded") -> Path:
        if self._closed:
            raise RuntimeError("episode already closed")
        self._closed = True
        try:
            self._status(
                "save" if keep else "discard", self._last_timestamp + 1 / self.metadata["fps"], keep
            )
            counts = self._capture.finish()
        finally:
            self._capture.close()
        final = self.directory / "recording.mcap"
        self.metadata.update(
            frames=self.frames,
            training_frames=self.training_frames,
            kept=keep,
            outcome=outcome,
            recorder="dimos.imitation.collection.CollectionRecorder",
            stream_counts=counts,
        )
        self.metadata["mcap_sha256"] = digest(final)
        self.metadata["schema_sha256"] = digest(self.directory / "schema.json")
        partial = self.directory / "metadata.json.partial"
        partial.write_text(json.dumps(self.metadata, indent=2) + "\n")
        partial.replace(self.directory / "metadata.json")
        return final


def read_episode(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(path.with_name("metadata.json").read_text())
    if metadata["schema_version"] != 2:
        raise ValueError("legacy episodes need an explicit coordinate migration")
    if (
        digest(path) != metadata["mcap_sha256"]
        or digest(path.with_name("schema.json")) != metadata["schema_sha256"]
    ):
        raise ValueError(f"episode changed after finalization: {path}")
    with path.open("rb") as stream:
        frames = [
            json.loads(message.data)
            for _, _, message in make_reader(stream).iter_messages(topics=["provenance"])
        ]
    return metadata, frames


def export_snapshot(
    root: Path, destination: Path, *, observation_profile: str | None = None
) -> Path:
    """Freeze raw input identities and select segments; conversion is an offline job."""
    if destination.exists():
        raise FileExistsError(f"snapshot already exists: {destination}")
    inputs: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    contract: dict[str, Any] | None = None
    for path in sorted((root / "episodes").glob("*/recording.mcap")):
        if not path.with_name("metadata.json").exists():
            continue
        if observation_profile is not None:
            metadata = json.loads(path.with_name("metadata.json").read_text())
            if metadata.get("observation_profile") != observation_profile:
                continue
        meta, frames = read_episode(path)
        if not meta["kept"] or meta["split"] != "train":
            continue
        current = {key: meta[key] for key in CONTRACT_KEYS}
        if contract is not None and current != contract:
            raise ValueError("cannot mix incompatible episode contracts")
        contract = current
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": meta["mcap_sha256"],
                "schema_sha256": meta["schema_sha256"],
            }
        )
        previous: dict[str, Any] | None = None
        segment: dict[str, Any] | None = None
        for frame in frames:
            eligible = frame["applied"] and frame["authority"] in ("human", "correction")
            age = frame.get("command_age", frame["state_ts"] - frame["command_ts"])
            eligible = eligible and 0 <= age <= 0.15
            if not eligible:
                previous = None
                continue
            contiguous = previous is not None and (
                frame["segment"] == previous["segment"]
                and frame["sim_generation"] == previous["sim_generation"]
                and 0.5 / meta["fps"]
                <= frame["image_ts"] - previous["image_ts"]
                <= 1.5 / meta["fps"]
            )
            if not contiguous:
                segment = {
                    "id": f"{path.parent.name}-{len(segments)}",
                    "path": str(path.resolve()),
                    "start_ts": max(
                        frame["image_ts"], frame.get("applied_ts", frame["command_ts"])
                    ),
                    "end_ts": frame["image_ts"],
                    "task": meta["task"],
                    "frames": 0,
                }
                segments.append(segment)
            assert segment is not None
            segment["end_ts"] = frame["image_ts"]
            segment["frames"] += 1
            previous = frame
    segments = [
        segment
        for segment in segments
        if segment["frames"] >= 2 and segment["end_ts"] > segment["start_ts"]
    ]
    # Tiny pieces left by a timing gap may contain only one aligned camera frame.
    # Freeze the quality decision here; never silently fill or join across that gap.
    qualified = []
    rejected = []
    for candidate in segments:
        path = Path(candidate["path"])
        schema = RecordingSchema.read(path.parent)
        store = open_recording(path)
        try:
            report = inspect_episode_quality(
                store,
                DataEpisode(
                    id=candidate["id"], start_ts=candidate["start_ts"], end_ts=candidate["end_ts"]
                ),
                {**schema.observation, **schema.action},
                schema.sync,
                schema.quality,
            )
        finally:
            store.stop()
        if report.emitted_frames < 2:
            report.valid = False
            report.rejection_reasons.append("fewer than two aligned frames")
        if report.valid:
            candidate["frames"] = report.emitted_frames
            qualified.append(candidate)
        else:
            rejected.append({"id": candidate["id"], "reasons": report.rejection_reasons})
    segments = qualified
    if not segments or contract is None:
        raise ValueError(
            "no contiguous saved human demonstrations/corrections in the training split"
        )
    destination.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        **contract,
        "frames": sum(s["frames"] for s in segments),
        "segments": len(segments),
        "episodes": inputs,
        "selection": segments,
        "rejected_segments": rejected,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def snapshot_samples(snapshot: Path) -> Iterator[Sample]:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for source in manifest["episodes"]:
        path = Path(source["path"])
        if (
            digest(path) != source["sha256"]
            or digest(path.with_name("schema.json")) != source["schema_sha256"]
        ):
            raise ValueError(f"snapshot source changed: {path}")
    for segment in manifest["selection"]:
        path = Path(segment["path"])
        schema = RecordingSchema.read(path.parent)
        episode = DataEpisode(
            id=segment["id"],
            start_ts=segment["start_ts"],
            end_ts=segment["end_ts"],
            task_label=segment["task"],
        )
        with ExitStack() as stack:
            store = open_recording(path)
            stack.callback(store.stop)
            quality = inspect_episode_quality(
                store, episode, {**schema.observation, **schema.action}, schema.sync, schema.quality
            )
            if not quality.valid:
                raise ValueError(
                    f"segment {episode.id} failed alignment: {quality.rejection_reasons}"
                )
            yield from iter_episode_samples(
                store,
                episode,
                {**schema.observation, **schema.action},
                schema.sync,
                schema.quality,
                set(schema.observation),
                set(schema.action),
            )
