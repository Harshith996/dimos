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

"""Export saved episodes to a LeRobot dataset while a collection session runs.

The recorder keeps appending to the session database; this module watches the
episode status stream and, after each saved episode, rebuilds the LeRobot
dataset from everything saved so far. The build lands in a sibling directory
and is swapped into place, so the dataset path always holds a complete dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.imitation.collection.episode import (
    DATASET_SUFFIX_MAX_LENGTH,
    EpisodeStatus,
    ExportState,
    ExportStatus,
    normalize_dataset_suffix,
)
from dimos.imitation.dataprep.build import run_dataprep
from dimos.imitation.dataprep.schema import DataPrepConfig, OutputConfig
from dimos.msgs.std_msgs.String import String
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# A long session can take minutes to re-encode; shutdown waits this long for
# an export already in flight before giving up on it.
EXPORT_JOIN_TIMEOUT_S = 600.0
EXPORT_READY_TIMEOUT_S = 10.0
EXPORT_RETRY_INTERVAL_S = 0.25
BUILDING_SUFFIX = ".building"
PREVIOUS_SUFFIX = ".previous"


class LeRobotExportConfig(ModuleConfig):
    # The session database the recorder writes.
    source: str | Path
    # DataPrepConfig JSON declaring features, sync, and quality rules.
    dataprep_config: str | Path
    # LeRobot dataset directory to (re)build.
    dataset_path: str | Path
    # Optional run label appended to dataset_path, e.g. ``_pick-red-cube``.
    dataset_suffix: str | None = Field(
        default=None,
        max_length=DATASET_SUFFIX_MAX_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    # Rebuild after every saved episode; False exports once at shutdown.
    export_on_save: bool = True


class LeRobotExportModule(Module):
    """Rebuild the LeRobot dataset after each saved episode and report progress."""

    config: LeRobotExportConfig

    # Own process: decoding every recorded frame and encoding MP4s must not
    # share a GIL with control or recording.
    dedicated_worker = True

    status: In[String]
    dataset_suffix: In[String]
    export_status: Out[String]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._rerun_requested = False
        self._saved = 0
        self._exported_saves = 0
        self._runtime_dataset_suffix = self.config.dataset_suffix

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.status.subscribe(self._on_status)))
        if (
            self.dataset_suffix.connection is not None
            or self.dataset_suffix._transport is not None
        ):
            self.register_disposable(
                Disposable(self.dataset_suffix.subscribe(self._on_dataset_suffix))
            )
        self._publish("idle")

    @rpc
    def stop(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=EXPORT_JOIN_TIMEOUT_S)
        # The recorder has flushed by now, so one last build covers anything a
        # failed or disabled on-save export left out.
        with self._lock:
            due = self._saved > self._exported_saves
        if due:
            self._export_once()
        super().stop()

    def _on_status(self, message: String) -> None:
        try:
            status = EpisodeStatus.from_json(message.data)
        except ValueError:
            logger.warning("Ignoring invalid episode status JSON", exc_info=True)
            return
        with self._lock:
            self._saved = status.episodes_saved
            if status.last_event != "save" or not self.config.export_on_save:
                return
            if self._thread is not None and self._thread.is_alive():
                # A save landed mid-export: run once more when it finishes.
                self._rerun_requested = True
                return
            self._thread = threading.Thread(
                target=self._export_loop, name="LeRobotExport", daemon=True
            )
            self._thread.start()

    def _on_dataset_suffix(self, message: String) -> None:
        suffix = normalize_dataset_suffix(message.data)
        if not suffix or suffix != message.data:
            logger.warning("Ignoring invalid dataset suffix", suffix=message.data)
            return
        with self._lock:
            if self._saved > 0 or (self._thread is not None and self._thread.is_alive()):
                logger.warning("Ignoring dataset suffix change after collection began")
                return
            self._runtime_dataset_suffix = suffix
        logger.info("Dataset suffix selected", suffix=suffix)
        self._publish("idle")

    def _export_loop(self) -> None:
        while True:
            self._export_once()
            with self._lock:
                if not self._rerun_requested:
                    self._thread = None
                    return
                self._rerun_requested = False

    def _export_once(self) -> None:
        with self._lock:
            saved = self._saved
        if saved == 0:
            return
        dataset = self._dataset_path()
        building = dataset.with_name(dataset.name + BUILDING_SUFFIX)
        self._publish("exporting")
        config = DataPrepConfig.model_validate_json(Path(self.config.dataprep_config).read_text())
        config = config.model_copy(
            update={
                "source": str(self.config.source),
                "output": OutputConfig(
                    format="lerobot", path=building, metadata=config.output.metadata
                ),
            }
        )
        deadline = time.monotonic() + EXPORT_READY_TIMEOUT_S
        while True:
            shutil.rmtree(building, ignore_errors=True)
            try:
                run_dataprep(config)
                episodes = _swap_in(building, dataset)
                break
            except RuntimeError as exc:
                shutil.rmtree(building, ignore_errors=True)
                if time.monotonic() < deadline:
                    logger.info(
                        "LeRobot export waiting for recorder to flush",
                        dataset=str(dataset),
                        error=str(exc),
                    )
                    threading.Event().wait(EXPORT_RETRY_INTERVAL_S)
                    continue
                logger.exception("LeRobot export failed", dataset=str(dataset))
                self._publish("failed", error=str(exc))
                return
            except Exception as exc:
                logger.exception("LeRobot export failed", dataset=str(dataset))
                shutil.rmtree(building, ignore_errors=True)
                self._publish("failed", error=str(exc))
                return
        with self._lock:
            self._exported_saves = max(self._exported_saves, saved)
        logger.info("LeRobot dataset exported", dataset=str(dataset), episodes=episodes)
        self._publish("done", episodes=episodes)

    def _publish(
        self, state: ExportState, *, episodes: int | None = None, error: str | None = None
    ) -> None:
        with self._lock:
            exported = self._exported_saves
        status = ExportStatus(
            ts=time.time(),
            state=state,
            dataset=str(self._dataset_path()),
            episodes=exported if episodes is None else episodes,
            error=error,
        )
        self.export_status.publish(String(status.to_json()))

    def _dataset_path(self) -> Path:
        dataset = Path(self.config.dataset_path)
        if self._runtime_dataset_suffix:
            return dataset.with_name(f"{dataset.name}_{self._runtime_dataset_suffix}")
        return dataset


def _swap_in(building: Path, dataset: Path) -> int:
    """Replace `dataset` with the freshly built directory and return its episode count."""
    info = json.loads((building / "meta" / "info.json").read_text())
    previous = dataset.with_name(dataset.name + PREVIOUS_SUFFIX)
    shutil.rmtree(previous, ignore_errors=True)
    if dataset.exists():
        dataset.rename(previous)
    building.rename(dataset)
    shutil.rmtree(previous, ignore_errors=True)
    return int(info["total_episodes"])
