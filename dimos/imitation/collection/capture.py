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

"""One native MCAP capture, drained and checked before accepting an episode."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from mcap.reader import make_reader

from dimos.core.transport import ZenohTransport
from dimos.imitation.collection.profile import CollectionProfile
from dimos.imitation.collection.recorder import CollectionRecorder, collection_recorder
from dimos.msgs.std_msgs.String import String
from dimos.protocol.pubsub.impl.zenohpubsub import QOS_NEVER_DROP, Topic


class EpisodeCapture:
    def __init__(self, directory: Path, profile: CollectionProfile, **kwargs: Any) -> None:
        self.directory = directory
        self.counts: dict[str, int] = {}
        self._stack = ExitStack()
        self._closed = False
        self._transports: dict[str, ZenohTransport[Any]] = {}
        identity = uuid4().hex
        blueprint = collection_recorder(
            profile=profile,
            recording=directory,
            publish_status=True,
            rpc_name=f"recording-{identity}",
            **kwargs,
        )
        atom = blueprint.blueprints[0]
        self._recorder = cast("CollectionRecorder", atom.module(**atom.kwargs))
        try:
            ports = dict(self._recorder.recording_inputs)
            ports["recording_status"] = String
            for name, kind in ports.items():
                transport: ZenohTransport[Any] = ZenohTransport(
                    Topic(
                        f"recording/{identity}/{name}",
                        kind,
                        qos=None if name == "recording_status" else QOS_NEVER_DROP,
                    )
                )
                self._transports[name] = transport
                self._stack.callback(transport.stop)
                self._recorder.set_transport(name, transport)
            self._stack.callback(self._recorder.stop)
            self._recorder.build()
            self._recorder.start()
            self._recorder.wait_written({})
        except BaseException:
            self._stack.close()
            raise

    def write(self, stream: str, message: Any) -> None:
        if self._closed:
            raise RuntimeError("Episode capture is closed")
        self._transports[stream].broadcast(None, message)
        self.counts[stream] = self.counts.get(stream, 0) + 1

    def finish(self) -> dict[str, int]:
        if self._closed:
            raise RuntimeError("Episode capture is closed")
        try:
            self._recorder.wait_written(self.counts)
        finally:
            self.close()
        with (self.directory / "recording.mcap").open("rb") as source:
            summary = make_reader(source).get_summary()
            if summary is None or summary.statistics is None:
                raise RuntimeError("Native recording did not finalize its MCAP index")
            counts = {
                summary.channels[channel].topic: count
                for channel, count in summary.statistics.channel_message_counts.items()
            }
        if counts != self.counts:
            raise RuntimeError(f"Recording counts differ: expected {self.counts}, found {counts}")
        return counts

    def close(self) -> None:
        self._closed = True
        self._stack.close()
