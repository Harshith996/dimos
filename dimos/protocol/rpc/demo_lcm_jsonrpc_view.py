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

"""Show a local Twist request and return value in both experimental encodings."""

import argparse
import json
from typing import Literal

from rich.console import Console, Group
from rich.highlighter import JSONHighlighter
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.protocol.rpc.demo_lcm_jsonrpc import (
    METHOD,
    PREFIX,
    REQUEST_ID,
    pack,
    pack_response,
    split_frame,
    unpack,
    unpack_response,
)

SIDE_BY_SIDE_WIDTH = 140
HEX_ROW_BYTES = 8


def json_view(data: bytes) -> Text:
    text = Text(
        json.dumps(json.loads(data), indent=2, ensure_ascii=False), overflow="fold", no_wrap=False
    )
    JSONHighlighter().highlight(text)
    return text


def hex_view(data: bytes) -> Text:
    return Text(
        "\n".join(
            f"{offset:04x}  {data[offset : offset + HEX_ROW_BYTES].hex(' ')}"
            for offset in range(0, len(data), HEX_ROW_BYTES)
        ),
        style="yellow",
    )


def show_encodings(
    console: Console,
    wire_a: bytes,
    wire_b: bytes,
    kind: Literal["request", "response"],
) -> None:
    header, body = split_frame(wire_b, "binary")
    prefix = wire_b[: PREFIX.size]
    envelope = json.loads(wire_a)
    value = envelope["result"] if kind == "response" else envelope["params"]["args"][0]
    location = "result" if kind == "response" else "params.args[0]"
    console.print(f"The LCM descriptor is stored at {location}.")

    panel_a = Panel(
        Group(
            Text(f"Complete {kind}: {len(wire_a)} bytes", style="bold cyan"),
            json_view(wire_a),
            Text(
                f"\nLook at {location}.data: {len(body)} bytes → {len(value['data'])} base64 characters."
            ),
            Text("Receiver: read data → decode base64 → decode LCM."),
        ),
        title="A: Binary as a string inside JSON",
        border_style="cyan",
    )
    panel_b = Panel(
        Group(
            Text(
                f"Complete {kind}: {len(wire_b)} = {len(prefix)} + {len(header)} + {len(body)} bytes",
                style="bold green",
            ),
            Text("1. Prefix (hex):"),
            Text(prefix.hex(" "), style="yellow"),
            Text(f"   DJR1 + JSON header length {len(header)}"),
            Text("2. JSON header: no data field"),
            json_view(header),
            Text("3. Raw LCM bytes after JSON (left column: payload offset):"),
            hex_view(body),
            Text("Hex is for display only; the frame carries raw bytes."),
            Text("Receiver: split JSON and payload using the prefix → decode LCM."),
        ),
        title="B: Raw bytes after JSON",
        border_style="green",
    )
    if console.width >= SIDE_BY_SIDE_WIDTH:
        columns = Table.grid(expand=True, padding=(0, 1))
        columns.add_column(ratio=1, overflow="fold")
        columns.add_column(ratio=1, overflow="fold")
        columns.add_row(panel_a, panel_b)
        console.print(columns)
    else:
        console.print(panel_a)
        console.print(panel_b)


def show_values(console: Console, title: str, values: list[tuple[str, Twist]]) -> None:
    comparison = Table(title=title)
    comparison.add_column("Source")
    comparison.add_column("Linear velocity")
    comparison.add_column("Angular velocity")
    for label, value in values:
        comparison.add_row(label, str(value.linear.to_tuple()), str(value.angular.to_tuple()))
    console.print(comparison)


def show(message: Twist, console: Console) -> None:
    raw = message.lcm_encode()
    wire_a = pack(raw, message.msg_name, "base64")
    wire_b = pack(raw, message.msg_name, "binary")

    console.print("[bold]One Twist call, two encodings: argument and return value[/bold]")
    console.print(f"Linear = {message.linear.to_tuple()}, angular = {message.angular.to_tuple()}")
    console.print(f"The existing LCM codec produces {len(raw)} bytes. All data below is real.")
    console.print("JSON is indented for readability; byte counts use the compact wire format.")
    console.print("This simulates both endpoints locally; no network or robot is involved.\n")

    console.rule("1. Request: client → server")
    show_encodings(console, wire_a, wire_b, "request")
    recovered_a = unpack(wire_a, message.msg_name, "base64")
    recovered_b = unpack(wire_b, message.msg_name, "binary")
    decoded_a = Twist.lcm_decode(recovered_a)
    decoded_b = Twist.lcm_decode(recovered_b)
    assert recovered_a == recovered_b == raw
    assert decoded_a == decoded_b == message
    show_values(
        console,
        "Argument decoded by the server",
        [("Client argument", message), ("Server A", decoded_a), ("Server B", decoded_b)],
    )

    console.rule("2. Return value: server → client")
    console.print(f"For this example, {METHOD} returns the received Twist unchanged (echo).")
    console.print("The response uses result, with no method or params fields.")
    console.print(f"The client checks response.id == request.id == {REQUEST_ID!r} before decoding.")
    response_a = pack_response(decoded_a.lcm_encode(), decoded_a.msg_name, "base64")
    response_b = pack_response(decoded_b.lcm_encode(), decoded_b.msg_name, "binary")
    show_encodings(console, response_a, response_b, "response")
    returned_a = unpack_response(response_a, Twist.msg_name, "base64")
    returned_b = unpack_response(response_b, Twist.msg_name, "binary")
    client_a = Twist.lcm_decode(returned_a)
    client_b = Twist.lcm_decode(returned_b)
    assert returned_a == returned_b == raw
    assert client_a == decoded_a and client_b == decoded_b
    show_values(
        console,
        "Return value decoded by the client",
        [("Server return", decoded_a), ("Client A", client_a), ("Client B", client_b)],
    )
    console.print(
        "[bold green]PASS: Both formats preserve argument and return values and LCM bytes; "
        "response IDs match.[/bold green]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Twist argument and return value in two encodings; local demo only."
    )
    parser.add_argument(
        "--linear", nargs=3, type=float, default=(1.0, 0.0, 0.0), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--angular", nargs=3, type=float, default=(0.0, 0.0, 0.5), metavar=("X", "Y", "Z")
    )
    args = parser.parse_args()
    show(Twist(linear=args.linear, angular=args.angular), Console())


if __name__ == "__main__":
    main()
