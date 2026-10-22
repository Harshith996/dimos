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

"""演示 MessagePack + LCM 的 RPC 数据往返, 在内存中模拟收发。

从仓库根目录运行:
    uv run --no-sync --with msgpack python docs/development/examples/demo_rpc_encoding.py
"""

from typing import Any

import msgpack  # type: ignore[import-untyped]

from dimos.msgs.geometry_msgs.Vector3 import Vector3

LCM_EXT = 1  # 示例约定: 这个扩展编号表示 LCM 消息。
VECTOR3_TYPE = "geometry_msgs.Vector3"


def calculate(vector: Vector3, scale: float) -> Vector3:
    """业务函数只处理解码后的 Python 值。"""
    return Vector3(vector.x * scale, vector.y * scale, vector.z * scale)


def encode_lcm(value: Any) -> msgpack.ExtType:
    """MessagePack 遇到 Vector3 时, 保存类型名和完整 LCM bytes。"""
    if type(value) is not Vector3:
        raise TypeError(f"Unsupported type: {type(value).__name__}")
    content = msgpack.packb({"type": VECTOR3_TYPE, "data": value.lcm_encode()}, use_bin_type=True)
    return msgpack.ExtType(LCM_EXT, content)


def decode_lcm(code: int, data: bytes) -> Vector3:
    """按双方约定的扩展编号和类型名选择 decoder。"""
    if code != LCM_EXT:
        raise ValueError(f"Unknown extension: {code}")
    content = msgpack.unpackb(data, raw=False)
    if content["type"] != VECTOR3_TYPE:
        raise ValueError(f"Unknown LCM type: {content['type']}")
    vector = Vector3.lcm_decode(content["data"])
    assert type(vector) is Vector3
    return vector


def pack(value: Any) -> bytes:
    payload: bytes = msgpack.packb(value, default=encode_lcm, use_bin_type=True)
    return payload


def unpack(payload: bytes) -> Any:
    return msgpack.unpackb(payload, ext_hook=decode_lcm, raw=False)


def handle_calculate(request: bytes) -> bytes:
    """服务端只接收和返回 bytes; 真实场景由 Zenoh 搬运。"""
    params = unpack(request)
    vector = params["args"][0]
    scale = params["kwargs"]["scale"]
    assert type(vector) is Vector3
    assert type(scale) is float
    print(f"3. 服务端解码: {type(vector).__name__}{vector.as_tuple}, scale={scale}")

    result = calculate(*params["args"], **params["kwargs"])
    print(f"4. 执行 calculate: Vector3{result.as_tuple}")
    reply = pack(result)
    print(f"5. 服务端编码返回值: {len(reply)} bytes")
    return reply


def main() -> None:
    print("内存演示: 真实 RPC 中由 Zenoh 传输 request 和 reply 两份 bytes。")
    print("1. 客户端调用: calculate(Vector3(1, 2, 3), scale=2.0)")
    request = pack({"args": [Vector3(1, 2, 3)], "kwargs": {"scale": 2.0}})
    print(f"2. 客户端编码请求: {len(request)} bytes")
    print(f"   请求的十六进制表示: {request.hex()}")

    # 暂时不使用 ext_hook, 观察普通 MessagePack decoder 能识别哪些信息。
    wire = msgpack.unpackb(request, raw=False)
    extension = wire["args"][0]
    assert isinstance(extension, msgpack.ExtType)
    content = msgpack.unpackb(extension.data, raw=False)
    print(f"   普通 MessagePack 解码: args[0] 是 ExtType, 编号={extension.code}")
    print(f"   扩展内部: type={content['type']}, data={len(content['data'])} LCM bytes")
    print(f"   普通参数: scale={wire['kwargs']['scale']}, 无需 LCM decoder")

    reply = handle_calculate(request)
    answer = unpack(reply)
    assert type(answer) is Vector3
    assert answer.as_tuple == (2.0, 4.0, 6.0)
    print(f"6. 客户端解码: {type(answer).__name__}{answer.as_tuple}")


if __name__ == "__main__":
    main()
