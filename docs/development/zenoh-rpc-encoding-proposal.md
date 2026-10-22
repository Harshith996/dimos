# Zenoh RPC 编码方案

## 选什么方案

**继续用 Zenoh，把 RPC 的 pickle 换成 MessagePack，并复用现有 LCM 编码。**

Zenoh 负责收发，MessagePack 负责打包参数和结果，LCM 负责已有消息类型的编码。调用方仍然使用现有的 @rpc、@skill 和 Spec API。

客户端和服务端一起升级，不需要兼容旧 pickle 节点。本次范围是 Zenoh RPC；本地 worker IPC 和其他 transport 中的 pickle 单独处理。

## 怎么编码

| Payload | 处理方式 |
| --- | --- |
| 数字、字符串、bytes、list、普通 dict | 直接用 MessagePack |
| LCM 消息 | 保存类型标识和原有 LCM bytes，接收端调用对应 decoder |
| NumPy 数组 | 保存 dtype、shape 和原始 bytes |
| tuple、enum、dataclass 等 | 加类型标识，确保收到后恢复原来的 Python 类型 |
| RPCClient、Transport、任务和 Blueprint | 为这些特殊对象提供专用适配器 |

这些类型可以嵌套。例如一个 dict 里同时包含 PoseStamped、数组和字符串，encoder 递归处理每个值，最终打包成一份二进制数据。bytes 保持二进制，不转 base64。[MessagePack 规范](https://github.com/msgpack/msgpack/blob/master/spec.md)

自定义对象需要登记“如何编码、如何恢复”。不支持的类型明确报错，不回退到 pickle。**保留兼容性的标准是收到原来的类型和数据，而不只是一个内容相似的 dict。**

## 最简示例

以 `calculate(2, b=3)` 为例，下面用 msgpack-python 在内存中演示完整的编码往返。实际 RPC 中，Zenoh 负责传输 request 和 reply 这两份 bytes。

```python
import msgpack


def calculate(a: int, b: int) -> int:
    return a + b


# 客户端：把位置参数和关键字参数编码成二进制。
request = msgpack.packb({"args": [2], "kwargs": {"b": 3}}, use_bin_type=True)

# 服务端：解码参数，调用 calculate，再编码结果。
params = msgpack.unpackb(request, raw=False)
result = calculate(*params["args"], **params["kwargs"])
reply = msgpack.packb(result, use_bin_type=True)

# 客户端：解码得到原来的 Python 值。
answer = msgpack.unpackb(reply, raw=False)
assert answer == 5
print(answer)
```

```results
5
```

如果参数中包含 LCM 消息或自定义对象，就由统一 codec 在打包前调用对应适配器，解包时恢复类型；`calculate` 的调用方式不变。

### 参数和返回值包含 LCM 类型

这次调用 `calculate(Vector3(1, 2, 3), scale=2.0)`：Vector3 使用 LCM 编码，scale 使用 MessagePack 原生浮点编码，返回值仍然是 Vector3。

为保持示例最小，约定 MessagePack 扩展编号 1 表示 Vector3；扩展内容直接存放完整的 LCM bytes。

```python
from typing import Any

import msgpack

from dimos.msgs.geometry_msgs.Vector3 import Vector3


def calculate(vector: Vector3, scale: float) -> Vector3:
    return Vector3(vector.x * scale, vector.y * scale, vector.z * scale)


def encode_lcm(value: Any) -> msgpack.ExtType:
    if type(value) is Vector3:
        return msgpack.ExtType(1, value.lcm_encode())
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def decode_lcm(code: int, data: bytes) -> Vector3:
    if code == 1:
        return Vector3.lcm_decode(data)
    raise ValueError(f"Unknown extension: {code}")


# 客户端：MessagePack 自动对嵌套的 Vector3 调用 encode_lcm。
request = msgpack.packb(
    {"args": [Vector3(1, 2, 3)], "kwargs": {"scale": 2.0}},
    default=encode_lcm,
    use_bin_type=True,
)

# 服务端：先恢复 Vector3，再正常调用 calculate。
params = msgpack.unpackb(request, ext_hook=decode_lcm, raw=False)
assert type(params["args"][0]) is Vector3
result = calculate(*params["args"], **params["kwargs"])
reply = msgpack.packb(result, default=encode_lcm, use_bin_type=True)

# 客户端：返回值也恢复为 Vector3。
answer = msgpack.unpackb(reply, ext_hook=decode_lcm, raw=False)
assert type(answer) is Vector3
assert (answer.x, answer.y, answer.z) == (2.0, 4.0, 6.0)
print(answer.x, answer.y, answer.z)
```

```results
2.0 4.0 6.0
```

正式实现支持多种 LCM 类型时，在统一的 LCM 扩展中增加类型标识，通过注册表选择 decoder。上述 encode_lcm/decode_lcm 收进统一 codec，业务函数无需处理编码。

可运行的小脚本：[demo_rpc_encoding.py](/docs/development/examples/demo_rpc_encoding.py)。它在 LCM 扩展中加入类型名，并逐步打印编码、类型标识、服务端调用和结果恢复；收发在内存中模拟。

在仓库根目录执行 `uv run --no-sync --with msgpack python docs/development/examples/demo_rpc_encoding.py`。最终输出为 `Vector3(2.0, 4.0, 6.0)`。

## 需要改哪里

1. **新增统一 codec。** 提供 encode/decode，集中处理上述类型，首版采用 msgpack-python。请求、返回值和异常都经过它。
2. **接入现有 ZenohRPC。** 替换 [zenohrpc.py](/dimos/protocol/rpc/zenohrpc.py) 中的 pickle 调用，增加编码版本标识，保留现有调用方式和异常传播。
3. **补齐特殊 payload，再统一切换。** 优先处理模块接线需要的 RPCClient、Transport，然后覆盖业务对象、任务和 Blueprint。验证编码往返及蓝图启动、调用、停止。

当前代码有三个需要顺手解决的问题：

- [StreamDescriptor](/dimos/core/module.py) 内部还装着一份 pickle bytes，需要连同 [读取端](/dimos/porcelain/remote_module_source.py) 一起替换。
- [GraspCandidate](/dimos/msgs/manipulation_msgs/GraspCandidate.py) 的 encode 实际使用 pickle，需要专用适配，不能直接当成 LCM。
- Image、PointCloud2 走现有 LCM 编码可能改变时间戳、数组可写性或精度，需要单独验证并补齐适配。

任务和 Blueprint 包含行为与状态，不能只转成普通 dict；其适配完成前，不能宣称现有 payload 已全部兼容。

这是实现方案，尚未修改运行时。第一步先把 codec 和现有类型适配做好，再接入 Zenoh RPC。
