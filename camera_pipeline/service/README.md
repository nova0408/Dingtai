# CameraPipeline Service

## 单一职责

`camera_pipeline.service` 是 CameraPipeline 唯一的网络服务子模块，负责把协议请求桥接到 `PipelineContext` 和纯计算算法。算法目录不创建 socket、不监听端口，也不管理服务进程。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 服务监听、帧发布地址和请求循环超时配置 |
| `protocol.py` | 统一请求、响应、operation 和协议版本 |
| `wire_codec.py` | 白名单协议对象、JSON 元数据和 NumPy 原始字节块编解码 |
| `transport.py` | ZMQ REQ/REP 收发与消息类型校验 |
| `client.py` | 外接开发机和 Orin 本地业务服务共用的客户端实现 |
| `frame_publisher.py` | 单线程发布 RGBD、彩色、深度最新帧 |
| `application.py` | 相机请求和 tray/opening/ball 业务调用编排 |
| `server.py` | 通用 REP 请求循环和统一异常边界 |
| `__main__.py` | 参数解析、信号处理、对象组装和资源释放 |

## 数据流

```text
CameraPipelineClient
  -> REQ/REP transport
  -> CameraPipelineServer
  -> CameraPipelineApplication
  -> PipelineContext.resolve_frame()
  -> pure algorithm service.compute()
  -> protocol response
```

opening 请求属于业务组合流程：application 使用同一个稳定帧先执行 tray，取得目标托盘 mask 后执行 opening。该组合不进入算法实现，也不进入网络 transport。

## 部署拓扑

服务端默认绑定：

```text
tcp://0.0.0.0:6200
```

Orin 本地业务服务使用公共入口：

```python
from camera_pipeline.client import CameraPipelineClient

client = CameraPipelineClient()  # 默认 tcp://127.0.0.1:6200
```

外接开发机显式指定 Orin 地址：

```python
client = CameraPipelineClient(service_addr="tcp://<orin-ip>:6200")
```

启动命令：

```bash
python -m camera_pipeline.service \
  --bind-addr tcp://0.0.0.0:6200 \
  --control-port 5570 \
  --stream-port 5562 \
  --camera-id LEFT \
  --camera-name left_hand_camera
```

`--stream-port/--camera-id/--camera-name` 指定默认算法相机的覆盖参数。
头部、胸腔和左臂运行时仍按端点表同时启动。

systemd 模板位于 `camera-pipeline.service`。Orin 本地其他业务服务应声明 `Requires=camera-pipeline.service` 和 `After=camera-pipeline.service`，并在启动后调用 `get_camera_status()` 判断首帧是否真正可用。

## 端口

- `6200`：统一请求响应服务。
- `6201`：完整 RGBD 帧。
- `6202`：彩色帧。
- `6203`：深度帧。

帧订阅响应返回服务端 bind 地址；客户端会将 `0.0.0.0` 或 `127.0.0.1` 替换为当前服务主机，因此同时支持外接开发机和 Orin loopback。

## 生命周期

`__main__.py` 按以下顺序释放资源：

1. 停止并关闭帧发布器。
2. 关闭 REP transport。
3. 关闭 `PipelineContext` 和相机运行时。

发布端口只在首次订阅时绑定。未订阅帧流时不会创建 PUB socket 或发布线程。
三个 XPUB 端口都使用以 `camera_name + NUL` 编码的 topic 前缀，因此头部、
胸腔和左臂可复用同一组发布端口，订阅端不会收到其他安装位的帧。
右臂 topic 和 client API 已保留，但当前端点标记为未连接，请求会明确返回服务错误。
发布器跟踪真实订阅与取消订阅事件，只编码当前存在订阅者的相机和帧类型。

## 协议与安全边界

请求包含 `protocol_version`，当前版本为 `3`。版本 2 增加了分相机内参请求和
帧 topic，版本 3 增加了分相机稳定帧请求。客户端与服务端版本不一致时服务端明确拒绝。

REQ/REP 和三路帧流使用同一显式二进制协议：固定头部携带 JSON 元数据长度，
元数据只允许白名单中的协议 dataclass、基础类型和元组，NumPy 图像与 mask 以
连续原始字节块附加，并在元数据中记录 `dtype`、`shape`、偏移和长度。协议不使用
Python pickle，不依赖 dataclass 的 Python 版本内存布局，可在 Windows Python
3.10+ 客户端与 Orin Python 3.8 服务之间互通。未知类型、未知协议标识、越界数组
和字段不匹配均会明确拒绝。

## 错误约定

请求解码失败和业务异常均由 `CameraPipelineServer` 捕获，写入统一响应的 `error`
字段；单个非法请求不会终止服务循环。客户端发现顶层错误或目标 payload 缺失时
抛出 `RuntimeError`。算法模块不负责网络错误转换。

## API Reference

全部公开 client API、请求/响应字段和错误边界已迁移到
[CameraPipeline API Reference](../API%20Reference.md)。

## 测试边界

无真实设备时可以验证协议、loopback RPC、请求路由、稳定帧和资源释放。真实相机连通性、模型性能、发布帧率和现场稳定阈值必须在 Orin 上另行验证。
