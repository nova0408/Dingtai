# CameraPipeline Service

## 单一职责

`camera_pipeline.service` 是 CameraPipeline 唯一的网络服务子模块，负责把协议请求桥接到 `PipelineContext` 和纯计算算法。算法目录不创建 socket、不监听端口，也不管理服务进程。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 服务监听、帧发布地址和请求循环超时配置 |
| `protocol.py` | 统一请求、响应、operation 和协议版本 |
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

## 协议与安全边界

请求包含 `protocol_version`，当前版本为 `1`。客户端与服务端版本不一致时服务端明确拒绝。

当前 REQ/REP 和帧流使用 Python pickle，仅允许在受信任的 Orin/开发内网使用。不得把端口直接暴露到不可信网络；pickle 也不适合作为跨语言公共协议。

## 错误约定

业务异常由 `CameraPipelineServer` 捕获，写入统一响应的 `error` 字段。客户端发现顶层错误或目标 payload 缺失时抛出 `RuntimeError`。算法模块不负责网络错误转换。

## 测试边界

无真实设备时可以验证协议、loopback RPC、请求路由、稳定帧和资源释放。真实相机连通性、模型性能、发布帧率和现场稳定阈值必须在 Orin 上另行验证。
