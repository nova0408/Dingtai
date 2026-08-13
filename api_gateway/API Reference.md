# Dingtai API Gateway API Reference

文档版本：`1.7.0`（2026-08-11）
Gateway 服务版本：`0.5.2`
正式客户端入口：`https://<orin-ip>`（标准 HTTPS 端口 `443`）

Gateway 使用 aiohttp 直接终止 TLS 并监听外部客户端；默认绑定 `0.0.0.0:443`，同时在
同一端口监听 `[::]:443`，不需要额外部署一层反向代理。CameraPipeline、RecordReplay 和
RobotControl 仍是 Orin 内部服务，Gateway 通过 `127.0.0.1` 访问它们的内部端口。

所有正式客户端在首次使用前都必须安装并信任 `CasiaHand Root CA`。平台安装方法见
[`certificates/README.md`](certificates/README.md)。禁止使用 `verify=False`、`curl -k` 或
忽略浏览器证书告警。CasiaHand CA 只作为签发机构，不验证主机名或 IP；服务器证书 SAN 包含
当前 Orin hostname、`192.168.100.70` 和 `192.168.1.1–192.168.1.254`。正式客户端可以使用
其中任意一个实际转发到 Gateway 443 的 IP 地址，客户端按 URL 中的 IP 执行标准 IP SAN 校验。
重新签发服务器证书不会改变 Root CA，客户端不需要重新安装同一个 Root CA。

CameraPipeline WebSocket 的单条 CPWS1 消息上限为 16 MiB，足以承载当前 RGBD 帧；超过
该上限的消息会被 Gateway 拒绝。

机器可读契约见 [`openapi.yaml`](openapi.yaml)。

## 1. 访问边界

API Gateway 是三个功能性服务的正式客户端入口：

- CameraPipeline
- RecordReplay
- RobotControl

正式客户端、GUI 和集成程序必须使用 Gateway 的 HTTPS `443` 端口，并通过 URL 前缀区分后端：

| 正式客户端 URL 前缀 | 后端服务 | 后端独立端口 | 后端路径映射 |
| --- | --- | ---: | --- |
| `/api/v1/camera/*` | CameraPipeline HTTP | 6400 | 去掉 `/api/v1/camera` 后转发 |
| `/api/v1/camera-ws/*` | CameraPipeline WebSocket | 6401 | 去掉 `/api/v1/camera-ws` 后转发 |
| `/api/v1/record-replay/*` | RecordReplay HTTP | 6300 | 去掉 `/api/v1/record-replay` 后转发 |
| `/api/v1/record-replay-ws` | RecordReplay 状态 WebSocket | 6301 | 转发到 `/api/v1/ws` |
| `/api/v1/robot-control/*` | RobotControl HTTP | 6500 | 去掉 `/api/v1/robot-control` 后转发 |

三个功能性服务的独立端口只允许用于人工测试、Orin 本地只读诊断和故障排查，不得作为
GUI 或其它正式客户端的默认访问地址。Gateway 只统一客户端入口和 URL 前缀，不合并
后端进程，也不改变后端 API、状态语义或 RobotControl/RecordReplay 的安全边界。

Orin 本机的服务间访问和只读诊断必须直接使用内部端口，不经过 Gateway：

| Orin 本机服务 | 直接访问地址 |
| --- | --- |
| CameraPipeline HTTP | `http://127.0.0.1:6400` |
| CameraPipeline WebSocket | `ws://127.0.0.1:6401` |
| RecordReplay HTTP | `http://127.0.0.1:6300` |
| RecordReplay 状态 WebSocket | `ws://127.0.0.1:6301` |
| RobotControl HTTP | `http://127.0.0.1:6500` |

Gateway 的 `443` 和 `/api/v1/*` 前缀只面向外部客户端。部署脚本对 Gateway 自身的健康检查
可以访问 Gateway 的 `443`，但不能据此改变 Orin 本机访问后端服务的边界。

Gateway 为每个 HTTP 请求生成 `X-Request-ID` 并写入访问日志；上游响应本身带请求 ID 时，
Gateway 另以 `X-Upstream-Request-ID` 保留，便于关联两层日志。后端返回的非 2xx JSON 会保持
原样；Gateway 自身的路由、CORS、会话和上游连接错误统一返回
`{"error_code": "gateway_*", "error_text": "...；request_id=..."}`。其中 `502` 表示无法连接
上游，`503` 表示 Gateway 客户端会话尚未就绪，`500` 表示 Gateway 未预期异常；完整异常堆栈
保留在 Gateway 日志中。

## 2. Gateway 接口

### 2.1 健康检查

```http
GET /api/v1/gateway/health
```

该接口只检查 Gateway 自身是否能够响应，不探测三个后端服务、相机或机器人。

示例响应：

```json
{
  "service_version": "0.5.2",
  "gateway_version": "0.5.2",
  "backend_ports": {
    "camera_http": 6400,
    "camera_websocket": 6401,
    "record_replay": 6300,
    "record_replay_websocket": 6301,
    "robot_control": 6500,
    "calibration": 6600
  },
  "backend_probe": false
}
```

后端服务健康检查必须通过对应的 Gateway 前缀执行，例如：

```http
GET /api/v1/camera/health
GET /api/v1/record-replay/status
GET /api/v1/record-replay/health
GET /api/v1/robot-control/health
GET /api/v1/calibration/health
```

### 2.2 RobotControl AR5 七轴软限位

Gateway 会原样转发 RobotControl 的 AR5 软限位只读接口。正式客户端使用统一 Gateway 前缀：

```http
GET /api/v1/robot-control/ar5/{side}/soft-limits
```

`side` 必须为 `left` 或 `right`。例如读取右臂：

```http
GET https://<orin-ip>/api/v1/robot-control/ar5/right/soft-limits
Accept: application/json
```

成功响应包含七个轴的软限位上下限，单位为 rad：

```json
{
  "side": "right",
  "enabled": true,
  "axis_count": 7,
  "limits_rad": [
    {"axis_index": 0, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 1, "lower_rad": -2.0, "upper_rad": 2.0},
    {"axis_index": 2, "lower_rad": -2.0, "upper_rad": 2.0},
    {"axis_index": 3, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 4, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 5, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 6, "lower_rad": -3.14, "upper_rad": 3.14}
  ]
}
```

该接口只读，不改变 AR5 电源、工作模式或拖动状态。RobotControl 返回 `503` 时，Gateway
保留后端的结构化错误响应，包括 `error`、`message`、`path` 和 `stage` 字段。

## 3. 客户端配置

统一入口地址作为所有正式客户端的公共 base URL，再为 typed client 配置服务前缀：

```python
from camera_pipeline.service.http_client import CameraPipelineHttpClient
from record_replay.service.client import RecordReplayClient
from robot_control.service.client import RobotControlClient

camera = CameraPipelineHttpClient(
    "https://<orin-ip>",
    websocket_url="wss://<orin-ip>",
    api_prefix="/api/v1/camera",
    websocket_prefix="/api/v1/camera-ws",
)
replay = RecordReplayClient(
    "https://<orin-ip>",
    api_prefix="/api/v1/record-replay",
)
# 状态订阅：wss://<orin-ip>/api/v1/record-replay-ws
robot = RobotControlClient(
    "https://<orin-ip>",
    api_prefix="/api/v1/robot-control",
)
```

RobotControl 的控制 POST 和 RecordReplay 的 `/start` 仍必须由现场人员手动发起；Gateway
不会把正式客户端入口变成自动控制授权。

## 4. 文档变更记录

| 文档版本 | 日期 | 内容 |
| --- | --- | --- |
| `1.7.0` | 2026-08-11 | Gateway 错误统一为带 request_id 的 JSON，并补充关键链路日志约定。 |
| `1.5.0` | 2026-08-07 | 补充 RobotControl AR5 七轴软限位读取接口的统一 Gateway 访问路径和响应契约。 |
| `1.4.0` | 2026-08-07 | 增加 RecordReplay 状态 WebSocket 的 WSS 统一入口。 |
| `1.3.0` | 2026-08-04 | Gateway 默认同时监听 IPv4 `0.0.0.0:443` 与 IPv6 `[::]:443`；后端 loopback、证书和 CORS 约束不变。 |
| `1.2.0` | 2026-08-03 | 修复 OpenSSL 1.1.1f CA 扩展兼容性，增加 Orin 一键注册脚本 |
| `1.1.0` | 2026-08-03 | Gateway 由 aiohttp 直接提供 443/TLS；增加 CasiaHand CA 安装前置要求 |
| `1.0.0` | 2026-08-03 | 明确三个功能性服务正式访问 Gateway，独立端口仅用于测试和诊断 |
