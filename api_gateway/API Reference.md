# Dingtai API Gateway API Reference

文档版本：`1.2.0`（2026-08-03）
Gateway 服务版本：`0.2.1`
正式客户端入口：`https://<orin-host>`（标准 HTTPS 端口 `443`）

Gateway 使用 aiohttp 直接终止 TLS 并监听外部客户端；默认绑定 `0.0.0.0:443`，不需要
额外部署一层反向代理。CameraPipeline、RecordReplay 和
RobotControl 仍是 Orin 内部服务，Gateway 通过 `127.0.0.1` 访问它们的内部端口。

所有正式客户端在首次使用前都必须安装并信任 `CasiaHand Root CA`。平台安装方法见
[`certificates/README.md`](certificates/README.md)。禁止使用 `verify=False`、`curl -k` 或
忽略浏览器证书告警。CasiaHand CA 只作为签发机构，不验证主机名或 IP；服务器证书 SAN 只包含
Orin 的 hostname，不包含 IP 地址。签发和安装脚本只按该 hostname 校验证书链，不执行 IP 校验。
正式客户端必须使用该 hostname 访问 Gateway，不支持用 IP 地址替代 hostname。

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
| RobotControl HTTP | `http://127.0.0.1:6500` |

Gateway 的 `443` 和 `/api/v1/*` 前缀只面向外部客户端。部署脚本对 Gateway 自身的健康检查
可以访问 Gateway 的 `443`，但不能据此改变 Orin 本机访问后端服务的边界。

## 2. Gateway 接口

### 2.1 健康检查

```http
GET /api/v1/gateway/health
```

该接口只检查 Gateway 自身是否能够响应，不探测三个后端服务、相机或机器人。

示例响应：

```json
{
  "gateway_version": "0.2.1",
  "backend_ports": {
    "camera_http": 6400,
    "camera_websocket": 6401,
    "record_replay": 6300,
    "robot_control": 6500
  },
  "backend_probe": false
}
```

后端服务健康检查必须通过对应的 Gateway 前缀执行，例如：

```http
GET /api/v1/camera/health
GET /api/v1/record-replay/status
GET /api/v1/robot-control/health
```

## 3. 客户端配置

统一入口地址作为所有正式客户端的公共 base URL，再为 typed client 配置服务前缀：

```python
from camera_pipeline.service.http_client import CameraPipelineHttpClient
from record_replay.service.client import RecordReplayClient
from robot_control.service.client import RobotControlClient

camera = CameraPipelineHttpClient(
    "https://<orin-host>",
    websocket_url="wss://<orin-host>",
    api_prefix="/api/v1/camera",
    websocket_prefix="/api/v1/camera-ws",
)
replay = RecordReplayClient(
    "https://<orin-host>",
    api_prefix="/api/v1/record-replay",
)
robot = RobotControlClient(
    "https://<orin-host>",
    api_prefix="/api/v1/robot-control",
)
```

RobotControl 的控制 POST 和 RecordReplay 的 `/start` 仍必须由现场人员手动发起；Gateway
不会把正式客户端入口变成自动控制授权。

## 4. 文档变更记录

| 文档版本 | 日期 | 内容 |
| --- | --- | --- |
| `1.2.0` | 2026-08-03 | 修复 OpenSSL 1.1.1f CA 扩展兼容性，增加 Orin 一键注册脚本 |
| `1.1.0` | 2026-08-03 | Gateway 由 aiohttp 直接提供 443/TLS；增加 CasiaHand CA 安装前置要求 |
| `1.0.0` | 2026-08-03 | 明确三个功能性服务正式访问 Gateway，独立端口仅用于测试和诊断 |
