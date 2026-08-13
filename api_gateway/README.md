# Dingtai 统一客户端入口

`api_gateway` 是五个业务服务之上的外部客户端统一入口。它使用 aiohttp 直接终止 TLS，默认
同时监听 IPv4 `0.0.0.0:443` 与 IPv6 `[::]:443`，不需要再部署一层反向代理。

完整访问契约见 [`API Reference.md`](API%20Reference.md)，机器可读契约见
[`openapi.yaml`](openapi.yaml)。

Gateway 同时写入 journald 控制台日志和独立的 `logs/api_gateway.log`。文件日志每小时轮转，
保留最近 7 天（168 个小时归档）；可用 `--log-path` 覆盖路径，不与后端服务日志合并存储。

> **客户端使用前必须安装 CasiaHand Root CA。** Windows、Linux、Android 安装指南和脚本见
> [`certificates/README.md`](certificates/README.md)。不得用关闭证书校验代替 CA 安装。

新 Orin 的证书注册流程见 [`certificates/README.md`](certificates/README.md)；在 Orin 上直接执行
`bash /home/wuji-brain/workspace/api_gateway/certificates/scripts/register_api_gateway.sh` 创建并安装服务器证书。注册完成并在客户端安装
CA 后，再使用仓库根目录的 `scripts/sync_and_restart_services.ps1 -ApiGatewayOnly` 同步并重启
Gateway。该流程不会自动发送任何机器人控制请求或 RecordReplay `/start`。

Windows 客户端可在仓库根目录直接执行
`api_gateway/certificates/scripts/install_casiahand_ca_windows.ps1`。脚本会先验证 `ssh orin`，
再下载并安装 CA；SSH 不通时会停止并提示先配置 SSH 别名。

重要边界：统一入口只改变外部客户端访问地址和 URL 前缀，不合并业务进程，也不会消除业务服务的内部端口。CameraPipeline 的 ZMQ、HTTP、WebSocket 端口，以及 RecordReplay、RobotControl 的独立 HTTP 端口仍然保留。Orin 本机的服务间访问和只读诊断直接使用 `localhost` 与对应内部端口，不经过 Gateway；Gateway 仅用于外部客户端访问，并在 Orin 内部通过 `127.0.0.1` 访问这些上游服务。

Orin 本机访问示例：CameraPipeline HTTP `http://127.0.0.1:6400`、CameraPipeline WebSocket
`ws://127.0.0.1:6401`、RecordReplay `http://127.0.0.1:6300`、RobotControl
`http://127.0.0.1:6500`、Calibration Service `http://127.0.0.1:6600`。外部客户端使用
`https://<orin-ip>` 和下方统一 URL 前缀；当前服务器证书支持 `192.168.100.70` 与
`192.168.1.1–192.168.1.254` 的 IP SAN（不包含 `.0` 和 `.255`）。客户端输入的 IP 必须是实际转发到 Orin Gateway
443 的入口地址。

## URL 映射

| 统一入口路径 | 后端服务 | 后端路径 | 后端端口 |
| --- | --- | --- | ---: |
| `/api/v1/camera/*` | CameraPipeline HTTP | `/api/v1/*` | 6400 |
| `/api/v1/camera-ws/*` | CameraPipeline WebSocket | `/api/v1/ws/*` | 6401 |
| `/api/v1/record-replay/*` | RecordReplay HTTP | `/*` | 6300 |
| `/api/v1/record-replay-ws` | RecordReplay 状态 WebSocket | `/api/v1/ws` | 6301 |
| `/api/v1/robot-control/*` | RobotControl HTTP | `/api/v1/*` | 6500 |
| `/api/v1/calibration/*` | Calibration Service HTTP | `/api/v1/*` | 6600 |

例如：

```text
GET  https://<orin-ip>/api/v1/camera/health
GET  https://<orin-ip>/api/v1/camera/cameras/head_camera/status
GET  https://<orin-ip>/api/v1/record-replay/health
GET  https://<orin-ip>/api/v1/record-replay/status
GET  https://<orin-ip>/api/v1/robot-control/health
GET  https://<orin-ip>/api/v1/calibration/health
GET  https://<orin-ip>/api/v1/calibration/status
SSE  https://<orin-ip>/api/v1/robot-control/status/stream?interval_s=0.2
WS   wss://<orin-ip>/api/v1/camera-ws/cameras/head_camera/color
WS   wss://<orin-ip>/api/v1/record-replay-ws
```

`GET /api/v1/gateway/health` 只表示 Gateway 自身能够响应，并返回后端端口配置；它不会探测后端服务、相机或机器人。后端服务健康检查必须访问对应的带前缀 URL。

所有 HTTP 响应带 Gateway 的 `X-Request-ID`；上游已有请求 ID 时另保留为
`X-Upstream-Request-ID`。Gateway 自身产生的路由、CORS、会话、上游连接和未知异常
统一返回 `{error_code, error_text}` JSON，错误文本包含同一请求 ID；上游服务已经返回的错误
JSON 则保持原样。Gateway 日志记录请求耗时、上游非 2xx、连接异常和完整堆栈。

## 客户端使用

统一入口地址是客户端的公共 base URL；各 typed client 通过服务前缀区分目标服务。底层 HTTP/WebSocket 协议、CameraPipeline CPWS1、RecordReplay 状态语义和 RobotControl 控制安全边界不变。

```python
from camera_pipeline.service.http_client import CameraPipelineHttpClient
from record_replay.service.client import RecordReplayClient
from robot_control.service.client import RobotControlClient
from calibration_service.service.client import CalibrationServiceClient

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
robot = RobotControlClient(
    "https://<orin-ip>",
    api_prefix="/api/v1/robot-control",
)
calibration = CalibrationServiceClient(
    "https://<orin-ip>",
    api_prefix="/api/v1/calibration",
)
```

正式客户端必须设置上述服务前缀并访问 Gateway；typed client 的空前缀仅保留给人工测试、
Orin 本地只读诊断和故障排查，不能作为 GUI 或其它正式客户端的默认配置。
RobotControl 的 SSE 状态流也沿用同一个 `/api/v1/robot-control` 前缀，Gateway 会保持
事件流连接，不会把它转换成一次性 JSON 响应。

## 启动

```bash
python -m api_gateway.service --host 0.0.0.0 --port 443 \
  --tls-cert /etc/dingtai/api-gateway/tls/api-gateway.fullchain.pem \
  --tls-key /etc/dingtai/api-gateway/tls/api-gateway.key.pem
```

Gateway 默认绑定所有 IPv4/IPv6 网卡，供外部客户端通过 Orin 的内网地址访问。应在 Orin 防火墙或网络
边界仅放行可信客户端访问 443；不要把内部 CameraPipeline、RecordReplay、RobotControl
端口暴露给外部。Gateway 不主动启动、停止或探测五个后端服务。

运行依赖 `aiohttp`。部署脚本会将它安装到 Orin 的 `wuji` 环境，不安装到系统 Python：

```bash
/home/wuji-brain/miniconda3/envs/wuji/bin/python -m pip install \
  -r /home/wuji-brain/workspace/api_gateway/requirements.txt
```
