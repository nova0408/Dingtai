# API Gateway 版本日志

当前版本：`0.3.0`

## 0.3.0 - 2026-08-07

- 新增 RecordReplay 状态 WebSocket 转发：`/api/v1/record-replay-ws` -> Orin 内部 6301。

## 0.2.2 - 2026-08-04

- Gateway 默认同时监听 `0.0.0.0:443` 与 `[::]:443`，支持 Flutter Web 通过 IPv6 访问。
- 保持 `wujibrain-desktop` hostname 证书、Flutter Web 调试 CORS 来源和后端 `127.0.0.1` 访问边界不变。

## 0.2.1 - 2026-08-03

- 修复 Orin OpenSSL 1.1.1f 使用系统默认配置时重复写入 CA 扩展导致证书链验证失败的问题。
- 增加无参数 `register_api_gateway.sh`，用于新 Orin 自动创建 CasiaHand CA、签发并安装 Gateway 证书。
- 安装脚本增加中文完成提示、安装后证书链/私钥/权限校验和可用时的只读 HTTPS 健康检查。
- 增加 Windows 一键通过 `ssh orin` 下载并安装 CA 的脚本，并在 SSH 不通时明确停止。
- 注册、服务器安装和客户端安装入口改为固定路径、自动读取 hostname 的无参数流程。
- 明确 CA 仅作为签发机构；服务器证书只签发当前 hostname 的 DNS SAN，签发和安装校验只验证
  hostname，不验证 IP 地址。

## 0.2.0 - 2026-08-03

- Gateway 由 aiohttp 直接终止 TLS，并将正式客户端入口改为 HTTPS 443。
- 增加 CasiaHand 自签名 CA、服务器证书生成与 Windows/Linux/Android 客户端安装方案。
- systemd 仅授予非 root 服务绑定低端口所需的 `CAP_NET_BIND_SERVICE`。
- 部署健康检查使用 CasiaHand CA 验证证书，不允许跳过 TLS 校验。

## 0.1.1 - 2026-08-03

- 修复 CameraPipeline RGBD WebSocket 帧超过 aiohttp 默认 4 MiB 限制后被 Gateway 关闭的问题。
- Gateway 客户端与上游 WebSocket 消息上限统一为 16 MiB。

## 0.1.0 - 2026-07-31

- 新增统一客户端 HTTP 入口，转发 CameraPipeline、RecordReplay 和 RobotControl。
- 新增 CameraPipeline WebSocket 二进制流转发。
- 明确统一入口不替代业务服务端口，也不改变后端 API 协议和硬件安全边界。
