---
name: robot-control-service-maintainer
description: 安全维护 Dingtai 根目录 robot_control 统一机器人控制服务。用于修改 qmlinker、AR5 xCoreSDK、HTTP API、设备网关、服务部署或只读状态验证；强制禁止 Codex、CI、hook 和无人值守流程执行任何控制 POST、运动测试或真实硬件控制。
---

# Robot Control Service Maintainer

## 安全红线

1. `robot_control` 的 qmlinker、AR5、头部、升降、夹爪、右手和 AGV 接口都视为真实硬件接口。
2. 禁止 Codex、CI、hook、静态检查脚本或无人值守任务：
   - 发送任意 `POST /api/v1/qmlinker/...` 或 `POST /api/v1/ar5/...`；
   - 调用 `RobotControlGateway`、`RobotControlApplication` 或第三方客户端的控制方法做测试；
   - 启动服务后自动探测并控制真实设备；
   - 运行运动冒烟、Jog、Move、上电、使能、导航、夹爪、升降或拖动测试。
3. 只读验证仅允许：
   - 静态检查、UTF-8 扫描、`py_compile`、`compileall`；
   - 不创建硬件客户端的协议对象离线构造；
   - 在明确的人工授权环境下只调用 `GET /api/v1/health`、`GET /api/v1/status` 或
     `GET /api/v1/devices`。Codex 默认不连接现场服务。
4. 真实控制测试只能由用户现场手动发起。最终报告必须明确“未执行控制测试，硬件行为未经本回合验证”。

## 代码边界

1. `robot_control/` 是根目录独立服务包，负责统一 HTTP 边界、设备生命周期、状态协议和控制路由。
2. `robot_control/gateway.py` 负责 qmlinker/xCoreSDK 适配，不负责 HTTP 路由；`robot_control/service/` 负责 API、启动和客户端。
3. 复用现有 `src/wuji` 客户端和 `sdk/xcoresdk`，不在 HTTP 层复制 SDK 协议细节。
   远端部署使用 Orin 已有 Linux xCoreSDK 二进制，不得同步本机 Windows SDK 二进制。
4. qmlinker 与 AR5 的设备对象必须延迟创建；导入模块、构造配置和健康检查不得连接或控制硬件。
5. 只读 GET 与控制 POST 必须保持清晰边界。AR5 控制客户端默认使用
   `initialize_toolset=False`，不得在 GET 或普通 RobotControl 控制请求中写入 tool/wobj；
   tool/wobj 固定配置属于 RecordReplay 的显式回放准备流程。
6. 不暴露原始 SDK 对象、channel、动态 `invoke(method, args)` 或 `getattr` 分发；HTTP body 使用显式字段和单位。
7. 控制方法必须串行访问设备，必须区分“请求已接受”和“动作已经完成”；`stop` 不等于硬件急停。
8. 服务默认绑定 `127.0.0.1`。如需现场内网访问，必须由用户明确确认网络、认证和安全边界；SSH 只能作为外层访问通道。
9. Orin 本机的服务间访问和只读诊断必须直接使用对应服务的 `localhost`/`127.0.0.1` 与内部端口，
   不经过 API Gateway。API Gateway 只用于外部客户端统一访问；不得把 Gateway 的 HTTPS 443
   或统一 URL 前缀作为 Orin 本机访问 RobotControl 的默认路径。
10. `robot_control/README.md`、`robot_control/API Reference.md`、`robot_control/openapi.yaml` 和
   `robot_control/CHANGELOG.md` 是强制维护的说明性文档；任何新增或修改公开 API、状态协议、
   配置、部署方式、端口、单位或安全边界，都必须在同一批改动中同步更新这四个文件。
   不能只改代码后再补文档，也不能以“实现未变”跳过文档核对。
11. RobotControl 版本号变化后，必须先执行
   `.agents/skills/robot-control-service-maintainer/scripts/check_robot_control_contract.ps1`。
   该检查会核对源码、CHANGELOG、API Reference、OpenAPI 的版本一致性，并在版本变化时确认
   README、API Reference、OpenAPI、CHANGELOG 四份强制文档都已修改；检查失败时禁止部署。
12. 版本契约检查通过后，必须执行
   `scripts/sync_and_restart_services.ps1 -RobotControlOnly` 同步并重启远端 RobotControl；
   必须记录同步文件、远端备份、服务状态、`GET /api/v1/health` 和远端实际版本结果。
   该流程只检查
   `GET /api/v1/health`，不得用 `/status` 做启动就绪检查，也不得发送控制 POST。
13. 版本变化仅以本机源码为依据时不得报告远端已更新；Orin 不可达、同步失败、SHA-256
   或远端实际版本不一致时，必须明确报告 RobotControl 未完成远端同步。

## 修改流程

1. 先阅读 `robot_control/README.md`、`API Reference.md`、`CHANGELOG.md`、`openapi.yaml`、服务入口和对应的现有 `src/wuji` 客户端。
2. 修改已有文件前，在项目根 `.archive/` 中按相对路径生成 UTF-8 快照；删除、移动和重命名也必须先快照。
3. 先设计 dataclass、设备边界和错误语义，再修改 gateway、application、server 或 client。
4. 仅为控制能力补充实现，不为通过检查改变默认单位、时序、重试、模式或设备安全门。
5. 任何协议或行为变化都要升级 `robot_control/CHANGELOG.md` 版本号；同一批改动只升级一次，
   并在同一批改动中复核 README、API Reference 和 OpenAPI 的路径、字段、单位、示例与版本号。
6. 版本号发生变化时，完成代码和文档修改后先执行
   `check_robot_control_contract.ps1`；只有检查返回成功，才能执行
   `pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RobotControlOnly`。
   不得颠倒顺序，也不得以静态检查通过替代契约检查或远端同步。
7. 不新增自动硬件测试。测试代码如确有必要，只能覆盖协议解析、状态序列化和显式 GET，且不得创建真实 SDK 对象。

## 验证流程

1. 使用 Dingtai 静态检查 skill 的脚本，先 ruff 后 pyright；固定 DingTai 环境。
2. 对新增 Python 文件执行 `py_compile` 或 `compileall`，不执行 `python -m robot_control.service` 启动冒烟。
3. 每次 RobotControl 版本变化必须执行
   `.agents/skills/robot-control-service-maintainer/scripts/check_robot_control_contract.ps1`，
   通过后才允许运行 `scripts/sync_and_restart_services.ps1 -RobotControlOnly`。
4. 必须检查 `API Reference.md`、OpenAPI YAML、JSON 协议对象和路由字符串的一致性；不得通过
   POST 路由验证控制分发。API Reference 不能只写链接，必须能让客户端按文档构造请求和解析响应。
5. 如用户明确授权现场只读检查，只允许 GET，且逐项记录服务地址、接口、返回状态和未验证的硬件边界。
6. 最终报告必须分别列出：静态验证、契约检查、同步/部署结果、只读验证、未执行的控制测试、未验证的硬件行为和部署/哈希状态。

## 单位与控制边界

- qmlinker arm 关节目标：`joint_deg`，单位 deg。
- AR5 关节目标：`joint_deg`，单位 deg；服务内部由 xCoreSDK 使用 rad。
- AR5 笛卡尔平移：`xyz_mm`，单位 mm；姿态 `rpy_deg`，单位 deg，采用 SciPy 小写外禀 xyz。
- AR5 速度：`speed_mm_s`，单位 mm/s；zone：`zone_mm`，单位 mm。
- 右手位置：归一化 `0.0` 到 `1.0`。
- `stop` 只表示软件停止请求，不替代硬件急停。
