# Camera Pipeline Architecture

## 目标

`camera_pipeline/` 的正确结构不是“本机多个测试脚本各自创建不同 client”，而是：

1. 远端部署一个统一的 `camera_pipeline_service`。
2. 远端服务内部统一持有 `PipelineContext`。
3. 本机只保留一个统一的 `camera_pipeline_client`。
4. 本机与远端所有测试都只验证这个统一服务的可用性，不再分别绕过服务直连相机或直连子模块。

这里最重要的边界是：

1. `PipelineContext` 只存在于远端服务内部。
2. 子模块只做纯计算，不构造也不持有 `PipelineContext`。
3. 本机测试脚本和远端测试脚本不是同一套文件，不能直接互相覆盖或上传替换。

---

## 分层职责

### `camera_stream/`

负责：

1. 在远端连接上游 `wuyou` 的 ZMQ 相机控制口和数据口。
2. 从 `192.168.100.60` 拉取 RGBD 流。
3. 维护最近帧缓存。

不负责：

1. 对外暴露给本机的统一业务 API。
2. 托盘、开口、球位姿算法。
3. 本机测试逻辑。

### `tray_detection/`、`opening_detection/`、`ball_pose_detection/`

负责：

1. 接收已经准备好的协议入参。
2. 只执行本模块算法。
3. 返回协议化结果。

不负责：

1. 相机连接。
2. `PipelineContext` 生命周期。
3. 本机或远端测试脚本中的连接策略。
4. 服务监听、服务编排和跨模块数据流动。

### `pipeline_context.py`

负责：

1. 在远端统一管理相机流运行时。
2. 在远端按 `frame_id` 选择相机帧。
3. 在远端把帧数据组织成子模块可消费的协议入参。
4. 在远端统一编排 tray、opening、ball 三类计算调用。

不负责：

1. 本机网络接入。
2. 本机调试脚本中的服务地址管理。
3. 算法实现本身。

### `camera_pipeline_service`

负责：

1. 作为远端唯一对外服务入口。
2. 内部持有 `PipelineContext`。
3. 对外暴露统一 API，例如：
   - 获取相机首帧摘要
   - 请求 tray detection
   - 请求 opening detection
   - 请求 ball pose detection
4. 屏蔽远端内部相机流、子模块和上下文细节。

### `camera_pipeline_client`

负责：

1. 作为本机唯一访问远端 `camera_pipeline_service` 的客户端封装。
2. 统一本机调用方式、超时策略和协议编解码。
3. 为本机测试脚本提供稳定 API。

不负责：

1. 本机自己直接构造 `PipelineContext`。
2. 本机自己连接 `192.168.100.60` 相机流。
3. 本机分别维护多个互不一致的 RPC client。

---

## 正确数据流

### 远端 tray detection

```text
camera_pipeline_service
  -> PipelineContext.get_frame(...)
  -> tray_detection.compute(frame, request)
  -> response
```

### 远端 opening detection

```text
camera_pipeline_service
  -> PipelineContext.get_frame(...)
  -> tray_detection.compute(frame, request)
  -> opening_detection.compute(frame, tray_mask, request)
  -> response
```

### 远端 ball pose detection

```text
camera_pipeline_service
  -> PipelineContext.get_frame(...)
  -> ball_pose_detection.compute(frame, request)
  -> response
```

### 本机调用

```text
local test script
  -> camera_pipeline_client
  -> tcp://<orin-host>:<service-port>
  -> camera_pipeline_service
```

本机不应该出现下面这种错误路径：

```text
local test script
  -> local PipelineContext
  -> direct camera stream access
```

---

## 测试边界

### 远端测试脚本

远端 `test/` 下脚本负责：

1. 在远端本机验证 `camera_pipeline_service` 是否能正确调用 `PipelineContext`。
2. 验证远端到 `192.168.100.60` 的相机流转发是否正常。
3. 验证远端算法服务调用链是否完整。

远端测试脚本可以直接连：

1. `127.0.0.1:<service-port>`

因为它们运行在 Orin 本机。

### 本机测试脚本

本机 `test/wuji/` 下脚本负责：

1. 验证本机到 Orin 上统一 `camera_pipeline_service` 的连通性。
2. 验证统一 `camera_pipeline_client` 的协议和结果是否正确。
3. 不直接验证本机到 `192.168.100.60` 的相机链路。

本机测试脚本应该连接：

1. `tcp://<orin-host>:<service-port>`

例如 `tcp://192.168.1.118:<service-port>`。

### 重要约束

1. 本机测试脚本和远端测试脚本不一致。
2. 远端脚本不能由本机脚本直接覆盖。
3. 本机脚本只验证“本机 -> Orin 服务”。
4. 远端脚本才验证“Orin 服务 -> PipelineContext -> 192.168.100.60 相机流”。

---

## 当前整改目标

后续代码整改必须满足：

1. 新增统一 `camera_pipeline_service`。
2. 新增统一 `camera_pipeline_client`。
3. 本机 `orin_camera.py` 不再直接构造 `PipelineContext`。
4. 本机 tray、opening、ball 测试不再各自直接持有分散 RPC client。
5. 远端 `PipelineContext` 只在服务内部使用。
6. 远端测试脚本与本机测试脚本分别维护，各自验证各自边界。

在这些目标完成前，当前实现只能视为中间态，不是最终架构。
