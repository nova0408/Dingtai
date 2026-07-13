# Camera Stream

## 单一职责

`camera_stream` 只负责连接上游相机控制口和数据口、解码 RGBD 数据、维护最新帧与按帧号索引的有限缓存。它不负责算法、统一 RPC、帧稳定性或业务编排。

## 模块结构

- `runtime.py`
  - `CameraStreamRuntimeConfig`：上游地址、超时、相机标识和缓存大小。
  - `CameraStreamRuntime`：控制相机、后台收帧、解码和缓存。
- `__init__.py`：只导出相机流运行时及其配置。

帧 Protocol 和传输 packet 统一定义在顶层 `camera_pipeline/protocol.py`。`camera_stream` 只生产具体 packet，算法模块只依赖 `RgbdFrameProtocol`。

## 输入与输出

运行时从上游接收压缩二进制帧，输出 `CameraFramePacket`：

- `frame_id`：上游 sequence。
- `timestamp_ms`：相机时间戳，单位 ms。
- `color_bgr`：`uint8 (H, W, 3)`。
- `depth_mm`：`uint16 (H, W)`，单位 mm，`0` 为无效深度。
- `fx/fy/cx/cy`：针孔相机内参，单位 pixel。

## 缓存语义

运行时同时维护最新帧和按 `frame_id` 查询的有限缓存。`PipelineContext` 当前默认配置 64 帧，以覆盖 1 秒稳定窗口的中点帧。缓存是运行时资源，不应被算法模块持有。

## 生命周期

```text
construct -> start -> background capture -> stop
```

`start()` 打开深度、读取内参并启动收流线程；`stop()` 停止线程、关闭 socket 并终止私有 ZMQ context。调用方必须显式关闭。

## 自愈与超时

连续收流超时达到配置次数后，运行时重建流 socket。控制命令和收流超时分别配置，不应由算法请求修改。

## 成功与失败语义

- 成功启动：后台接收并缓存 `CameraFramePacket`。
- 首帧未到达：`wait_until_ready()` 返回 `False`。
- 帧号不存在：`get_frame_by_id()` 返回 `None`，表示缓存淘汰或尚未接收。
- 控制超时、协议头错误或图像解码失败：运行时记录错误并尝试自愈；上层就绪检查失败时报告服务错误。

## 局限性

- 当前解码格式与上游 wuyou 二进制协议耦合。
- 缓存按帧数而非时间长度配置，高帧率下需要重新核算容量。
- 内参来自启动时控制请求；上游运行中改变内参时不会自动同步。
- 未连接真实设备时只能验证类型、解码辅助逻辑和生命周期代码，不能证明实际流兼容。
