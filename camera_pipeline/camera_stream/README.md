# Camera Stream

## 单一职责

`camera_stream` 只负责 ZMQ 模式下连接上游相机控制口和数据口、解码相机数据，并
分别维护彩色帧与 RGBD 帧的最新值和有限缓存。它不负责本机 USB 相机、算法、
统一 RPC、帧稳定性或业务编排。本机 USB 采集独立位于 `camera_local`。

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
- `distortion`：彩色相机畸变参数，按 OpenCV 8 参数顺序
  `(k1, k2, p1, p2, k3, k4, k5, k6)` 排列。

上游 wuyou ZMQ 控制响应使用 `data["dist"]`，并按 Orbbec SDK 字段顺序
`(k1, k2, k3, k4, k5, k6, p1, p2)` 传输。`CameraStreamRuntime` 在控制协议
边界严格校验八个数值并转换为上述 OpenCV 顺序；算法模块不接触上游字段名或
SDK 顺序。当前暂不消费上游未传输的畸变模型字段。

## 缓存语义

运行时同时维护最新帧和按 `frame_id` 查询的有限缓存。`PipelineContext` 当前默认配置 64 帧，以覆盖 1 秒稳定窗口的中点帧。缓存是运行时资源，不应被算法模块持有。

## 生命周期

```text
construct -> start -> background capture -> stop
```

`start()` 立即建立本地 SUB 连接并启动收流线程，不等待上游控制服务；
`stop()` 停止线程、关闭 socket 并终止私有 ZMQ context。每个已连接安装位由
`PipelineContext` 持有一个独立运行时，因此头部、胸腔和左臂的内参缓存不会混用。
调用方必须显式关闭上下文，由上下文依次释放各路运行时。

## 自愈与超时

后台线程负责收敛上游状态：开启深度和读取内参失败时按 `2 s` 起步、最大
`30 s` 的指数退避持续重试。控制响应成功后仍以真实 RGBD 帧作为深度恢复依据；
连续 30 帧只有彩色载荷时重新进入深度开启流程。连续收流超时达到配置次数后，
运行时重建 SUB socket、清空旧内参并重新执行控制恢复。上游控制服务和数据服务
可以早于或晚于 CameraPipeline 启动，不需要人工重启 CameraPipeline。

上游偶发只发送彩色帧、并在帧头中明确设置 `depth_format=0` 时，运行时仍解码并
缓存 `CameraColorFramePacket`，但不更新 RGBD 缓存，也不将其误判为 LZ4 损坏或
触发流重连。真正的协议版本错误、消息总长度不一致、深度尺寸不一致或压缩载荷
损坏仍按解码失败处理并自愈。

## 成功与失败语义

- 成功启动：后台接收并缓存 `CameraFramePacket`。
- 首帧未到达：`wait_until_ready()` 返回 `False`。
- 帧号不存在：`get_frame_by_id()` 返回 `None`，表示缓存淘汰或尚未接收。
- 控制超时、协议头错误或图像解码失败：运行时记录错误并尝试自愈；上层就绪检查失败时报告服务错误。
- 上游瞬时缺少深度载荷：更新彩色帧缓存，不更新 RGBD 缓存、不重建 socket。
- 上游持续缺少深度载荷：彩色缓存继续更新，并按退避计划重新请求开启深度。
- 上游控制服务启动较晚：服务进程保持运行，日志记录重试次数和下次间隔，成功后
  记录控制恢复及实际深度流确认。
- `get_intrinsics` 缺少 `dist`、系数不是八个或包含非数值：判定上游协议错误，
  不构造带有错误畸变顺序的帧。

## 局限性

- 当前解码格式与上游 wuyou 二进制协议耦合。
- 缓存按帧数而非时间长度配置，高帧率下需要重新核算容量。
- 内参在启动及收流恢复阶段由控制请求刷新；上游无断流地改变内参时不会主动轮询。
- 未连接真实设备时只能验证类型、解码辅助逻辑和生命周期代码，不能证明实际流兼容。
