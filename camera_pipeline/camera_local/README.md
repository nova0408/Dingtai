# Camera Local

## 单一职责

`camera_local` 只负责通过 `pyorbbecsdk` 连接本机 USB Orbbec 相机、按 SN 选择设备、
按配置启用 RGBD profile、读取当前帧与彩色相机标定参数，并在断线后自动重连。
它不负责 ZMQ 上游协议、算法、统一 RPC 或帧发布。

`config.py` 不导入 `pyorbbecsdk`；只有 `PipelineContext` 实际选择 `usb` 模式时才延迟
导入 `runtime.py`，因此默认 ZMQ 模式不会加载本机相机 SDK。

## 输入与输出

- 输入：逻辑相机名、LEFT/RIGHT/HEAD/CHEST 安装位、SN、彩色和深度 profile、超时与重试间隔。
- 输出：顶层 `CameraFramePacket`，包含 BGR `uint8 (H, W, 3)`、毫米深度
  `uint16 (H, W)`、彩色相机内参与畸变系数。
- SDK 配置启用软件 D2C 对齐，因此输出深度图使用彩色流坐标系与分辨率。

## 当前帧语义

本机模式只保存最新帧，不建立历史帧缓存。`get_frame_by_id()` 仅在请求 ID 仍对应
当前帧时返回；稳定帧编排在本机模式下返回形成稳定证据时的当前帧，不回取时间窗中点。

## 生命周期与重连

每台启用相机持有一个守护采集线程：

```text
枚举 SN -> 创建 pipeline -> 启用配置 profile -> 持续取帧
   ^                                                |
   +---- 释放 pipeline <- 超时、断线或 SDK 异常 <---+
```

连接失败、SN 未找到、取帧超时或 SDK 异常都会写入 Loguru 日志。首次失败等待
`reconnect_initial_interval_s`，连续失败按 2 倍指数退避，最多等待
`reconnect_max_interval_s`；成功收到帧后发生断线时从初始间隔重新开始。SN 为空时配置
在当前进程内不可能自行变化，因此线程只记录一次警告并休眠到服务停止，不执行无效枚举。
停止服务时由 `PipelineContext.close()` 请求线程退出。

## 失败语义

- SN 为空：保持离线、记录警告并持续重试。
- 找不到设备或 profile 不匹配：记录完整异常，释放本轮资源后重试。
- 尚无首帧：`get_latest_frame()` 返回 `None`，`wait_until_ready()` 超时返回 `False`。
- 断线：立即清除旧帧，避免上层继续把断线前画面当作在线数据。

未连接真实相机时，静态检查不能证明设备枚举、profile 支持或 D2C 行为正确。
