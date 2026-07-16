# CameraPipeline TODO

## 远端确认 Orbbec 畸变参数协议

### 当前状态

本机当前未连接实际部署 ZMQ 相机服务的远端，无法读取真实
`get_intrinsics` 响应，也无法核对上游服务如何把 Orbbec SDK 畸变对象序列化。
在连接远端并取得真实证据前，不继续猜测字段名、系数数量或排列顺序。

当前 `camera_pipeline/camera_stream/runtime.py` 按 `data["distortion"]` 读取畸变参数。
仓库内既有 `src/wuji/zmq_camera_client.py` 则按 `data["dist"]` 读取。两处协议约定
不一致，需要在远端验证后统一。

### 已确认事实

本地 `pyorbbecsdk.pyi` 明确提供完整相机标定参数：

- `Pipeline.get_camera_param() -> OBCameraParam`
- `OBCameraParam.depth_intrinsic`
- `OBCameraParam.depth_distortion`
- `OBCameraParam.rgb_intrinsic`
- `OBCameraParam.rgb_distortion`
- `OBCameraParam.transform`

`OBCameraDistortion` 包含以下字段：

```text
k1, k2, k3, k4, k5, k6, p1, p2
```

`VideoStreamProfile` 也提供：

```text
get_intrinsic() -> OBCameraIntrinsic
get_distortion() -> OBCameraDistortion
```

因此 SDK 本身没有缺失 distortion。待确认的问题位于上游 ZMQ 序列化协议及
`camera_pipeline` 的协议消费边界。

### 远端连接后需要检查

1. 直接调用每个已连接相机的 `get_intrinsics`，保存或打印完整响应：

   ```text
   camera_name
   camera_id
   response
   response["data"].keys()
   response["data"]["fx/fy/cx/cy"]
   response["data"] 中实际的畸变字段
   ```

2. 运行：

   ```powershell
   python experiments/zmq_camera_link_check.py
   ```

   记录每个相机输出的 `dist` 内容、长度和数值。

3. 在远端 ZMQ 服务实现中定位 `get_intrinsics` 的生产代码，确认：

   - 畸变字段实际命名是 `dist`、`distortion` 还是其它名称。
   - 使用的是 `rgb_distortion` 还是 `depth_distortion`。
   - 当前传输图像是原始彩色图还是已经去畸变的彩色图。
   - 畸变参数长度是 4、5、8、12 还是 14。
   - 畸变模型是否为 OpenCV pinhole/rational model。
   - 参数排列顺序是否已经转换为 OpenCV `distCoeffs` 顺序。

4. 对同一相机同时读取 SDK 原生参数并与 ZMQ 响应逐项比较：

   ```text
   rgb_intrinsic.fx/fy/cx/cy
   rgb_distortion.k1/k2/k3/k4/k5/k6/p1/p2
   ZMQ fx/fy/cx/cy
   ZMQ distortion sequence
   ```

5. 使用真实稳定帧执行一次 ChArUco 检测，比较：

   - 使用 ZMQ 畸变参数的 `error_px`
   - 使用零畸变参数的 `error_px`
   - `t_cam_board_mm`
   - `marker_num`
   - `charuco_num`
   - debug overlay 中投影坐标轴是否与板面一致

### OpenCV 参数顺序风险

SDK 对象字段的声明顺序不是可直接传入 OpenCV 的数组顺序。

OpenCV 常用 8 参数顺序为：

```text
k1, k2, p1, p2, k3, k4, k5, k6
```

SDK 对象可访问字段为：

```text
k1, k2, k3, k4, k5, k6, p1, p2
```

不能按 SDK 字段排列直接构造 `cv2.solvePnP` 使用的 `distCoeffs`。必须以远端生产
代码或 SDK 原始值对比为证据，确认 ZMQ 是否已经完成顺序转换。

若远端确认传输完整 8 参数，推荐协议明确使用：

```python
dist = (
    rgb_distortion.k1,
    rgb_distortion.k2,
    rgb_distortion.p1,
    rgb_distortion.p2,
    rgb_distortion.k3,
    rgb_distortion.k4,
    rgb_distortion.k5,
    rgb_distortion.k6,
)
```

### 确认后的处理边界

远端证据确认后再实施以下修改：

1. 在 ZMQ 适配边界统一唯一字段名，不同时长期兼容 `dist` 和 `distortion`。
2. `CameraStreamRuntime` 负责把外部响应校验并收窄为明确的
   `tuple[float, ...]`。
3. `CameraFramePacket` 和 `RgbdFrameProtocol` 继续承载已经规范化为 OpenCV 顺序的
   彩色相机 distortion。
4. `charuco_detection` 只消费协议字段，不读取 ZMQ 响应、不访问 SDK、不增加兼容
   分支。
5. 更新 `camera_pipeline/API Reference.md` 和相关 README，写明字段名、参数数量、
   顺序、畸变模型以及图像是否已去畸变。
6. 使用真实相机完成 ChArUco 位姿和 overlay 验证后，才能标记该 TODO 完成。

### 完成条件

- 已取得真实 `get_intrinsics` 响应。
- 已定位远端 ZMQ 生产代码。
- 已确认彩色图对应的 distortion 来源、模型、长度和顺序。
- `camera_pipeline` 与 `src/wuji` 使用同一份明确协议。
- ChArUco 使用真实相机验证通过，重投影误差和 pose overlay 合理。
- 文档与最终实现一致。
