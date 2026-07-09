# Ball Pose Detection

`camera_pipeline.ball_pose_detection` 只负责 RGBD 小球检测，不负责坐标系生成、位姿求解或结果回传拼装。

## 算法逻辑

输入是一帧 RGBD 图像和一组球先验。

服务端只做下面这些事情：

1. 根据球颜色在 RGB 图中分割候选区域。
2. 对每个候选区域提取轮廓，计算圆心像素位置和像素半径。
3. 结合深度图把圆心反投影到相机坐标系，得到球圆心三维坐标。
4. 根据相机内参与深度估计一个物理半径，作为检测半径输出。
5. 按先验颜色顺序返回三个球的检测结果。

服务端不构建黄球原点坐标系，不构建红球 x 轴，不判断紫球是否落在 xoy 平面，也不输出任何刚体变换矩阵。

## 输入数据

### `BallPoseDetectionRequest`

- `request_id`: 请求编号
- `camera_name`: 相机名称
- `frame_id`: 要处理的 frame 编号
- `enable_debug`: 是否返回 debug 数据
- `priors`: 球先验列表，主要提供颜色和半径先验

### `BallPosePriorInfo`

- `color_hex`: 球颜色，使用 HEX 码表示
- `radius_mm`: 球半径，单位 mm
- `model_center_mm`: 预留字段，检测端不参与坐标系构建

## 输出数据

### `BallPoseDetectionResponse`

- `matched_count`: 成功检测到并估计出圆心的球数量
- `detections`: 每个球的检测结果摘要，包含：
  - `color_hex`
  - `detected`
  - `center_px`
  - `center_mm`
  - `radius_mm`
  - `radius_px`
  - `center_norm`
  - `radius_norm`
  - `point_count`
  - `status`
- `debug`: 调试数据
- `error`: 错误信息

## Debug 数据

`BallPoseDetectionDebugArtifacts` 中包含：

- `color_bgr`: 原始彩色图
- `depth_mm`: 原始深度图
- `camera_intrinsics`: 相机内参 `(fx, fy, cx, cy)`
- `overlay_bgr`: 仅叠加检测结果的图
- `detection_overlay_bgr`: 仅叠加检测结果的图
- `detections`: 每个球的 debug 字典

当 `enable_debug=False` 时，不再生成这些大对象。

## 坐标系说明

坐标系构建由 `test/wuji/ball_pose_detection.py` 在本地完成：

- 黄球圆心作为原点
- 红球圆心定义 `x` 轴
- 紫球圆心约束在 `xoy` 平面

服务端只返回三个球的圆心坐标，后续位姿、相对变换、对比和可视化都由本地脚本处理。

## 备注

该模块不负责相机采集、不负责 RPC 端点配置，也不负责先验采集脚本。
