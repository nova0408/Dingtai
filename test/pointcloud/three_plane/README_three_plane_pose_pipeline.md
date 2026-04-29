# Orbbec 三平面位姿优化管线说明

## 目标

本文档记录 `test/pointcloud/test_orbbec_three_plane_pose_optimized_pipeline.py` 对应的完整方案。该方案用于 Orbbec Gemini 305 实时点云中检测三个结构平面，并以三平面交点为原点计算测试坐标系，同时排除料盘区域对平面拟合的干扰。

当前实现把可复用能力沉淀到 `src`：

- `src/rgbd_camera`：Orbbec 会话、点云裁切、相机内参/外参数据结构。
- `src/pointcloud/tray_detection.py`：料盘 zero-shot 检测与点云排除。
- `src/pointcloud/tray_projection.py`：点云到图像的投影。
- `src/pointcloud/three_plane_pose.py`：三平面分割、平面排序、坐标系计算。
- `src/pointcloud/three_plane_types.py`：三平面配置、结果结构和位姿稳定器。
- `test/pointcloud/test_orbbec_three_plane_pose_optimized_pipeline.py`：实时采集、线程调度、Open3D/CV2 预览和日志。

## 运行方式

直接运行：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe test\pointcloud\test_orbbec_three_plane_pose_optimized_pipeline.py
```

常用参数：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe test\pointcloud\test_orbbec_three_plane_pose_optimized_pipeline.py `
  --tray-exclusion `
  --tray-use-sam `
  --pose-smooth-frames 5 `
  --compute-min-interval-s 0.10
```

参数含义：

- `--tray-exclusion / --no-tray-exclusion`：是否启用料盘识别排除。
- `--tray-use-sam / --no-tray-use-sam`：是否使用 SAM 生成精细料盘 mask。关闭时退化为检测框区域。
- `--pose-smooth-frames`：位姿稳定窗口，只使用最近实际完成计算的帧，最大 15。
- `--compute-min-interval-s`：提交计算任务的最小间隔，单位秒。

## 数据流

主流程如下：

1. `Gemini305` 启动 Orbbec 会话。
2. `session.get_projection_intrinsics()` 读取当前投影内参。
3. `session.calculate_points_from_frames()` 从 SDK 帧生成点云。
4. `session.filter_points_for_sensor()` 使用 Gemini 305 默认视锥裁切点云。
5. 预览线程持续更新原始点云。
6. 若计算线程空闲，将当前帧放入单元素队列。
7. 计算线程执行料盘排除、三平面检测和坐标系估计。
8. 预览线程消费最新结果，更新 2D overlay、3D 平面点云、坐标系和日志。

计算队列是单元素模式：如果上一帧计算尚未完成，当前帧直接丢弃，不进入计算队列。这样可以避免积压旧帧，保持预览和计算结果尽量接近实时状态。

## 相机内参和外参

内参不再以 `fx/fy/cx/cy` 散落参数传递，而是封装为：

- `CameraIntrinsics`
- `CameraExtrinsics`

会话提供以下方法：

- `get_depth_intrinsics()`
- `get_color_intrinsics()`
- `get_projection_intrinsics()`
- `get_depth_to_color_extrinsics()`

`project_points_to_image(xyz, intrinsics)` 只接收点云和 `CameraIntrinsics`。投影函数使用 `intrinsics.width/height/fx/fy/cx/cy` 完成针孔投影，不做畸变校正。

## 料盘排除

料盘排除由 `TrayPointExcluder` 负责：

1. GroundingDINO 根据 prompt 检测候选料盘。
2. 严格关键词过滤保留目标标签。
3. 默认启用 SAM，生成实际料盘 mask。
4. 对 mask 做闭运算和最小面积过滤。
5. 使用 `project_points_to_image()` 得到每个点云点的像素坐标。
6. 通过 `collect_indices_in_mask()` 将 2D mask 映射到点云 `(N,) bool` 排除掩码。
7. `estimate_three_plane_pose()` 对排除点标记为 `-2`，不参与 RANSAC 和 PCA。

2D 预览中料盘区域按 `TrayDetection.mask` 实际区域半透明填充；只有关闭 SAM 时，该 mask 才会退化为检测框区域。

## 三平面位姿算法

三平面算法入口为：

```python
estimate_three_plane_pose(xyz, excluded_mask=excluded, config=pose_cfg)
```

输入：

- `xyz`：裁切后的点云，形状为 `(N, 3)` 或 `(N, C)`，单位 mm。
- `excluded_mask`：料盘排除掩码，形状为 `(N,)`，`True` 表示不参与平面拟合。
- `PlanePoseConfig`：RANSAC 阈值、PCA 精修阈值、底面参考轴、坐标系 X/Z 参考方向。

处理步骤：

1. 从非排除点中使用 Open3D RANSAC 分割最多三个平面。
2. 把全量点分配给最近平面，排除点写为 `-2`。
3. 可选使用 PCA 对每个平面法线二次精修。
4. 根据 `bottom_axis` 识别底面。
5. 剩余两个面按点云 X 均值排序为 `left_side` 和 `right_side`。
6. 求三个平面交点作为测试坐标系原点。
7. 底面法线按 `frame_z_hint` 定向为 Z 轴。
8. X 轴优先使用投影到底面切平面的 `frame_x_hint`。
9. 使用 `Axis(origin, x_axis, z_axis)` 构造右手坐标系。

`CoordinateFramePose` 只保存 `Axis`、`rpy_deg` 和 `residual`。`Transform`、`origin_mm` 和 `rotation` 都是从 `Axis` 派生的属性，其中旋转矩阵通过 `axis.to_transform().as_SE3()[:3, :3]` 提取。

## 位姿稳定

`PoseWindowStabilizer` 只平滑实际完成计算的位姿帧，不缓存相机原始帧。默认窗口为 5，最大 15。

稳定策略：

- 以最新完成帧为参考，对历史 X/Z 轴做符号对齐。
- 对原点、X 轴、Z 轴分别求均值。
- 使用 `Axis` 重新生成右手坐标系。
- `residual` 保留最新完成帧的残差。

该策略用于降低法线符号翻转和局部抖动，同时避免缓存过多帧导致跟随性明显下降。

## 实时线程模型

脚本分为预览线程和计算线程：

- 预览线程：相机取帧、点云裁切、Open3D 原始点云刷新、CV2 2D 图显示、计算任务提交。
- 计算线程：料盘检测、点云排除、三平面估计、2D overlay 生成。

队列策略：

- `job_queue` 最大长度为 1。
- `result_queue` 最大长度为 2。
- 计算线程忙时，预览线程丢弃当前帧。
- 结果队列满时，丢弃旧结果，只保留最新结果。

该设计优先保证实时性，不追求每帧都计算。

## 预览内容

2D 窗口：

- 实际料盘 mask 半透明红色覆盖。
- 料盘轮廓和置信度。
- 平面分配点数。
- 当前 XYZ/RPY。

3D 窗口：

- 原始裁切点云。
- 三平面高亮点云。
- 当前测试坐标系。
- 当前原点黄色标记。

预览颜色属于测试脚本职责。`PlanePatch` 不保存颜色，算法层只输出几何结果和语义标签。

## 默认参数

关键默认值在测试脚本顶部直接定义：

- `DEFAULT_TIMEOUT_MS = 120`
- `DEFAULT_CAPTURE_FPS = 30`
- `DEFAULT_MAX_DEPTH_MM = 5000.0`
- `DEFAULT_MAX_PREVIEW_POINTS = 100_000`
- `DEFAULT_COMPUTE_MIN_INTERVAL_S = 0.10`
- `DEFAULT_ENABLE_TRAY_EXCLUSION = True`
- `DEFAULT_TRAY_USE_SAM = True`
- `DEFAULT_POSE_SMOOTH_FRAMES = 5`

`TrayDetectionConfig` 的模型和阈值默认值定义在 `src/pointcloud/tray_detection_types.py` 字段上，不再通过同文件 `DEFAULT_*` 二次包装。

## 结果解释

日志中的当前坐标系：

- `XYZ`：三平面交点，单位 mm。
- `RPY`：从当前坐标系旋转矩阵转换的欧拉角，单位 deg。
- `X`：测试坐标系 X 轴在相机坐标系下的方向。
- `Z`：测试坐标系 Z 轴在相机坐标系下的方向。
- `residual`：三平面交点线性方程残差，单位 mm。

相对参考 delta：

- 启动后第一帧有效位姿会被锁定为参考。
- 后续输出当前位姿相对参考位姿的 XYZ/RPY 差值。

## 已知限制

- `filter_points_for_sensor()` 已裁切点云，算法假设输入单位为 mm。
- `project_points_to_image()` 不做畸变校正。
- SAM 打开后料盘区域更接近真实形状，但计算耗时更高。
- SAM 关闭时料盘 mask 会退化为检测框区域，预览会显示矩形区域。
- 三平面排序依赖底面参考轴和侧面 X 坐标均值，视角或场景变化过大时需要调整 `PlanePoseConfig`。
- 文档描述的是当前工程实现；实机效果仍受相机曝光、点云质量、料盘颜色和模型缓存影响。

## 最小验证

当前代码做过以下无硬件验证：

- 相关 Python 文件 `py_compile`。
- `CameraIntrinsics -> project_points_to_image()` 投影冒烟。
- 合成三平面点云估计和 `Axis` 位姿派生一致性冒烟。
- 2D 料盘 mask overlay 纯图像冒烟。

实机实时效果需要连接 Orbbec Gemini 305 后运行测试脚本确认。
