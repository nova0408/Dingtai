# Orbbec 三平面位姿测试方案说明

## 目标

本文档记录 Orbbec Gemini 305 三平面位姿测试方案，覆盖当前优化脚本和参考脚本：

- 当前优化脚本：`test/pointcloud/three_plane/test_orbbec_three_plane_pose_optimized_pipeline.py`
- 参考脚本：`test/pointcloud/three_plane/test_orbbec_realtime_three_plane_pose_frame.py`

方案目标是在 Orbbec Gemini 305 实时点云中检测三个结构平面，并以三平面交点为原点计算测试坐标系，同时排除料盘区域对平面拟合的干扰。测试重点是观察当相机视角转动后，计算出的测试坐标系是否仍稳定落在同一空间位置。

当前实现把可复用能力沉淀到 `src`：

- `src/rgbd_camera`：Orbbec 会话、点云裁切、相机内参/外参数据结构。
- `src/pointcloud/tray_detection/detector.py`：料盘 zero-shot 检测与点云排除。
- `src/pointcloud/tray_detection/projection.py`：点云到图像的投影。
- `src/pointcloud/three_plane_pose.py`：三平面分割、平面排序、坐标系计算。
- `src/pointcloud/three_plane_types.py`：三平面配置、结果结构和位姿稳定器。
- `test/pointcloud/three_plane/test_orbbec_three_plane_pose_optimized_pipeline.py`：实时采集、线程调度、Open3D/CV2 预览和日志。
- `src/pointcloud/motion_shift.py`：跨模块复用的平移估计与 mask 平移工具。

## 脚本关系

### 参考脚本

`test_orbbec_realtime_three_plane_pose_frame.py` 是较早的验证脚本，包含完整实验逻辑：

- 三平面检测。
- 料盘 zero-shot 排除。
- 2D 平面区域绘制。
- 3D 平面 patch 和坐标系绘制。
- 参考坐标系锁定和相对 delta 日志。
- 较多实验参数和本地辅助函数。

该脚本适合作为行为参考，尤其是 2D 料盘区域绘制、三平面可视化和日志格式。但它把较多算法、可视化和测试辅助逻辑放在同一文件中，长期维护成本较高。

### 当前优化脚本

`test_orbbec_three_plane_pose_optimized_pipeline.py` 是当前推荐使用的性能优化测试入口。它把可复用算法迁移到 `src`，测试脚本只负责实时采集、队列调度、预览和日志。

当前脚本继承参考脚本的核心行为：

- 仍然使用三平面交点作为测试坐标系原点。
- 仍然用底面法线确定 Z 方向。
- 仍然输出当前 XYZ/RPY 和相对参考 delta。
- 仍然在 2D 和 3D 中预览检测结果。
- 料盘绘制按实际 mask 区域显示，默认启用 SAM，避免只画检测框。
- 料盘检测改为异步独立线程，主计算帧不阻塞等待检测完成。
- 在异步检测间隙使用图像平移估计对历史托盘 mask 进行预测补偿。
- 通过帧索引同步保护避免“未来帧检测结果”回灌到历史计算帧。

当前脚本做出的结构调整：

- 三平面算法进入 `src/pointcloud/three_plane_pose.py`。
- 三平面结果结构进入 `src/pointcloud/three_plane_types.py`。
- 料盘识别进入 `src/pointcloud/tray_detection/detector.py`。
- 相机内参/外参封装进入 `src/rgbd_camera/orbbec_models.py`。
- 实时脚本只保留测试管线和预览职责。

如果后续需要修改算法，应优先修改 `src`；如果只是改窗口、颜色、日志或临时测试策略，应修改当前测试脚本。

## 运行方式

推荐直接运行当前优化脚本：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe test\pointcloud\three_plane\test_orbbec_three_plane_pose_optimized_pipeline.py
```

常用参数：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe test\pointcloud\three_plane\test_orbbec_three_plane_pose_optimized_pipeline.py `
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

参考脚本仍可用于对照行为：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe test\pointcloud\three_plane\test_orbbec_realtime_three_plane_pose_frame.py
```

建议只在需要对比旧行为、排查预览差异或确认迁移结果时运行参考脚本。

## 算法思路

三平面位姿估计基于一个明确的几何假设：目标场景中存在一个底面和两个倾斜侧面，三者可通过局部点云拟合为三个平面。三个平面的交点作为测试坐标系原点，底面法线提供 Z 轴方向，X 轴由固定参考方向投影到底面切平面内得到。

核心设计点：

- 坐标系必须有固定参考方向，避免平面法线符号翻转导致预览坐标系跳变。
- Z 轴先由底面法线确定，再按 `frame_z_hint` 修正正负方向。
- X 轴默认使用 `frame_x_hint` 投影到底面切平面内，而不是完全依赖两个侧面法线叉乘。
- `Axis` 负责生成右手坐标系，避免手工拼接旋转矩阵。
- `CoordinateFramePose` 只保存 `Axis`、RPY 和残差；`Transform`、原点数组和旋转矩阵都从 `Axis` 派生。
- 位姿平滑只使用最近实际完成计算的帧，最多 15 帧，不缓存被丢弃的相机帧。

料盘排除的核心思路：

- 先在 2D 彩色图或投影图上识别料盘区域。
- 将 3D 点云投影到图像坐标。
- 位于料盘 mask 内的点标记为排除点。
- 三平面拟合阶段不使用这些点，避免料盘被误识别为结构平面。

性能优化的核心思路：

- 预览和计算分线程。
- 计算队列最大长度为 1。
- 上一帧计算未完成时，新帧直接丢弃。
- 结果队列保留最新结果，避免旧结果阻塞预览。
- 3D 预览点云做点数上限控制。
- 托盘检测单独线程异步运行，主计算帧只消费“最新已完成快照”。
- 托盘快照包含参考灰度图，计算帧通过 phase correlation 估计平移并预测补偿 mask。

## 数据流

主流程如下：

1. `Gemini305` 启动 Orbbec 会话。
2. `session.get_projection_intrinsics()` 读取当前投影内参。
3. `session.calculate_points_from_frames()` 从 SDK 帧生成点云。
4. `session.filter_points_for_sensor()` 使用 Gemini 305 默认视锥裁切点云。
5. 预览线程持续更新原始点云。
6. 若计算线程空闲，将当前帧放入单元素队列。
7. 托盘线程异步执行 `TrayPointExcluder.detect`，更新托盘快照。
8. 计算线程读取托盘快照，估计当前帧相对快照帧平移，先平移 mask 再做点云排除。
9. 计算线程执行三平面检测和坐标系估计。
10. 预览线程消费最新结果，更新 2D overlay、3D 平面点云、坐标系和日志。

计算队列是单元素模式：如果上一帧计算尚未完成，当前帧直接丢弃，不进入计算队列。这样可以避免积压旧帧，保持预览和计算结果尽量接近实时状态。

## 管线过程

当前优化脚本的管线可以分为四层。

### 1. 相机层

入口在 `Gemini305`：

1. 启动 Orbbec Pipeline。
2. 选择深度流和彩色流 profile。
3. 启用彩色对齐。
4. 读取 SDK 原生 `OBCameraParam`。
5. 通过 `get_projection_intrinsics()` 输出 `CameraIntrinsics`。
6. 创建 `PointCloudFilter`。

相机层输出：

- 当前帧集合。
- 裁切后的点云 `(N, 3)` 或 `(N, 6)`，单位 mm。
- 用于 2D 投影的 `CameraIntrinsics`。

### 2. 预览与调度层

入口在 `_run_pipeline()`：

1. 主线程持续等待相机帧。
2. 调用 `_capture_filtered_points()` 得到裁切点云。
3. 更新 Open3D 原始点云。
4. 若计算线程空闲且达到最小提交间隔，则把当前帧打包为 `CaptureJob`。
5. 同时提交 `TrayDetectJob`（仅用于异步托盘识别快照刷新）。
6. 若对应队列繁忙则丢弃旧任务，仅保留最新任务。
7. 若计算线程繁忙或队列非空，则丢弃当前帧。
8. 消费 `PipelineResult`，刷新 2D/3D 结果。

### 3. 托盘异步层

入口在 `_tray_worker_loop()`：

1. 从 `tray_job_queue` 取最新 `TrayDetectJob`。
2. 调用 `TrayPointExcluder.detect(base_bgr)` 获取 2D 托盘检测结果。
3. 同步生成托盘快照参考灰度图（缩放）用于平移预测。
4. 写入 `TrayDetectionSnapshot(frame_idx, detections, detect_ms, tracking_gray, tracking_scale)`。

该层只做 2D 识别，不做点云投影与排除。

### 4. 计算层

入口在 `_run_compute_job()`：

1. 从 `CaptureJob.points` 提取 `xyz` 和 `rgb`。
2. 使用 `project_points_to_image(xyz, job.intrinsics)` 得到 `uv` 和 `valid_proj`。
3. 构造 2D 基底图：优先使用彩色帧，否则用点云颜色栅格化。
4. 读取托盘快照，执行 `_build_excluded_mask_from_snapshot()`：
   - 若快照帧号晚于当前计算帧，直接丢弃快照（索引保护）。
   - 若快照滞后，先做图像平移估计，再平移历史 mask。
   - 将平移后的 mask 投影到当前点云，得到 `excluded_mask`。
5. 执行 `estimate_three_plane_pose()`。
6. 绘制 2D overlay。
7. 返回 `PipelineResult`。

### 5. 可视化层

2D 可视化：

- 使用 `TrayDetection.mask` 做半透明红色区域叠加。
- 从实际 mask 提取轮廓。
- 显示料盘置信度和排除点数。
- 显示平面分配点数、当前 XYZ/RPY。

3D 可视化：

- 原始裁切点云。
- 三平面标签点云。
- 当前坐标系 frame。
- 当前原点 marker。

预览颜色由测试脚本定义，不写入 `PlanePatch` 或算法层。

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

料盘排除由 “异步检测 + 同步投影” 两段组成：

1. 托盘线程调用 `TrayPointExcluder.detect()` 只输出 2D `TrayDetection` 列表。
2. 计算线程读取快照帧号，若快照晚于当前帧则弃用（避免时间错乱）。
3. 快照滞后时，调用 `src/pointcloud/motion_shift.py` 做 `phaseCorrelate` 平移估计。
4. 对快照中的 `mask` 做平移补偿后，再投影回当前帧点云生成 `(N,) bool` 排除掩码。
5. `estimate_three_plane_pose()` 对排除点标记为 `-2`，不参与 RANSAC 和 PCA。

2D 预览中料盘区域按 `TrayDetection.mask` 实际区域半透明填充；只有关闭 SAM 时，该 mask 才会退化为检测框区域。

## 偏移估计复用设计

当前项目已把图像平移估计抽到 `src/pointcloud/motion_shift.py`，统一提供：

- `prepare_tracking_gray()`：生成缩放灰度跟踪图。
- `estimate_phase_shift()`：基于 `cv2.phaseCorrelate` 估计平移（含响应阈值与可选限幅）。
- `warp_mask()`：按平移量高效平移 mask（最近邻插值）。

复用方：

- `test_orbbec_three_plane_pose_optimized_pipeline.py`：托盘快照补偿与实时排除。
- `src/pointcloud/tray_detection/pipeline.py`：通用托盘管线运动补偿。

这样做的收益：

- 避免多个模块维护不同的平移估计实现。
- 降低后续“平面检测/位姿流程也要接入偏移估计”的接入成本。
- 统一性能优化入口（缩放策略、响应阈值、限幅策略）。

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

脚本分为预览线程、托盘线程和计算线程：

- 预览线程：相机取帧、点云裁切、Open3D 原始点云刷新、CV2 2D 图显示、计算任务提交。
- 托盘线程：托盘 2D 检测与快照更新。
- 计算线程：托盘快照补偿、点云排除、三平面估计、2D overlay 生成。

队列策略：

- `job_queue` 最大长度为 1。
- `tray_job_queue` 最大长度为 1。
- `result_queue` 最大长度为 2。
- 计算线程忙时，预览线程丢弃当前帧。
- 托盘线程忙时，预览线程只保留最新托盘任务。
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

`TrayDetectionConfig` 的模型和阈值默认值定义在 `src/pointcloud/tray_detection/types.py` 字段上，不再通过同文件 `DEFAULT_*` 二次包装。

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
- 当托盘检测明显慢于实时帧率时，显示与排除依赖“旧快照 + 偏移预测”，极端快速运动下仍可能出现边界偏差。
- 平移预测当前仅建模 2D 平移，不建模尺度变化和旋转。
- 三平面排序依赖底面参考轴和侧面 X 坐标均值，视角或场景变化过大时需要调整 `PlanePoseConfig`。
- 文档描述的是当前工程实现；实机效果仍受相机曝光、点云质量、料盘颜色和模型缓存影响。

## 最小验证

当前代码做过以下无硬件验证：

- 相关 Python 文件 `py_compile`。
- `CameraIntrinsics -> project_points_to_image()` 投影冒烟。
- 合成三平面点云估计和 `Axis` 位姿派生一致性冒烟。
- 2D 料盘 mask overlay 纯图像冒烟。

实机实时效果需要连接 Orbbec Gemini 305 后运行测试脚本确认。

