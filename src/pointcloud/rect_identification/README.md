# 矩形标记件识别算法设计

## 目标

本模块用于识别图像中具有颜色先验和位置先验的矩形标记件，并根据重建出的四个角点计算矩形中心像素坐标。

当前阶段先在 `test/pointcloud/rect_identification` 建立可运行入口完成采集、调试和验证；测试稳定后，算法代码沉淀到 `src/pointcloud/rect_identification`。

设计重点：

- 数据驱动：先验、配置、输入帧、过程结果和最终结果都通过明确数据结构传递。
- 最小算法单元：颜色分割、ROI 生成、边缘拟合、评分等步骤可以单独替换。
- 局部鲁棒验证：利用先验位置降低全图误检风险，并容忍相机定位偏差、光照变化和局部油污遮挡。
- 可调试：每个阶段保留可视化和中间结果，便于实验阶段快速定位失败原因。

## 设计背景

本识别任务不是开放场景下的通用矩形检测，而是强先验条件下的局部目标验证。已知信息包括：

- 绝对精确位置条件下采集得到的矩形四角点像素坐标。
- 矩形标记件颜色先验 RGB 值。
- 相机内参和畸变参数。
- 一幅图像中可能存在三个固定位置的并行识别任务。

工程现场中需要额外面对：

- 相机实际定位和理想定位之间存在平移、旋转或尺度偏差。
- 现场光照变化会改变 RGB 绝对值。
- 标记件表面可能有反光、阴影或油污遮挡。
- 背景中可能存在颜色接近的干扰区域。

因此本模块不采用“全图找最大矩形”的思路，而采用“先验 ROI 内多证据验证”的思路：先用位置先验限定搜索范围，再用颜色、边缘、几何和评分共同判断目标是否存在。

## 总体算法思路

推荐流程：

1. 输入先验四角点、颜色先验和相机内参。
2. 对输入图像做去畸变和共享预处理。
3. 基于先验四角点生成外扩 ROI。
4. 在 ROI 内做 Lab、HSV 和归一化 RGB 的颜色相似度计算。
5. 对颜色支持区域做形态学处理和连通域统计。
6. 结合颜色候选和先验位置生成初始旋转矩形。
7. 在初始矩形四边附近建立边缘搜索窄带。
8. 对四条边分别采样边缘点并做鲁棒直线拟合。
9. 由四条拟合直线求交得到四角点。
10. 根据四角点计算中心点。
11. 根据中心偏移、面积、角度、长宽比、颜色支持率、边缘支持率和拟合残差综合评分。
12. 输出 `detected`、`missing` 或 `uncertain`，并记录失败原因和调试数据。

三项并行检测任务共享同一个 `PreparedFrame`，各自使用独立的 `RectMarkerPrior` 生成 ROI 并完成后续判断。实验阶段优先顺序执行，算法稳定且存在性能瓶颈后再并行化。

## 设计原则

### 以先验验证替代全图搜索

矩形标记件的理想位置已知，算法应优先验证先验位置附近是否存在目标，而不是在全图内重新寻找所有矩形。这样可以显著降低背景误检概率，也能减少计算量。

### 颜色是证据，不是唯一判据

颜色分割适合作为候选生成和支持率计算，但不能作为唯一判定依据。现场光照、反光和油污都会破坏固定阈值，因此颜色应输出相似度和支持率，而不是只输出一个硬二值结果。

### `minAreaRect` 只做初始化

`minAreaRect` 对破碎 mask、局部遮挡和背景连通干扰敏感。它可以提供初始旋转矩形，帮助确定四边搜索区域，但最终角点应来自四条边的局部边缘拟合结果。

### 边缘拟合按四条边独立进行

矩形内部被油污遮挡时，整体轮廓可能不完整，但四条边仍可能存在局部可见段。按边建立窄带并独立拟合，可以比全局轮廓拟合更好地利用残缺边缘。

### 评分必须可解释

最终结果不能只给 `True` 或 `False`。每次检测都应输出评分分解和失败原因，方便在现场样本上判断问题来自颜色、边缘、几何、相机偏位还是先验数据错误。

### 数据结构优先稳定

实验阶段算法细节会频繁替换，因此应先稳定数据契约。颜色算法、边缘算法和评分算法可以替换，但它们的输入输出对象应保持一致。

## 关键设计原因

### 为什么先去畸变

先验角点、ROI、边缘搜索带和最终中心点都在像素坐标中计算。若图像存在明显畸变，而先验点和当前图像不在同一坐标空间，边缘拟合会出现系统性偏差。采集先验和检测流程应统一到同一坐标空间，推荐使用 `undistorted_pixel`。

### 为什么使用 Lab、HSV 和归一化 RGB

RGB 绝对值受亮度影响明显。Lab 更适合表达感知颜色差异，HSV 可以弱化 Value 通道影响，归一化 RGB 能抵御整体明暗变化。三者组合可以让颜色证据更稳，但最终仍要通过几何和边缘交叉验证。

### 为什么不只取最大连通域

油污遮挡会让真实矩形区域被切碎，背景相似色区域也可能形成更大连通域。只取最大连通域容易把背景当目标，或在遮挡时丢失真实目标。候选区域需要结合先验中心、面积比例、颜色支持率和边缘证据综合判断。

### 为什么需要 `uncertain`

现场图像中会出现部分证据成立但不足以稳定输出中心点的情况。直接输出失败会丢失诊断信息，直接输出成功会污染后续定位。`uncertain` 用于表达需要复拍、重试或人工检查的中间状态。

### 为什么先顺序执行三个任务

三个 ROI 检测任务理论上可以并行，但实验阶段的主要目标是调试算法质量。顺序执行更容易记录日志、保存中间图和定位失败原因。并行化应放在算法稳定之后。

## 目录规划

```text
src/pointcloud/rect_identification/
  README.md
  types.py
  io.py
  preprocess.py
  roi.py
  color_segmentation.py
  morphology.py
  edge_fit.py
  scoring.py
  pipeline.py
  visual_debug.py

test/pointcloud/rect_identification/
  collect_rect_marker_prior.py
  test_rect_marker_detection.py
  data/
```

职责边界：

- `src/pointcloud/rect_identification`：长期维护的算法、数据结构、IO 和调试可视化工具。
- `test/pointcloud/rect_identification`：面向真实相机和样本图像的实验入口，不承载长期算法逻辑。
- `test/pointcloud/rect_identification/data`：实验阶段的先验 JSON、采样图像和检测输出。

## 数据设计

### CameraCalibration

相机内参与图像坐标空间说明。

字段建议：

- `image_width: int`
- `image_height: int`
- `fx: float`
- `fy: float`
- `cx: float`
- `cy: float`
- `distortion: list[float]`
- `distortion_model: str`
- `coordinate_space: str`

`coordinate_space` 用于明确先验角点所在图像空间，建议优先使用 `undistorted_pixel`。如果采集先验时使用原始图像，则必须保存为 `raw_pixel`，后续检测前需要统一坐标空间。

### RectMarkerPrior

单个矩形标记件的稳定先验。

字段建议：

- `marker_id: str`
- `corners_px: list[tuple[float, float]]`
- `corner_order: str`
- `rgb_prior: tuple[int, int, int]`
- `rgb_median: tuple[int, int, int]`
- `rgb_mad: tuple[float, float, float]`
- `expected_area_px: float`
- `expected_angle_deg: float`
- `expected_aspect_ratio: float`
- `max_center_shift_px: float`
- `max_angle_delta_deg: float`
- `min_area_ratio: float`
- `max_area_ratio: float`
- `roi_expand_px: int`
- `roi_expand_ratio: float`

四角点顺序必须固定，推荐 `top_left, top_right, bottom_right, bottom_left`。采集入口负责将鼠标点击顺序规范化，不把无序点传给算法。

### RectMarkerSetPrior

一幅图像中多个矩形标记件的先验集合。

字段建议：

- `schema_version: str`
- `created_at: str`
- `source_image: str`
- `camera: CameraCalibration`
- `markers: list[RectMarkerPrior]`

一幅图像需要并行判断三个位置是否存在矩形框时，使用三个 `RectMarkerPrior` 描述三项独立检测任务。

### RectMarkerDetectionConfig

检测配置总入口，内部建议拆成子配置。

- `PreprocessConfig`：去畸变、白平衡、CLAHE、模糊核。
- `RoiConfig`：ROI 外扩像素、外扩比例、最大平移容忍。
- `ColorConfig`：Lab 距离阈值、HSV Hue 容忍、颜色支持率阈值。
- `MorphologyConfig`：开闭运算核大小、最小连通域面积。
- `EdgeFitConfig`：边缘搜索带宽、Canny 阈值、RANSAC 残差阈值、最小边缘覆盖率。
- `ScoreConfig`：评分权重、硬失败阈值、最终通过阈值。
- `DebugConfig`：是否保存中间图、结果叠加图和过程 JSON。

### PreparedFrame

一帧图像的共享预处理结果。多个矩形检测任务共用同一个 `PreparedFrame`。

字段建议：

- `raw_bgr: np.ndarray`
- `undistorted_bgr: np.ndarray`
- `hsv: np.ndarray`
- `lab: np.ndarray`
- `gray: np.ndarray`
- `gradient: np.ndarray`
- `camera: CameraCalibration`

### MarkerDetectionResult

单个标记件检测结果。

字段建议：

- `marker_id: str`
- `status: str`
- `detected: bool`
- `center_px: tuple[float, float] | None`
- `corners_px: list[tuple[float, float]]`
- `score: float`
- `score_breakdown: MarkerScoreBreakdown`
- `failure_reasons: list[str]`

`status` 建议使用三类：

- `detected`：硬约束和综合评分均通过。
- `missing`：颜色和边缘证据均不足，可判断目标不存在。
- `uncertain`：存在部分证据，但质量不足，不能稳定确认。

## 先验采集入口

入口文件：

```text
test/pointcloud/rect_identification/collect_rect_marker_prior.py
```

职责：

1. 打开 Orbbec 彩色图像预览。
2. 按键冻结当前帧。
3. 为每个标记件输入或选择 `marker_id`。
4. 鼠标点击四个角点。
5. 将角点规范化为固定顺序。
6. 在矩形内部收缩区域采样颜色。
7. 保存先验 JSON、原始图像、标注图和可选深度图。

推荐输出：

```text
test/pointcloud/rect_identification/data/prior_sessions/
  20260615_153000/
    color.png
    depth_u16.png
    annotated.png
    prior.json
```

采集阶段建议直接在去畸变图像上点击角点，并在 JSON 中记录 `coordinate_space = "undistorted_pixel"`。这样后续检测流程无需在先验点和检测图像之间做额外坐标转换。

## 检测实验入口

入口文件：

```text
test/pointcloud/rect_identification/test_rect_marker_detection.py
```

职责：

1. 读取先验 JSON。
2. 读取离线图像或实时相机帧。
3. 构建 `PreparedFrame`。
4. 对每个 `RectMarkerPrior` 独立执行检测。
5. 汇总三个位置的检测结果。
6. 显示 ROI、颜色 mask、边缘点、拟合边线、最终角点和中心点。
7. 保存检测输出 JSON 和调试叠加图。

实验阶段优先顺序执行三个检测任务，便于调试。只有在算法稳定且确实存在帧率瓶颈后，再将单个标记件检测任务放入线程池并行执行。

## 算法流水线

### 1. 全帧预处理

输入原始 BGR 图像和相机内参，输出 `PreparedFrame`。

处理内容：

- 图像尺寸校验。
- 按配置执行去畸变。
- 生成 HSV、Lab、灰度图和梯度图。
- 可选执行白平衡、CLAHE 或轻量滤波。

该步骤只对一帧图像执行一次，三个标记件共享结果。

### 2. ROI 生成

输入 `PreparedFrame` 和 `RectMarkerPrior`，输出当前标记件的局部 ROI。

ROI 依据先验四角点外扩生成，外扩量由固定像素和矩形尺寸比例共同决定：

- 平移误差通过 `roi_expand_px` 覆盖。
- 旋转和尺度误差通过 `roi_expand_ratio` 覆盖。
- ROI 裁剪必须保留其在全图中的偏移量，后续角点需要还原到全图坐标。

### 3. 颜色相似度

输入 ROI 图像和颜色先验，输出颜色支持结果。

建议同时使用：

- Lab 颜色距离，降低光照变化影响。
- HSV Hue 和 Saturation 约束，避免亮度 Value 过度影响。
- 归一化 RGB 或色度比例，辅助抵御整体明暗变化。

输出不应只有二值 mask，还应包含颜色相似度热力图和颜色支持率。

### 4. 形态学与候选区域

输入颜色 mask，输出清理后的候选区域。

处理内容：

- 小区域过滤。
- 闭运算连接轻微断裂。
- 开运算去除孤立噪声。
- 连通域统计。

油污遮挡会导致真实区域破碎，因此不能只取最大连通域作为唯一依据。候选区域需要结合先验中心、面积、颜色支持和几何形状共同判断。

### 5. 初始矩形

输入候选区域和先验，输出初始旋转矩形。

候选来源：

- 颜色 mask 主区域。
- 先验四角点对应的预测矩形。
- `minAreaRect` 生成的初始旋转矩形。

`minAreaRect` 只作为边缘搜索初始化，不作为最终角点输出。

### 6. 边缘搜索

输入初始矩形和梯度图，输出四条边附近的边缘点。

处理方式：

- 沿初始矩形四条边建立窄带搜索区。
- 在每条边的窄带内采样梯度强点。
- 每条边单独统计边缘点数量、覆盖长度和方向一致性。

这种局部边缘搜索比全图 Canny 更稳定，也更符合强先验场景。

### 7. 四边拟合

输入四组边缘点，输出四条拟合直线和四个角点。

拟合建议：

- 使用 RANSAC 或 Huber 思路抵御污渍、反光和局部缺边。
- 每条边独立拟合并计算残差。
- 四条直线求交得到四个角点。
- 校验四边形是否自交、面积是否合理、长宽比是否接近先验。

中心点建议同时计算：

- 四角点均值。
- 两条对角线交点。

两者差异可以作为角点稳定性指标。

### 8. 综合评分

评分拆成硬约束和软评分。

硬失败条件：

- ROI 严重越界。
- 四边形自交。
- 面积比例超限。
- 长宽比严重异常。
- 中心偏移超过最大容忍。
- 有效边数量不足。

软评分项：

- 中心偏移分。
- 面积一致性分。
- 角度一致性分。
- 长宽比一致性分。
- 颜色支持率分。
- 边缘支持率分。
- 线拟合残差分。
- 四角点稳定性分。

最终输出 `detected`、`missing` 或 `uncertain`，并记录 `failure_reasons`。

## 最小算法单元接口

后续实现时建议保持如下函数边界：

```text
prepare_frame(image_bgr, calibration, config) -> PreparedFrame
build_marker_roi(prepared_frame, marker_prior, config) -> MarkerRoi
compute_color_support(marker_roi, marker_prior, config) -> ColorSegmentationResult
clean_color_mask(color_result, config) -> MorphologyResult
build_initial_rect(morphology_result, marker_prior, config) -> InitialRectResult
extract_edge_points(prepared_frame, initial_rect, config) -> EdgeSupportResult
fit_marker_quad(edge_support, initial_rect, config) -> QuadFitResult
score_marker(quad_fit, color_result, marker_prior, config) -> MarkerScoreBreakdown
detect_one_marker(prepared_frame, marker_prior, config) -> MarkerDetectionResult
detect_marker_set(prepared_frame, marker_set_prior, config) -> list[MarkerDetectionResult]
```

每个算法单元只使用显式输入，不读取全局变量，不直接访问相机，不直接写文件。相机采集、文件读写和可视化由入口脚本或 IO 层负责。

## 调试输出

实验阶段建议保存以下内容：

- 原图和去畸变图。
- 每个标记件的 ROI 图。
- Lab/HSV 颜色支持热力图。
- 颜色 mask 和形态学后 mask。
- 初始旋转矩形。
- 四边搜索带。
- 边缘点。
- 拟合直线。
- 最终四角点和中心点。
- 评分分解 JSON。

调试输出目录建议：

```text
test/pointcloud/rect_identification/data/debug_runs/
  20260615_160000/
    result.json
    overview.png
    marker_1_roi.png
    marker_1_mask.png
    marker_1_edges.png
    marker_1_fit.png
```

## 迁移原则

实验验证完成后按以下规则迁移：

1. 数据结构迁移到 `src/pointcloud/rect_identification/types.py`。
2. 先验 JSON 读写迁移到 `io.py`。
3. 全帧预处理迁移到 `preprocess.py`。
4. ROI 生成迁移到 `roi.py`。
5. 颜色分割迁移到 `color_segmentation.py`。
6. 形态学处理迁移到 `morphology.py`。
7. 边缘搜索和四边拟合迁移到 `edge_fit.py`。
8. 评分逻辑迁移到 `scoring.py`。
9. `detect_one_marker` 和 `detect_marker_set` 迁移到 `pipeline.py`。
10. 调试叠加图生成迁移到 `visual_debug.py`。

迁移后，`test/pointcloud/rect_identification` 中的脚本只保留入口职责：采集数据、读取配置、调用 pipeline、展示和保存结果。

## 避坑建议

- 不要把固定 HSV 阈值当成长期方案。它适合快速验证，但现场光照变化会很快暴露问题。
- 不要在原始图像采集先验、在去畸变图像执行检测，除非明确做了坐标转换。
- 不要让测试入口直接沉淀算法逻辑。入口只负责采集、读取、调用、展示和保存。
- 不要用无结构 `dict` 传递先验和配置。调参项较多时应拆成 dataclass。
- 不要让任意算法单元隐式读取全局配置。所有参数应来自显式输入。
- 不要在颜色 mask 破碎时直接判定目标不存在，应检查边缘证据和几何一致性。
- 不要在边缘不足时强行输出中心点，应返回 `uncertain` 并记录边缘支持不足。
- 不要把三个标记件任务的中间状态混在一起。每个 `marker_id` 应有独立调试输出。
- 不要只保存最终叠加图。实验阶段必须保存 ROI、mask、边缘点、拟合线和评分 JSON。
- 不要在未接入真实相机和现场样本前声称算法具备工程鲁棒性。

## 当前阶段约束

- 不保持旧接口兼容性，优先把数据结构设计清楚。
- 不在 UI 或测试入口中沉淀算法逻辑。
- 不使用无结构 `dict` 长距离透传参数。
- 不使用隐藏全局状态控制算法行为。
- 不把 `minAreaRect` 结果直接作为最终角点。
- 不只输出布尔结果，必须输出评分分解和失败原因。
- 未连接真实相机或未采集现场数据前，不能声称算法已验证工程鲁棒性。
