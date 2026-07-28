# Ball Pose Detection

## 日志约定

模块通过 Loguru 发送结构化事件，不自行创建文件 sink。服务入口启用日志后：

- `INFO`：记录 request、camera、实际 frame、先验数量、每种颜色候选数量、最终状态、匹配数量和总耗时；
- `WARNING`：记录某种颜色没有候选，或最终匹配数量少于先验数量；
- 不记录输入图像、深度图、mask、轮廓数组或三维点数据；
- 算法异常不在本模块重复捕获，由统一 service server 记录一次 `ERROR` 和堆栈。

## 单一职责

`ball_pose_detection` 根据一帧 RGBD 和球颜色/直径先验，输出各球的二维圆、相机坐标系三维圆心和估计直径。它不构造多球坐标系、不访问相机、不等待稳定帧，也不创建 RPC。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `types.py` | 帧协议、配置和内部观测结果 |
| `priors.py` | 球颜色、直径和模型位置先验 |
| `detector.py` | 颜色分割、轮廓几何筛选和深度反投影 |
| `service.py` | 将先验和检测结果组装为响应/debug |
| `protocol.py` | 请求、响应和 debug 协议 |

## 算法理论

### 颜色分割

输入 BGR 转换到 HSV，根据请求中每个球的 HEX 颜色动态换算 HSV 区间并生成 mask。
任意颜色跨越 HSV 首尾时都会自动拆成两个区间合并。模块不维护黄、红、绿等具名
颜色表；连通域按面积筛选后提取轮廓。

`color_hex` 是调用方为球指定的稳定身份标签，也是首次先验记录时的参考颜色。若
`BallPosePriorInfo.hsv_ranges` 非空，检测器优先使用该球标定得到的专属窄范围；
只有范围为空时才以 `color_hex` 动态生成参考宽范围。每个有效候选还会返回
轮廓内颜色像素的 `observed_hsv`，Hue 使用周期为 180 的圆均值，避免红色跨越
0/179 时得到错误色相。

### 圆形几何筛选

候选轮廓计算最小外接圆、面积、周长和填充率。圆形度定义为：

```text
circularity = 4πA / P²
```

圆形度、填充率和图像边界余量只用于单色候选内部排序，不能单独形成有效检测。

### 三维圆心

圆心附近有效深度先排序并按 `depth_trim_ratio` 去除两端异常值，再取稳健深度。针孔反投影为：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_mm
```

物理直径近似为：

```text
diameter_mm = 2 * radius_px * Z / ((fx + fy) / 2)
```

估计直径相对先验直径的误差超过 `max_diameter_error_ratio` 时，该色块直接判定为
`diameter_mismatch`，不进入三球组合评分。记录首次先验时没有球间位置先验，算法按
直径误差优先、颜色和轮廓分数次之的顺序选择候选。

### 三球相对位置联合评分

当请求恰好包含三个具有实际毫米尺度的 `model_center_mm` 先验时，算法枚举每种颜色
保留候选的笛卡尔积。对每个组合比较模型三条边长与相机坐标系检测三条边长：

```text
relative_error(i, j) =
    abs(detected_distance(i, j) - model_distance(i, j)) / model_distance(i, j)
```

刚体变换不改变球间距，因此该评分不依赖相机外参。算法先按最大边长误差、再按平均
边长误差排序，单球颜色/轮廓分数只用于几何误差相近时打破平局。任一边误差超过
`max_relative_distance_error_ratio` 的组合不可用；没有完整可用组合时，三球统一返回
`relative_geometry_mismatch` 未检出结果，不输出错误坐标。

`(0, 0, 0)/(1, 0, 0)/(0, 1, 0)` 这类不含实际毫米尺度的占位先验不会启用边长约束，
用于首次先验采集时仍逐球选择，但候选必须通过深度和物理直径硬校验。服务以
“请求包含球先验，但这些先验不含有效相对位置关系”判定先验采集模式；该模式必须
设置 `enable_debug=True`，否则直接拒绝请求。完全不携带球先验是独立的空先验模式，
不属于先验采集。

## 输入输出

`BallPoseDetectionRequest` 包含请求号、相机名、帧号、debug 开关和球先验。响应包含 `matched_count`、明确类型的 `BallDetectionInfo` 序列、实际帧号和 `debug_artifacts`。未检测到的坐标使用空元组表达，关闭 debug 时 `debug_artifacts` 为空元组。

`model_center_mm` 用于三球相对边长联合评分，不在本模块内构建多球坐标系。多球坐标系
仍由上层业务根据通过校验的球心结果建立。

先验记录脚本收集 30 个不同且完整的三球帧，用三球中心和直径组成的 12 维特征执行
MAD 异常剔除，至少保留 24 帧后才允许写入。球心、直径和实测 HSV 使用保留帧均值；
HSV 波动的 90% 分位数用于生成 Hue 半宽不超过 8 的每球窄范围。记录结果同时保存
`observed_color_hex`，便于人工核对现场真实颜色。

关闭 debug 时不复制 RGBD 图、不绘制轮廓叠加图，也不构造 debug 检测列表。先验
采集模式不允许关闭 debug；采集端必须把返回的 overlay 落盘用于人工核验。

## 成功与失败语义

- 成功：返回 `BallPoseDetectionResponse`，`detections` 与输入先验顺序一致。
- 单个球没有颜色、深度或直径合格候选：仍是成功响应；该项 `detected=False`，坐标字段为空元组，`status` 说明原因。
- 三球相对位置不一致：三个结果均为 `detected=False`、`status=relative_geometry_mismatch`，`matched_count=0`。
- 精确颜色范围缺失：根据请求中的 HEX 颜色生成参考宽范围，不视为协议错误；重新执行 30 帧先验记录后生成专属范围。
- 没有先验：`detections=()`，算法不会自行猜测目标颜色。
- 包含球先验但没有有效相对位置关系且关闭 debug：请求无效，服务拒绝执行。
- 输入帧、深度或内参不符合协议：算法抛出异常，由服务层转换为统一错误。
- RPC 超时、模型/服务不可用：由 `CameraPipelineServiceResponse.error` 承载，算法响应不重复定义 `error`。

## 调参建议

1. 先在现场照明下采集 HSV 分布，再调整通用 HEX→HSV 宽范围容差。
2. `min_component_area_px` 随分辨率和拍摄距离变化，应以最远有效球为下限标定。
3. `min_circularity` 越高越排斥遮挡和透视变形，过高会漏检。
4. `min_fill_ratio` 用于排除细环和破碎区域，应结合反光导致的孔洞调整。
5. `min_color_sample_pixels` 控制实测 HSV 所需的最少颜色像素。
6. `depth_trim_ratio` 用于抑制边缘混合深度；过大可能丢失小球有效点。
7. `min_depth_points` 必须与球像素尺寸和深度空洞率共同标定。
8. `min_center_distance_ratio` 用于识别不含实际尺度的占位位置先验；球间距小于
   最大先验直径的该倍数时，不启用相对位置评分。
9. `max_diameter_error_ratio` 是单球硬阈值，应该按直径估计误差而不是 HSV 漏检率标定。
10. `max_relative_distance_error_ratio` 是三球组合硬阈值，应该使用真实安装尺寸和现场深度噪声标定。
11. 调参时同时检查二维圆、有效深度点数、三维直径、实测 HSV 和球间几何关系。

## 局限性

- HSV 阈值对光照、曝光、白平衡和高光敏感。
- 圆模型不适合严重遮挡、强透视或非球形目标。
- 单点邻域深度容易受到透明、反光和边缘混合影响。
- 直径公式是针孔小角度近似，不是完整球面拟合。
- 当前不执行跨帧跟踪；三球先验只做候选组合评分和拒绝，不做非线性联合优化。
- 输出位于相机坐标系，外参误差不属于本模块处理范围。
