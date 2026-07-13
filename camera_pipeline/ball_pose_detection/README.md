# Ball Pose Detection

## 单一职责

`ball_pose_detection` 根据一帧 RGBD 和球颜色/尺寸先验，输出各球的二维圆、相机坐标系三维圆心和估计半径。它不构造多球坐标系、不访问相机、不等待稳定帧，也不创建 RPC。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `types.py` | 帧协议、配置和内部观测结果 |
| `priors.py` | 球颜色、半径和模型位置先验 |
| `detector.py` | 颜色分割、轮廓几何筛选和深度反投影 |
| `service.py` | 将先验和检测结果组装为响应/debug |
| `protocol.py` | 请求、响应和 debug 协议 |

## 算法理论

### 颜色分割

输入 BGR 转换到 HSV，根据每种 HEX 颜色映射的 HSV 区间生成 mask。红色跨越 HSV 首尾，因此使用两个区间合并。连通域按面积筛选后提取轮廓。

### 圆形几何筛选

候选轮廓计算最小外接圆、面积、周长和填充率。圆形度定义为：

```text
circularity = 4πA / P²
```

圆形度、填充率、图像边界余量和候选间距离共同用于排除噪声及重复球。

### 三维圆心

圆心附近有效深度先排序并按 `depth_trim_ratio` 去除两端异常值，再取稳健深度。针孔反投影为：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_mm
```

物理半径近似为：

```text
radius_mm = radius_px * Z / ((fx + fy) / 2)
```

## 输入输出

`BallPoseDetectionRequest` 包含请求号、相机名、帧号、debug 开关和球先验。响应包含 `matched_count`、明确类型的 `BallDetectionInfo` 序列、实际帧号和 `debug_artifacts`。未检测到的坐标使用空元组表达，关闭 debug 时 `debug_artifacts` 为空元组。

`model_center_mm` 仅随先验传递，当前检测算法不使用它构建位姿。多球坐标系由上层业务根据球心结果建立。

关闭 debug 时不复制 RGBD 图、不绘制轮廓叠加图，也不构造 debug 检测列表。

## 调参建议

1. 先在现场照明下采集 HSV 分布，再调整 `color_ranges`。
2. `min_component_area_px` 随分辨率和拍摄距离变化，应以最远有效球为下限标定。
3. `min_circularity` 越高越排斥遮挡和透视变形，过高会漏检。
4. `min_fill_ratio` 用于排除细环和破碎区域，应结合反光导致的孔洞调整。
5. `depth_trim_ratio` 用于抑制边缘混合深度；过大可能丢失小球有效点。
6. `min_depth_points` 必须与球像素尺寸和深度空洞率共同标定。
7. `min_center_distance_ratio` 控制相邻同色候选去重。
8. 调参时同时检查二维圆、有效深度点数、三维半径和球间几何关系。

## 局限性

- HSV 阈值对光照、曝光、白平衡和高光敏感。
- 圆模型不适合严重遮挡、强透视或非球形目标。
- 单点邻域深度容易受到透明、反光和边缘混合影响。
- 半径公式是针孔小角度近似，不是完整球面拟合。
- 当前不执行跨帧跟踪，也不利用模型球间距离联合优化。
- 输出位于相机坐标系，外参误差不属于本模块处理范围。
