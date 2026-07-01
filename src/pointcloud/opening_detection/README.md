# opening_detection 模块说明

这个目录下的代码实现的是“托盘开口检测 + 开口相关掩码生成 + 抓取位姿估计”的完整几何流程。

模块入口主要有两个：

- `OpeningDetectionPipeline`：负责从图像和托盘掩码中找开口，并生成后续掩码与局部区域
- `GraspPoseEstimator`：负责基于开口和局部平面估计抓取位姿

## 1. 总体结论

这里**不是预训练模型推理链路**，核心实现是**传统计算机视觉 + 几何方法**。

具体来说：

- 开口检测主要使用 `cv2` 的灰度阈值、形态学、轮廓提取、最小外接旋转矩形和启发式打分
- 掩码扩展主要使用 `cv2` 的连通域、膨胀、开闭运算，以及基于灰度/边缘的区域生长
- 顶面法向量和抓取姿态主要使用点云平面拟合、PCA/协方差分解和像素射线与平面求交
- 唯一引入的 3D 算法库是 `open3d`，用于平面 RANSAC 和点云基础处理，不是深度学习模型

换句话说，这套逻辑是“传统 CV + 3D 几何”，不是“检测网络 + 回归网络”。

## 2. 开口检测逻辑

开口检测入口是 `OpeningDetectionPipeline.detect_opening()`，实际调用的是 `_detect_rect_opening_auto()`。

### 2.1 输入

它依赖三类输入：

- `rgb_bgr`：BGR 图像
- `tray_mask`：托盘区域掩码
- `hp_gray`：高精度灰度图，通常用于更稳定的局部对比判断

### 2.2 前提检查

先检查托盘掩码是否有效：

- 如果 `tray_mask` 的非零像素太少，直接报错
- 再从 `tray_mask` 计算包围盒，限制开口搜索范围

### 2.3 搜索 ROI

搜索并不是整图扫描，而是只在托盘包围盒的下半部偏前区域找开口：

- `x` 方向取托盘宽度的 `8% ~ 92%`
- `y` 方向取托盘高度的 `72% ~ 97%`

这说明作者默认开口大概率出现在托盘前立面附近，而不是任意位置。

### 2.4 候选生成

对 ROI 做以下处理：

1. BGR 转灰度
2. 高斯模糊
3. 取多个灰度阈值，阈值来源是局部灰度分位数
4. 对每个阈值构造二值暗区掩码
5. 结合 `roi_tray` 限制只保留托盘内部区域
6. 用横向较强的矩形结构元素做闭运算，再做开运算
7. 提取外轮廓

这里的思路是把“开口”当作托盘前立面上的一段明显暗槽或暗孔来找。

### 2.5 候选筛选

每个轮廓会经过一组规则过滤：

- 面积太小的不要
- 用 `cv2.minAreaRect()` 得到旋转矩形
- 长宽比必须落在合理范围
- 相对托盘宽高的比例也要合理
- 候选位置必须更偏下，符合开口位于前立面的先验

### 2.6 候选打分

通过多个启发式特征给候选打分，最后取最高分：

- 原始灰度暗度
- 高频灰度暗度
- 长宽比偏长的奖励
- 开口区域与周边环带的灰度对比
- 候选中心是否接近托盘中心线
- 候选是否足够靠下

最后输出 `OpeningDetection`：

- `center_uv`
- `bbox_xywh`
- `quad_uv`
- `score`

## 3. 位姿计算逻辑

位姿相关逻辑主要在 `pose_pipeline.py`，和 `opening_pipeline.py` 里的开口检测是分开的。

### 3.1 先估平面

`GraspPoseEstimator.estimate_plane()` 会对开口附近的局部点云做平面拟合：

- 点多时先做统计滤波
- 用 `open3d` 的 `segment_plane()` 做 RANSAC 平面拟合
- 得到平面模型 `n·x + d = 0`
- 再把法向量归一化，并统一朝向

这一步是纯几何拟合，不是学习模型。

### 3.2 由开口中心求抓取点

`compute_grasp()` 会把开口中心像素 `center_uv` 通过针孔相机模型变成一条射线，然后与平面求交：

- 射线由 `fx/fy/cx/cy` 构造
- 与平面求交得到 `grasp_point`

也就是说，抓取点不是直接在图像上回归出来的，而是由“像素中心 + 平面”几何算出来的。

### 3.3 由开口长边定抓取横轴

为了决定抓取姿态的水平朝向，代码会：

- 找出开口四边形四条边中最长的一条
- 取这条长边方向作为横向参考
- 在长边方向两侧偏移一点点
- 分别把两个像素点投到平面上
- 用两点连线方向作为 `x_axis`

这样得到的横轴与开口长边方向一致，更符合开口槽位的几何方向。

### 3.4 由参考法向量定抓取纵轴

`top_ref_normal` 是可选的顶部参考法向量：

- 如果传入了，就优先用它来构造 `y_axis`
- 如果没有，就回退到一个默认参考方向 `y_ref = [0, -1, 0]`

然后把 `y_axis` 投影到与 `x_axis` 正交的平面上，再做归一化。

### 3.5 构造右手坐标系

最后通过叉乘得到：

- `z_axis = x_axis × y_axis`
- 再反算一次 `y_axis = z_axis × x_axis`

最终 `rotation = [x_axis, y_axis, z_axis]` 组成旋转矩阵。

返回的抓取结果是：

- `grasp_point`：抓取点
- `pre_grasp_point`：沿 `z_axis` 外偏 80 mm 的预抓取点
- `rotation`：抓取姿态旋转矩阵

### 3.6 时序稳定

这个位姿估计器还提供两个稳定器：

- `stabilize_top_normal()`
- `stabilize_grasp_result()`

它们通过 EMA、角度门控和中值滤波平滑跨帧抖动，属于时序后处理，不改变核心几何求解方式。

## 4. 掩码与局部区域逻辑

`OpeningDetectionPipeline` 不只负责找开口，还会围绕开口生成几个后续模块会用到的区域。

### 4.1 邻近暗平面掩码

`_build_near_dark_plane_mask()` 的作用是找开口附近偏暗的局部平面区域：

- 先围绕开口构造一个环带 ROI
- 再从开口边界采种子点
- 用灰度差阈值 + 边缘阻断做区域生长
- 最后再做形态学清理和连通域选择

这通常用于提取开口周围的局部暗区域。

### 4.2 无孔顶面掩码

`_build_no_hole_top_plane_mask()` 的目标是找托盘前部的“无孔顶面”区域：

- 先构造一个更靠近开口下方的 ROI 多边形
- 如果有 `near_plane_mask`，就从其周边带再做一次区域生长
- 如果生长失败，就退化为边缘较弱区域的形态学筛选

这一步的作用是为顶面法向量估计和后续 `top_quad_uv` 拟合提供稳定区域。

### 4.3 顶点四边形拟合

`fit_rotated_quad()` 直接对 `no_hole_mask` 做：

- `cv2.findContours`
- `cv2.minAreaRect`
- `cv2.boxPoints`

输出一个旋转四边形，用作顶面区域的几何近似。

## 5. 这套方法到底是不是预训练模型

不是。

从代码看，这个目录下没有：

- PyTorch / TensorFlow 推理
- ONNX Runtime 推理
- 检测器权重加载
- 语义分割网络输出
- 关键点回归网络输出

它的判断依据全是：

- 灰度阈值
- 连通域
- 轮廓
- 旋转矩形
- 区域生长
- 平面拟合
- 像素射线求交

所以它属于传统方法链路。

## 6. 代码入口速查

- `src/pointcloud/opening_detection/opening_pipeline.py`
  - `OpeningDetectionPipeline.detect_opening()`
  - `OpeningDetectionPipeline.compute_mask_pipeline()`
  - `OpeningDetectionPipeline.estimate_top_plane_normal()`
  - `OpeningDetectionPipeline.fit_rotated_quad()`
- `src/pointcloud/opening_detection/pose_pipeline.py`
  - `GraspPoseEstimator.estimate_plane()`
  - `GraspPoseEstimator.compute_grasp()`
  - `GraspPoseEstimator.stabilize_top_normal()`
  - `GraspPoseEstimator.stabilize_grasp_result()`
- `src/pointcloud/opening_detection/types.py`
  - `OpeningDetection`
  - `PlaneResult`
  - `GraspResult`
  - `TrayMaskResult`

## 7. 适合后续维护的理解方式

可以把这套流程理解成三层：

1. **2D 开口发现**
   - 在图像里找暗槽开口
2. **局部掩码扩展**
   - 围绕开口生成邻近区域和无孔区域
3. **3D 位姿求解**
   - 用点云平面和相机几何把像素结果转成抓取位姿

如果后续要替换成学习模型，最适合替换的是“开口检测”这一层，但当前实现仍然完全可以靠传统 CV 运行。
