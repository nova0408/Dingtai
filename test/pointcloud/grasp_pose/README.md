# Grasp Pose 方案说明（Gemini 305 实时）

## 目录

- 脚本：`test_orbbec_realtime_estimate_grasp_pose_pipeline.py`
- 文档：`README.md`

## 目标

在 Gemini 305 实时数据上，稳定估计抓取位姿。核心要求：

- 先分割托盘，再检测开口。
- 基于开口构造附近平面与上平面。
- 输出稳定坐标系，减少方向抖动。
- 实时预览支持 RGB 与高反差图并排查看。

## 当前脚本主流程

1. 相机采集：Gemini 305，点云 + 彩色帧。
2. 托盘分割：优先 zero-shot（来自 `test_orbbec_realtime_plane_segmentation_zero_shot`）。
3. 开口检测：
   - 在托盘前立面 ROI 内检测。
   - 基于 `estimate_grasp.py` 风格（阈值 + 形态学 + 候选评分），高反差灰度作为辅助得分。
   - 保留细长几何约束。
4. 附近平面（Opening Plane）：
   - 以开口周围 ring 区域为约束。
   - 在高反差域中做邻域生长（连续性与边缘约束）。
5. 上平面（Top Plane）：
   - 从附近平面邻接区域生成。
   - 输出规则四边形（旋转矩形）用于显示。
6. 点云平面法线：
   - 由 top plane 对应全部点云点估计法线（带轻量离群点截断）。
7. 姿态构造：
   - `X` 由开口长边方向给出，并用参考 `(+1,0,0)` 做符号消歧。
   - `Y` 由 top plane 法线主导，用参考 `(0,-1,0)` 只做正负号消歧。
   - `Z` 由 `X/Y` 回构，保证正交稳定。

## 预览

- 单窗口并排：左 RGB，右高反差图。
- 叠加标注：
  - `Tray`
  - `Opening`
  - `Center`
  - `Opening Plane`
  - `Top Plane`

## 运行方式

在项目根目录执行：

```bash
python test/pointcloud/grasp_pose/test_orbbec_realtime_estimate_grasp_pose_pipeline.py
```

## 已知边界

- 该方案依赖托盘分割质量；zero-shot 空检会明显影响后续。
- 极端反光、遮挡、前立面纹理变化会影响开口候选排序。
- 当前验证为脚本级联调与语法检查，不包含硬件稳定性长时间统计。

## 调参建议（优先级）

1. 开口候选几何约束（细长比、前立面位置范围）。
2. 附近平面生长参数（local/global diff、max pixels）。
3. top plane 邻接区域大小与最小面积约束。
4. 高反差域参数（sigma、canny）。
