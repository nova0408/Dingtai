# Grasp Pose 实时估计方案（Gemini 305）

## 1. 文档目标

本文档描述 `test_orbbec_realtime_estimate_grasp_pose_pipeline.py` 的**当前实现**，重点说明：

- 实时链路如何做到“预览连续 + 计算快速响应”。
- 托盘分割、开口检测、平面/法线估计之间的先后关系与并行关系。
- 已落地的提速策略、失败分类机制、以及耗时分析方法。

适用脚本：

- `test/pointcloud/grasp_pose/test_orbbec_realtime_estimate_grasp_pose_pipeline.py`
- `test/pointcloud/grasp_pose/analyze_grasp_pose_timing.py`

---

## 2. 总体设计（先快后稳）

当前方案采用“**快速主链 + 异步精化链**”：

1. 主链优先给出抓取结果（低延迟响应）。
2. 掩码与法线在后台并行补全（精度增益）。
3. 预览以实时帧为底图，计算结果按可用程度叠加，不再全-or-无。

这不是离线最优解，而是面向实时工程场景的权衡：  
**先可用，再逐步变准；先连续，再抑制跳变。**

---

## 3. 数据流与线程模型

### 3.1 线程职责

- 主线程：
  - 相机采集
  - Open3D 3D 刷新
  - cv2 2D 预览（实时底图 + 计算叠加）
- 后台 worker 线程：
  - 计算帧处理
  - 步骤耗时记录（CSV）
- 掩码子线程池（worker 内）：
  - 并行执行掩码链路（Opening/Top mask）

### 3.2 队列策略

- `job_queue(maxsize=1)`：只保留最新待计算帧，避免积压导致“算出来已过时”。
- `result_queue(maxsize=2)`：结果只保留最近项，主线程优先展示最新结果。

---

## 4. 计算帧执行顺序（含并行）

每个计算帧的主要流程：

1. `segment_tray`（托盘掩码）
2. `detect_opening`（开口检测）
3. 并行分叉：
   - 快速主链：
     - `filter_local_points_fast`（先不依赖 no_hole_mask）
     - `estimate_opening_plane`
     - `compute_grasp_fast`（无 top normal）
   - 掩码精化链（线程池）：
     - `build_near_plane_mask`
     - `build_top_plane_mask`
     - `enforce_disjoint_masks`
4. 掩码链在预算时间内（`DEFAULT_MASK_SYNC_BUDGET_MS`）完成时：
   - `fit_top_quad`
   - `estimate_top_plane_normal`
   - `compute_grasp_refined`（有 top normal，姿态更稳）
5. 绘制输出：
   - 左图：叠加结果
   - 右图：高反差诊断图

说明：

- 若掩码链超时，不阻塞主链响应，直接使用 `compute_grasp_fast`。
- 若后续步骤失败，前面成功结果仍保留并绘制。

---

## 5. 托盘分割的激进加速策略

`segment_tray` 采用了“前台快、后台准”的混合策略：

### 5.1 前台快速可用（当帧即时返回）

- 快速 fallback：亮度阈值 + 连通域选择（偏向画面下中部大连通域）
- 成本低，保证每帧都有托盘近似区域

### 5.2 后台 zero-shot 异步刷新缓存

- 不在当前帧阻塞等待零训练推理
- 满足刷新条件时异步启动 refine 任务
- refine 成功后更新 `cached_mask`

### 5.3 缓存掩码运动先验补偿（降跳变）

- 相邻帧灰度做快速平移估计（`phaseCorrelate`）
- 对 `cached_mask` 执行平移 warp，再用于当前帧
- 配合平滑与限幅，减少“隔几帧突然跳一下”

---

## 6. 预览语义（实时帧优先）

2D 预览已调整为：

- 底图始终来自**当前实时帧**（彩色帧或实时点云栅格化）
- 计算结果作为叠加层按可用内容绘制

这样避免了“显示的是历史计算帧整图”的时序错位问题。

---

## 7. 失败分类与可观测性

### 7.1 失败类别

失败按步骤分类：

- `tray`：托盘分割失败（`segment_tray`）
- `opening`：开口检测失败（`detect_opening`）
- `other`：其余步骤失败

### 7.2 日志输出

失败日志包含：

- `failed_step`
- `error`
- `completed_steps`（按先后顺序，用 `>` 串联）

用于快速判断“算到了哪一步失败”。

### 7.3 单帧 OpenCV 异常保护

- `cv2.error` 不再导致整条流退出
- 单帧失败会被降级显示并继续处理后续帧

---

## 8. 耗时统计与分析

### 8.1 原始计时 CSV

运行主脚本会输出：

- `logs/grasp_pose_pipeline_timing.csv`

记录字段包括：

- `run_id, frame_idx, step_name, start_ts, end_ts, elapsed_ms, status, error, frame_total_elapsed_ms`

### 8.2 分析脚本

```bash
python test/pointcloud/grasp_pose/analyze_grasp_pose_timing.py --show
```

分析脚本支持：

- 自动排除启动阶段异常帧
- 按计算帧分阶段（前/中/后）对比步骤耗时
- 输出图表与 summary CSV

---

## 9. 关键参数（建议从这里调）

与响应速度相关：

- `DEFAULT_COMPUTE_MIN_INTERVAL_S`
- `DEFAULT_MASK_SYNC_BUDGET_MS`
- `DEFAULT_TRAY_DETECT_EVERY_N`
- `DEFAULT_TRAY_DETECT_MAX_SIDE`
- `DEFAULT_TRAY_USE_SAM`

与托盘缓存稳定性相关：

- `DEFAULT_TRAY_MOTION_DOWNSAMPLE`
- `DEFAULT_TRAY_MOTION_SMOOTH_ALPHA`
- `DEFAULT_TRAY_MOTION_MAX_SHIFT_PX`

与掩码区域生长相关：

- `DEFAULT_NEAR_GROW_LOCAL_DIFF`
- `DEFAULT_NEAR_GROW_GLOBAL_DIFF`
- `DEFAULT_NEAR_GROW_MAX_PIXELS`

---

## 10. 验证口径说明

已覆盖：

- 代码级语法验证（`py_compile`）
- 计时链路与分析脚本闭环

未覆盖：

- 本文档不声称已完成全部硬件场景实机验证
- 相机运动剧烈、极端反光遮挡下仍需现场调参

---

## 11. 当前结论（工程视角）

相较早期串行方案，当前实现在三个方面更均衡：

1. 延迟：主链不再被掩码和 zero-shot 强阻塞。
2. 连续：缓存掩码引入帧间运动先验，降低跳变。
3. 可诊断：失败步骤与耗时链路可直接定位瓶颈。

后续若继续迭代，建议优先做“质量评分驱动的自适应策略”（高置信时更激进提速，低置信时自动回到稳态配置）。
