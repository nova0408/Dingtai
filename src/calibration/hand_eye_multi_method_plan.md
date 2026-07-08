# 手眼标定多方法扩展方案

## 1. 目标

在**保留当前闭式 `AX=XB` 求解**的前提下，为 `src/calibration/hand_eye.py` 增加一层多方法适配与统一评估框架，实现以下能力：

1. 保留现有自研闭式法作为一个独立候选方法。
2. 增加 OpenCV `calibrateHandEye` 的 4 种方法适配。
3. 对所有候选方法统一做残差评估。
4. 增加交叉验证，避免“单次样本自洽但泛化不稳”的结果。
5. 增加外参稳定性指标，衡量不同子集上的解是否漂移。
6. 输出一个可解释的最优方法选择结果，而不是只返回单个 `Transform`。

---

## 2. 设计原则

1. **不破坏现有闭式法**  
   当前 `calibrate_hand_eye_ax_xb()` 保持语义不变，仍然可直接调用。

2. **多方法层只做编排，不混入算法细节**  
   算法求解、残差计算、交叉验证、统计聚合应分层清晰。

3. **输入语义保持统一**  
   上游继续提供同步的绝对位姿序列或已构造好的相对运动对；多方法层不要隐式猜测坐标语义。

4. **结果必须可解释**  
   每个候选方法都应返回：
   - 求解结果
   - 全量残差
   - 交叉验证统计
   - 稳定性统计
   - 综合评分

5. **最小修改现有调用链**  
   现有只需要 `Transform` 的场景，允许通过默认策略继续拿到最佳结果。

---

## 3. 现有代码基线

当前模块已经具备以下能力：

- `make_relative_motion_pairs(...)`
- `calibrate_hand_eye_ax_xb(...)`
- `evaluate_hand_eye_solution(...)`
- `calibrate_hand_eye_from_pose_sequences(...)`
- `generate_synthetic_motion_pairs(...)`

当前问题在于：

- 只有一个求解器，没有多方法选择层。
- 残差只评估“同一批样本上的闭环误差”，缺少泛化评估。
- 没有对子集稳定性做统计。
- 返回值过窄，难以比较不同方法。

---

## 4. 拟新增的方法集

### 4.1 保留的现有闭式法

保留 `calibrate_hand_eye_ax_xb(a_motions, b_motions, min_required_samples=3) -> Transform` 作为独立候选。

定位：

- 这是当前项目内部自研、可控、无需依赖 OpenCV 的基线方法。
- 适合做对照组、回退方案和数学自测基线。

### 4.2 OpenCV 四方法适配层

新增 OpenCV 适配器，统一映射以下四种方法：

1. `Tsai`
2. `Park`
3. `Horaud`
4. `Daniilidis`

实现形式建议：

- 以 `Literal` 或枚举定义方法名。
- 用单独的适配函数封装 OpenCV 输入转换和输出转换。
- 不在业务函数里散落 `cv2` 调用。

### 4.3 统一入口

新增一个总入口，例如：

```python
calibrate_hand_eye_multi_method(...)
```

职责：

1. 统一构造候选方法列表。
2. 对每个候选方法求解 `X`。
3. 对每个候选方法计算训练残差。
4. 对每个候选方法执行交叉验证。
5. 汇总稳定性指标。
6. 计算综合评分并返回最佳结果。

---

## 5. 建议的数据结构

### 5.1 方法名

```python
HandEyeMethodName = Literal[
    "closed_form",
    "opencv_tsai",
    "opencv_park",
    "opencv_horaud",
    "opencv_daniilidis",
]
```

### 5.2 单方法求解结果

```python
@dataclass(frozen=True, slots=True)
class HandEyeMethodResult:
    method_name: str
    transform: Transform | None
    residual: HandEyeResidualStats | None
    cv_residual: HandEyeCrossValidationStats | None
    stability: HandEyeStabilityStats | None
    score: float | None
    error_message: str | None
```

说明：

- `transform is None` 表示该方法失败。
- `error_message` 必须保留，方便追查方法失败原因。
- 不建议直接抛出异常中断整轮比较；单方法失败应降级为候选失败。

### 5.3 交叉验证统计

建议新增：

```python
@dataclass(frozen=True, slots=True)
class HandEyeCrossValidationStats:
    fold_count: int
    train_rotation_rmse_deg_mean: float
    train_translation_rmse_mean: float
    val_rotation_rmse_deg_mean: float
    val_translation_rmse_mean: float
    val_rotation_rmse_deg_max: float
    val_translation_rmse_max: float
```

说明：

- `train_*` 用于看方法在训练子集上是否拟合正常。
- `val_*` 用于看泛化能力是否稳定。
- 重点看验证集指标，而不是只看训练集。

### 5.4 外参稳定性统计

建议新增：

```python
@dataclass(frozen=True, slots=True)
class HandEyeStabilityStats:
    fold_count: int
    rotation_mean_pairwise_deg: float
    rotation_max_pairwise_deg: float
    translation_mean_pairwise: float
    translation_max_pairwise: float
    rotation_std_deg: float
    translation_std: float
```

说明：

- `rotation_mean_pairwise_deg`：不同折得到的 `X` 两两旋转差均值。
- `translation_mean_pairwise`：不同折得到的平移差均值。
- `rotation_std_deg` / `translation_std`：折间漂移的标准差。

### 5.5 多方法总结果

建议新增：

```python
@dataclass(frozen=True, slots=True)
class HandEyeMultiMethodResult:
    best_method: str | None
    best_result: HandEyeMethodResult | None
    candidates: tuple[HandEyeMethodResult, ...]
```

如果需要更强可视化，可再补充：

- 候选方法排序
- 分项评分
- 失败原因列表

---

## 6. 统一评估设计

### 6.1 训练残差

继续沿用现有 `evaluate_hand_eye_solution()`，用于评估：

- `rotation_rmse_deg`
- `rotation_max_deg`
- `translation_rmse`
- `translation_max`

这部分仍然是基础指标，不能单独作为最终选优依据。

### 6.2 交叉验证

建议采用 **K-fold**，默认 `K=5`，数据不足时自动降级：

- `n < 6`：不做交叉验证，只做全量残差。
- `6 <= n < 10`：使用 `K=3`
- `n >= 10`：使用 `K=5`

折分建议：

- 优先按采样顺序做分层切片，避免所有相邻样本同时落入同一折。
- 若采样已经是较强相关序列，可优先做“间隔抽样”或“分块折分”。

交叉验证流程：

1. 将样本划分为 `K` 折。
2. 每次取 `K-1` 折训练，1 折验证。
3. 在训练集上求 `X_train`。
4. 计算训练残差和验证残差。
5. 汇总所有折的均值和最大值。

### 6.3 外参稳定性

稳定性不是看某一折误差小，而是看**不同折解出来的外参是否一致**。

建议计算：

1. 所有折输出 `X_i`。
2. 将每个 `X_i` 与 `X_j` 做相对差：
   - 旋转差角
   - 平移差范数
3. 统计两两差的均值、最大值、标准差。

稳定性指标用于识别：

- 数据分布偏斜
- 运动激励不足
- 某种方法对局部样本特别敏感

---

## 7. 综合评分策略

### 7.1 评分目标

综合评分应优先区分以下几类情况：

- 训练残差低，但验证残差高
- 验证残差还行，但外参漂移大
- 部分方法数值上能解，但不稳定

### 7.2 建议评分形式

建议采用加权和，先简单可解释，再根据实测调权重：

```text
score =
    w1 * val_rotation_rmse_deg +
    w2 * val_translation_rmse +
    w3 * rotation_mean_pairwise_deg +
    w4 * translation_mean_pairwise +
    w5 * train_rotation_rmse_deg +
    w6 * train_translation_rmse
```

建议默认权重优先级：

- 验证旋转误差权重最高
- 验证平移误差次之
- 稳定性指标第三
- 训练误差仅作辅助

如果某方法失败：

- `score = +inf`
- 保留失败原因

### 7.3 选优规则

优先级建议：

1. 验证集指标最优
2. 稳定性更好
3. 训练残差更低
4. 再比较总体评分

不要只按训练残差最小选最优。

---

## 8. 接口改造建议

### 8.1 保留旧接口

以下接口保持兼容：

- `calibrate_hand_eye_ax_xb(...)`
- `evaluate_hand_eye_solution(...)`
- `calibrate_hand_eye_from_pose_sequences(...)`

其中 `calibrate_hand_eye_from_pose_sequences(...)` 可新增一个可选参数，例如：

- `method: Literal["closed_form", "multi_method"] = "closed_form"`

或新增一个更明确的入口，避免旧语义变化。

### 8.2 新增推荐入口

建议新增：

```python
calibrate_hand_eye_multi_method(
    group_a_poses: list[PoseLike],
    group_b_poses: list[PoseLike],
    pair_mode: PairMode = "all",
    methods: Sequence[HandEyeMethodName] | None = None,
    cv_folds: int | None = None,
) -> HandEyeMultiMethodResult
```

默认行为：

- `methods is None` 时，自动启用全部候选方法。
- `cv_folds is None` 时，根据样本量自动选择折数。

### 8.3 与现有测试脚本的关系

`debug/test_handeye_math.py` 和实验脚本应切换到多方法入口，输出：

- 每种方法的结果
- 排名
- 稳定性
- 交叉验证统计

但不建议一开始就改所有实验页；优先保证 `src/calibration` 层稳定。

---

## 9. OpenCV 适配注意事项

### 9.1 输入格式

OpenCV 常用接口需要绝对位姿列表：

- `R_gripper2base`
- `t_gripper2base`
- `R_target2cam`
- `t_target2cam`

因此适配层要做明确的输入转换，不要在上层隐式拼接。

### 9.2 坐标语义必须固定

在实现前必须明确：

- `group_a` 到底是 `gripper2base` 还是 `base2gripper`
- `group_b` 到底是 `target2cam` 还是 `cam2target`

这个语义如果不统一，四方法的结果看起来可能“都能算”，但实际是错的。

### 9.3 失败兜底

OpenCV 方法可能因数据病态、维度错误或数值问题失败。

处理方式：

- 单方法失败只记录，不中断全局结果。
- 若全部失败，返回失败原因集合。

---

## 10. 测试与验证计划

### 10.1 单元测试

建议新增测试覆盖：

1. 现有闭式法可继续在合成数据上跑通。
2. 多方法入口能返回 5 个候选中的若干个结果。
3. 单方法失败不会影响其他方法。
4. 交叉验证在样本数不足时自动降级。
5. 稳定性统计在输入完全一致时应接近 0 漂移。

### 10.2 合成数据验证

用 `generate_synthetic_motion_pairs(...)` 扩展出：

- 低噪声
- 中噪声
- 高噪声
- 偏少样本
- 运动分布偏单一

观察：

- 哪种方法在什么条件下更稳
- 多方法选优是否能避开病态样本

### 10.3 真实数据验证

至少检查：

- 同一批数据重复运行结果是否稳定
- 不同折之间的外参漂移是否可接受
- 选出的最佳方法是否在视觉/机械臂实际误差上更优

未接真实硬件前，只能做软件级验证，不能宣称硬件精度已提升。

---

## 11. 建议实施顺序

### Phase 1：接口与数据结构

1. 新增方法名类型。
2. 新增多方法结果、交叉验证结果、稳定性结果的数据结构。
3. 保留现有闭式法不动。

### Phase 2：OpenCV 适配层

1. 添加 OpenCV 统一适配函数。
2. 接入 4 种方法。
3. 处理失败兜底。

### Phase 3：统一评估与选优

1. 补充残差统一评估。
2. 增加 K-fold 交叉验证。
3. 增加外参稳定性统计。
4. 增加综合评分与排序。

### Phase 4：外部入口切换

1. 更新 `__init__.py` 导出。
2. 更新实验脚本输出。
3. 更新 README 文档。

### Phase 5：验证

1. 合成数据验证。
2. 退化场景验证。
3. 低样本量验证。
4. 结果稳定性对比。

---

## 12. 风险点

1. **坐标语义反了**  
   这是最大风险。四方法都会“算出结果”，但结果可能全错。

2. **样本数量太少**  
   交叉验证会变得不稳定，最小样本数必须提前约束。

3. **姿态分布太集中**  
   某些方法会数值退化，稳定性指标会暴露这个问题。

4. **只看训练误差**  
   这是最容易误判的地方，必须把验证集和稳定性放进选优规则。

5. **OpenCV 依赖差异**  
   不同环境下 OpenCV 版本或编译选项可能影响 `calibrateHandEye` 可用性。

---

## 13. 本次改动的完成定义

当以下条件满足时，视为方案落地完成：

1. 现有闭式法仍可单独调用。
2. 多方法入口可返回全部候选结果。
3. 四种 OpenCV 方法均已接入。
4. 每个候选方法都有统一残差输出。
5. 支持交叉验证。
6. 支持外参稳定性统计。
7. 支持综合评分和自动选优。
8. README 与实验脚本同步更新。

