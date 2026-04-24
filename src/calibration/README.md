# 手眼标定子模块说明

本目录提供纯数学 `AX=XB` 手眼标定能力，不依赖硬件 SDK。

实现文件：

- `src/calibration/hand_eye.py`
- `src/calibration/__init__.py`

## 1. 可用接口

- `make_relative_motion_pairs(group_a_poses, group_b_poses, mode="all")`
- `calibrate_hand_eye_ax_xb(a_motions, b_motions, min_required_samples=3) -> Transform`
- `evaluate_hand_eye_solution(a_motions, b_motions, x) -> HandEyeResidualStats`
- `calibrate_hand_eye_from_pose_sequences(group_a_poses, group_b_poses, pair_mode="all") -> HandEyeCalibrationResult`
- `generate_synthetic_motion_pairs(...)`

接口输入支持：

- `src.utils.datas.Transform`（推荐）
- `np.ndarray` 形状 `4x4` 的 SE(3) 矩阵

## 2. 快速调用示例

```python
from src.calibration import calibrate_hand_eye_from_pose_sequences
from src.utils.datas import Transform

# 两组同步位姿（长度必须一致）
group_a: list[Transform] = [...]
group_b: list[Transform] = [...]

result = calibrate_hand_eye_from_pose_sequences(
    group_a_poses=group_a,
    group_b_poses=group_b,
    pair_mode="all",  # "all" 或 "adjacent"
)

x = result.transform
residual = result.residual

print("X =", x.as_string(with_name=True))
print("rotation_rmse_deg =", residual.rotation_rmse_deg)
print("translation_rmse =", residual.translation_rmse)
```

`result.transform` 即求解出的 `X`，满足 `A_k X = X B_k`。

## 3. 接入真实硬件必须提供的数据

接入实机后，核心是“同一时刻的两组位姿序列”：

- `group_a_poses[i]`：第 `i` 次采样的 A 组位姿
- `group_b_poses[i]`：第 `i` 次采样的 B 组位姿

两组必须严格同步、长度一致、单位一致。

### 3.1 推荐数据对象格式（代码内）

每一帧位姿都转为 `Transform`：

- 平移：`[x, y, z]`，单位建议用 `mm`（与项目约定一致）
- 旋转：四元数 `[q1, q2, q3, q4]`，顺序固定为 `[w, x, y, z]`

也可直接传 `4x4` SE(3) 矩阵。

### 3.2 推荐落盘格式（CSV）

建议每次采样保存一行，字段如下：

```text
timestamp,
a_x,a_y,a_z,a_qw,a_qx,a_qy,a_qz,
b_x,b_y,b_z,b_qw,b_qx,b_qy,b_qz
```

说明：

- `timestamp`：同一行内 A/B 必须来自同一时刻（或已做时间对齐）
- `a_*`：A 组绝对位姿
- `b_*`：B 组绝对位姿
- 四元数必须归一化
- 禁止混用单位（例如 A 用 mm，B 用 m）

## 4. 坐标语义要求（必须先定清）

本模块只解数学方程，不自动判断你是哪种安装方式。  
你需要在采集时固定语义，并持续一致。

常见做法：

1. 眼在手上（Eye-in-Hand）

- A 组：机械臂末端相对基座位姿（例如 `T_base_hand`）
- B 组：标定板相对相机位姿（例如 `T_cam_target`）

2. 眼在手外（Eye-to-Hand）

- A/B 含义会变化，但必须能构造到同一 `AX=XB` 语义下
- 若语义不一致，结果会看似“有值但错误”

建议在采集脚本中明确写下注释：`group_a` 和 `group_b` 分别是什么物理量。

## 5. 数据质量最低要求

- 采样数量：建议 `>= 15`，最低不低于 `3`
- 运动激励：末端姿态要有充分旋转变化（不只平移）
- 视野约束：标定板需稳定识别，避免抖动和跳变
- 异常过滤：剔除识别置信度低或机器人状态不稳定的样本

## 6. 结果验收建议

至少检查以下指标：

- `rotation_rmse_deg`
- `translation_rmse`
- `rotation_max_deg`
- `translation_max`

并做一个闭环 spot-check：随机抽样验证 `A_k X` 与 `X B_k` 是否接近。

## 7. 注意事项

- 当前模块不做时间同步，请在上游完成时间对齐。
- 当前模块不自动做离群点鲁棒优化，如需抗异常值可在上游先过滤。
- 未接真实硬件前，只能声明软件级验证通过，不能声明硬件行为已验证。


