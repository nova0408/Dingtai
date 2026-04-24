---
name: utils-transform-conventions
description: 项目中基本数学单位优先使用 `src.utils.datas` 自定义数据类型。涉及 Degree/Point/Vector/Transform/Quaternion/Translation/Axis/SE(3) 变换/矩阵序列化与构造时，优先遵循本技能。
---

# Utils Transform Conventions

## 目标（基于 `src/utils/datas` 全量扫描）

让以下两类内容与项目当前实现保持一致：

1. `src.utils` / `src.utils.datas` 的导入边界
2. 刚体变换协议（`as_SE3` / `from_SE3` / `from_SO3`）与矩阵形状约定

## 适用场景

当任务涉及以下任一内容时使用本技能：

- `Degree`、`Radian`、`Point`、`Vector`、`Box`
- `Transform`、`Quaternion`、`Translation`、`Axis`
- 姿态表达、坐标变换、旋转/平移组合
- 变换矩阵输入输出、协议类型标注
- `Point.transformed(...)` / `Vector.transformed(...)` / `Transform @ ...` 复合

## 真实导入约定（按当前项目）

### 1) 导入边界

- 几何/运动学数据类型：从 `src.utils.datas` 导入。
- `src.utils` 只用于通用工具（`HighPrecisionTimer`、`check_dict`、`check_config`、`SE3_2_xyzr`、`SE3_string`），不用于导出运动学类型。

推荐：

```python
from src.utils.datas import (
    Degree, Radian, Point, Vector, Box,
    Transform, Quaternion, Translation, Axis,
    ANGULAR_TOLERANCES, ARC_TOLERANCES, LINEAR_TOLERANCES,
)
```

不推荐：

```python
from src.utils import Transform  # src.utils 当前并未暴露 Transform
```

### 2) 协议类型导入

优先从 `src.utils.datas` 直接导入协议：

```python
from src.utils.datas import MatrixSerializable, MatrixConstructible, HomogeneousTransformProtocol
```

在当前主干中，若你必须绕过聚合导入（例如排查导入链问题），请使用真实文件路径：

```python
from src.utils.datas.kinematics.transform_protocol import (
    MatrixSerializable, MatrixConstructible, HomogeneousTransformProtocol
)
```

## 变换协议约定（核心）

### 1) 主接口是 `as_SE3` / `from_SE3`

当前实现中协议以 `as_SE3()` 与 `from_SE3(...)` 为标准接口：

- `MatrixSerializable.as_SE3() -> np.ndarray`
- `MatrixConstructible.from_SE3(mat) -> Self`

`as_matrix()` / `from_matrix()` 已在主干代码移除，不再作为约定接口。

### 2) `as_SE3()` 返回形状

- `Transform.as_SE3()`：返回 `4x4` SE(3) 齐次矩阵（平移 + 旋转）
- `Translation.as_SE3()`：返回 `4x4` 平移矩阵
- `Quaternion.as_SE3()`：返回 `4x4` 纯旋转矩阵（平移为 0）

结论：只要走协议接口，默认就是 `4x4`。

### 3) `from_SE3(...)` 约束

- `Transform.from_SE3(...)` 期望 `4x4`
- `Quaternion.from_SE3(...)` 仅接受 `4x4` SE(3)，非 `4x4` 直接抛错
- 若输入是旋转矩阵 `3x3`，应使用 `Quaternion.from_SO3(...)`

## 点/向量变换约定

### 1) Point

`Point.transformed(...)` 接受：

- 支持 `as_SE3()` 的对象
- `np.ndarray`（`3x3` 或 `4x4`）

语义：

- `4x4`：按齐次点 `w=1` 参与平移
- `3x3`：仅线性变换

### 2) Vector

`Vector.transformed(...)` 接受与 `Point` 相同输入类型。

语义：

- `4x4`：按齐次向量 `w=0`，自动忽略平移
- `3x3`：仅线性变换

## 组合与姿态表达约定

- `Transform` 通过 `@` 进行复合：`Transform @ (Transform|Translation|Quaternion)`。
- 四元数内部顺序固定为 `[w, x, y, z]`（`q1,q2,q3,q4`）。
- 欧拉角构造默认遵循 ZYX（与项目现有 `from_zyx` 保持一致）。
- `Transform.to_list()`：
  - `zyx=False` 返回 `[x, y, z, q1, q2, q3, q4]`
  - `zyx=True` 返回 `[x, y, z, rz, ry, rx]`

## 主要类型推荐用法（按模块）

1. `Degree` / `Radian`
- 角度优先用 `Degree`，与 `Quaternion.from_zyx`、`from_euler` 无缝协作。
- `Radian` 适合三角函数直接计算，边界归一化优先 `Radian.normalized()`。

2. `Point` / `Vector`
- 点位变换用 `Point.transformed(...)`，方向变换用 `Vector.transformed(...)`。
- 需要“点到点向量”时优先 `Vector.from_points(start, end)`。

3. `Translation` / `Quaternion` / `Transform`
- 单位姿优先 `Transform.Identity()`；不要手写裸矩阵。
- 当输入为 `3x3` 旋转时，统一 `Quaternion.from_SO3(...)`。
- 变换复合优先 `@`，避免手工拼接矩阵细节。

4. `Axis` / `Box`
- 坐标系与位姿互转：`Axis.to_transform()`、`Axis.from_transform(t)`。
- 包围盒构造优先 `Box.from_list(...)` 或 `Box.from_center(...)`。

## 当前代码状态提示（避免误用）

1. 本仓库统一使用小写路径：`src.utils.datas`、`src.utils.datas.kinematics`。
2. 协议定义文件为 `src/utils/datas/kinematics/transform_protocol.py`。
3. 新代码不要再引入历史大小写路径（如 `Datas`、`Kinematics`、`TransformProtocol.py`）。

## 修改代码时的执行清单
## 文本编辑安全补充

- 使用 PowerShell 做批量替换时，必须同时遵循 `windows-powershell-utf8-safe-edit`。
- 禁止把 `` `r`n `` 当作普通文本写回文件；替换后需做字面量污染检查。

1. 先判断是否应从 `src.utils.datas` 导入（而不是 `src.utils`）。
2. 新代码统一调用 `as_SE3/from_SE3`；若输入为 `3x3` 旋转矩阵，显式使用 `from_SO3`。
3. 涉及矩阵传递时，明确 `3x3` / `4x4` 语义，避免点与向量混用导致平移污染。
4. 不随意改变现有公共方法名；若新增能力，优先沿 `SE3/SO3` 协议扩展。

## 最小示例

```python
from src.utils.datas import Transform, Translation, Quaternion, Point, Vector

base = Transform(Translation(10, 0, 0), Quaternion.Identity())
step = Transform(Translation(0, 5, 0), Quaternion.from_zyx(0, 0, 90))

composed = base @ step
T = composed.as_SE3()  # 推荐主接口，4x4

p = Point(1, 0, 0).transformed(T)   # 受平移影响
v = Vector(1, 0, 0).transformed(T)  # 不受平移影响
```
## 快照要求（补充）

- 任何文件修改前都必须做快照。
- 快照必须存放在仓库根目录 `.archive` 下并保留目录结构。
- 若通过 PowerShell 做批量替换，必须同时遵循 `windows-powershell-utf8-safe-edit`。

