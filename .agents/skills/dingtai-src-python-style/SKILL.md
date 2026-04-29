---
name: dingtai-src-python-style
description: Dingtai 项目 `src/` Python 代码编写与重构规范。用于创建、修改、整理或评审 `src/**/*.py` 时，强制执行中文详细注释、NumPy 风格中文 docstring、dataclass 字段说明、class 设计说明、region 分区、禁止魔术式动态分发、单一职责和最小验证要求。
---

# Dingtai Src Python Style

## 目标

在 Dingtai 项目 `src/` 下编写或修改 Python 代码时，优先保证可读、可维护、可调试。所有新增或明显修改的 `src/**/*.py` 代码必须让后续维护者只读当前文件即可理解数据结构、数组形状、线程/硬件边界和算法步骤。

## 使用范围

1. 适用于 `src/` 下所有 Python 文件。
2. 修改 `test/` 或 `experiments/` 时，若代码会沉淀到 `src/`，也按本规范编写核心逻辑。
3. 与更具体技能冲突时，先遵循更具体技能，再补齐本技能的注释、结构和验证要求。

## 文件职责

1. 每个代码页尽可能保持单一职责：一个文件只承载一个清晰主题，例如算法、IO、可视化、硬件适配、数据结构或工具函数。
2. 不在算法文件中混入 UI、硬件采集、模型下载、CLI 或临时实验逻辑。
3. 若函数数量增长导致文件职责不清，先提出或实施小范围拆分：数据结构、核心算法、设备适配、预览脚本分开。
4. 公共接口放在 `src/` 合适模块内；测试脚本只负责组装、预览、日志和冒烟验证。

## 分区要求

每个 `src` 代码页尽量使用 `# region` / `# endregion` 分区。常见顺序：

```python
# region 数据结构
# endregion

# region 配置
# endregion

# region 主入口
# endregion

# region 核心算法
# endregion

# region IO 与适配
# endregion

# region 基础工具
# endregion
```

分区名称必须是中文或中英混合且语义明确，不使用含糊名称如 `misc`、`helpers`、`utils2`。

## 数据结构注释

所有新增或明显修改的 `dataclass`、配置类、结果类必须添加中文类 docstring 和字段说明。字段说明使用紧跟字段后的字符串字面量，格式如下：

```python
@dataclass(frozen=True)
class TrayDetection:
    """料盘检测结果。

    该结构是 2D 识别结果与 3D 点云排除流程之间的数据契约。类本身不持有检测器、
    图像对象或点云对象，避免把重资源生命周期绑定到结果数据上。

    设计思想：
    - 使用不可变 dataclass，确保跨线程传递时不会被预览线程或计算线程意外改写。
    - 只保存后续排除点云必需的信息：标签、置信度、轮廓、掩码和排除点数。
    - NumPy 数组字段保持调用方提供的坐标系语义，不在结构体内部做隐式转换。

    继承关系：
    - 不继承业务基类，避免引入隐式生命周期或动态分发。
    - 仅依赖 dataclass 生成初始化与只读字段约束。
    """

    label_text: str
    "检测标签文本。"
    confidence_2d: float
    "2D 检测置信度，范围由底层检测模型定义。"
    contour: np.ndarray
    "轮廓点数组，常见形状为 (K, 1, 2) 或 (K, 2)，dtype 通常为 int32，坐标单位为像素。"
    mask: np.ndarray
    "检测掩码，形状为 (H, W)，非零像素表示料盘区域，dtype 通常为 uint8。"
    excluded_points: int = 0
    "由该检测结果排除的点云数量，单位 点。"
```

字段说明必须包含以下信息中适用的部分：

1. 数据含义。
2. 单位，例如 mm、像素、deg、秒、点。
3. NumPy 数组形状，例如 `(N, 3)`、`(H, W)`、`(4, 4)`。
4. dtype 或数值范围，例如 `float64`、`int32`、`0-1 RGB`。
5. 坐标系或索引语义，例如相机坐标系、图像坐标、点云原始索引。

## Class 注释

所有新增或明显修改的 class 必须添加详细中文 docstring，不只写用途。至少覆盖：

1. 职责边界：这个类负责什么，不负责什么。
2. 设计思想：为什么用类、为什么用不可变结构、为什么持有状态或不持有状态。
3. 生命周期：是否可跨线程复用，是否持有硬件、模型、文件句柄或 GPU 资源。
4. 继承关系：说明是否继承业务基类；若没有继承，也明确“不继承业务基类”的原因。
5. 线程/异步语义：若涉及队列、线程、协程或硬件回调，说明谁写、谁读、何时释放。

避免只写：

```python
"""料盘识别器。"""
```

应写成能指导维护者改代码的完整说明。

## 方法注释

`src` 中每个新增或明显修改的方法/函数都必须使用中文 NumPy 风格 docstring。推荐结构：

```python
def project_points_to_image(
    xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将相机坐标系点云投影到图像平面。

    Parameters
    ----------
    xyz:
        点云坐标数组，形状为 `(N, 3)`，单位 mm。第 0/1/2 列分别为 X/Y/Z。
    fx, fy:
        相机焦距，单位 像素。
    cx, cy:
        主点坐标，单位 像素。
    w, h:
        输出图像宽高，单位 像素。

    Returns
    -------
    uv:
        像素坐标数组，形状为 `(N, 2)`，dtype 为 `int32`。第 0 列为 U，第 1 列为 V；
        无效点填 `-1`。
    valid_proj:
        有效投影掩码，形状为 `(N,)`，dtype 为 `bool`。`True` 表示该点可用于图像索引。

    Notes
    -----
    该函数只做针孔模型投影，不做畸变校正，也不改变输入点云顺序。
    """
```

最少包含：

1. 一句话说明函数做什么。
2. `Parameters`：每个参数的含义、形状、单位和 dtype。
3. `Returns`：返回值含义、形状、单位和 dtype。
4. `Raises`：存在显式异常时必须写。
5. `Notes`：涉及坐标系、线程、硬件、GPU、数组广播、索引语义、性能权衡时必须写。

## NumPy 中文注释

除了 docstring，复杂 NumPy 代码旁必须写中文行内注释。注释重点不是解释语法，而是解释数组语义：

1. 形状：`xyz: (N, 3)`、`labels: (N,)`、`rotation: (3, 3)`。
2. dtype：`float64`、`int32`、`uint8`、`bool`。
3. 掩码：`True` 和 `False` 分别代表什么。
4. 广播：例如 `(K, 3) - (1, 3)`。
5. 高级索引：例如 `mask[v, u]`、`xyz[labels == plane_id]`。
6. 矩阵乘法：例如 `(N, 3) @ (3,) -> (N,)`。
7. 坐标系：相机坐标系、图像坐标系、测试坐标系、单位 mm/像素/deg。
8. 性能：是否复制大数组，是否返回视图，是否要求连续内存。

示例：

```python
# labels: (N,) int32；-1 未分配，0/1/2 表示三平面，-2 表示料盘排除点。
labels = np.full((xyz.shape[0],), -1, dtype=np.int32)

# xyz @ normal: (N, 3) @ (3,) -> (N,)，得到每个点到该平面的有符号距离。
dist = xyz @ model[:3] + float(model[3])

# center.reshape(1, 3) 与 pts: (K, 3) 广播相减，输出 centered: (K, 3)。
centered = pts - center.reshape(1, 3)
```

## 禁止事项

1. 禁止新增不必要的魔术方法或魔术式动态分发，例如无明确必要的 `__getattr__`、`__setattr__`、`__getattribute__`、动态字符串调用。
2. 不使用 `getattr(obj, name)` 作为常规分发机制；优先显式函数、显式映射或数据结构。
3. 不用无结构 `dict` 长距离透传参数；参数过多时提取 dataclass 或配置类，并补齐字段说明。
4. 不在 `src` 公共模块中依赖 `test` 模块作为长期实现。若临时复用测试实现，必须在注释中说明迁移计划和边界。
5. 不在硬件、GUI、算法、模型加载之间做隐式副作用耦合。

## 修改流程

1. 先定位最小归属模块，确认新逻辑应放在 `src`、`test` 还是 `experiments`。
2. 修改前按项目规则在 `.archive` 下做快照。
3. 先设计数据结构和职责边界，再写函数实现。
4. 写代码时同步补齐 class docstring、dataclass 字段说明、函数 NumPy 风格 docstring、关键 NumPy 行内中文注释和 region 分区。
5. 修改后做最小验证：优先 `py_compile`、导入检查、无硬件冒烟；涉及硬件时明确“未连接硬件验证”。

## 输出要求

最终回复中说明：

1. 修改了哪些 `src` 文件。
2. 是否补齐了数据结构字段说明、class 设计说明、函数 NumPy 风格 docstring 和关键数组注释。
3. 是否保持单一职责，若仍有跨层依赖，明确残留原因。
4. 已验证内容和未验证内容。
