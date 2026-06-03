from __future__ import annotations

from dataclasses import dataclass


# region 数据结构


@dataclass(frozen=True, slots=True)
class JointLimit:
    """单个受控轴的可动范围。

    职责边界：
    - 只描述服务层暴露给二次开发接口的轴范围。
    - 不绑定 gRPC 消息、硬件句柄、运动学模型或 UI 控件。

    设计思想：
    - 使用不可变 dataclass，避免调用方修改服务发布的限制。
    - 使用 `unit` 明确角度、比例或其它线性单位，避免裸 `float` 失去语义。

    生命周期：
    - 可跨线程读取，不持有外部资源。

    继承关系：
    - 不继承业务基类，保持数据契约单一。
    """

    name: str
    "受控轴名称。"

    minimum: float
    "最小允许值，单位由 `unit` 定义。"

    maximum: float
    "最大允许值，单位由 `unit` 定义。"

    unit: str
    "数值单位，例如 `deg`、`ratio`。"


@dataclass(frozen=True, slots=True)
class MotionLimit:
    """单个受控轴的速度或加速度限制。

    职责边界：
    - 只描述速度、加速度或减速度的绝对上限。
    - 不负责轨迹规划、限速插补或硬件急停。

    设计思想：
    - 使用同一结构表达速度和加速度，语义由 `unit` 与调用上下文决定。
    - `maximum` 表达绝对值上限，方向由目标值变化决定。

    生命周期：
    - 可跨线程读取，不持有外部资源。

    继承关系：
    - 不继承业务基类，保持数据契约单一。
    """

    name: str
    "受控轴名称。"

    maximum: float
    "最大绝对值，单位由 `unit` 定义。"

    unit: str
    "数值单位，例如 `deg/s`、`ratio/s`、`deg/s^2`。"


# endregion


# region 基础工具


def coerce_float_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    """将输入数值序列转换为浮点 tuple。

    Parameters
    ----------
    values:
        调用方传入的数值序列，形状语义为 `(N,)`。

    Returns
    -------
    tuple[float, ...]
        转换后的浮点数值序列，形状语义为 `(N,)`。
    """

    return tuple(float(value) for value in values)


def validate_joint_values(joint_values: tuple[float, ...], limits: tuple[JointLimit, ...]) -> None:
    """校验关节值是否处于可动范围内。

    Parameters
    ----------
    joint_values:
        关节值序列，形状语义为 `(N,)`。
    limits:
        可动范围表，长度必须与 `joint_values` 一致。

    Raises
    ------
    ValueError
        当数量不匹配或任一值超出范围时抛出。
    """

    if len(joint_values) != len(limits):
        raise ValueError(f"关节数量不匹配：expected={len(limits)}, actual={len(joint_values)}")
    for value, limit in zip(joint_values, limits, strict=True):
        if value < limit.minimum or value > limit.maximum:
            raise ValueError(f"{limit.name} 超出可动范围：{value} {limit.unit}")


def validate_motion_values(values: tuple[float, ...], limits: tuple[MotionLimit, ...], label: str) -> None:
    """校验速度或加速度是否处于限制内。

    Parameters
    ----------
    values:
        速度或加速度序列，形状语义为 `(N,)`。
    limits:
        对应限制表，长度必须与 `values` 一致。
    label:
        错误信息标签，例如 `速度` 或 `加速度`。

    Raises
    ------
    ValueError
        当数量不匹配或任一绝对值超过限制时抛出。
    """

    if len(values) != len(limits):
        raise ValueError(f"{label}数量不匹配：expected={len(limits)}, actual={len(values)}")
    for value, limit in zip(values, limits, strict=True):
        if abs(value) > limit.maximum:
            raise ValueError(f"{limit.name} 超出{label}限制：{value} {limit.unit}")


# endregion
