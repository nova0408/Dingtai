from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np


@runtime_checkable
class MatrixSerializable(Protocol):
    """可序列化为 SE(3) 齐次矩阵的协议。"""

    def as_SE3(self) -> np.ndarray:
        """返回 4x4 SE(3) 齐次矩阵。"""
        ...


@runtime_checkable
class MatrixConstructible(Protocol):
    """可由 SE(3) 齐次矩阵构造对象的协议。"""

    @classmethod
    def from_SE3(cls, mat: np.ndarray) -> Self:
        """由 4x4 SE(3) 齐次矩阵构造实例。"""
        ...


@runtime_checkable
class HomogeneousTransformProtocol(MatrixSerializable, MatrixConstructible, Protocol):
    """齐次变换协议：同时支持 `as_SE3` 与 `from_SE3`。"""


@runtime_checkable
class ListSerializable(Protocol):
    """可序列化为浮点列表的协议。"""

    def to_list(self) -> list[float]:
        """返回对象的列表表示，常用于配置、日志或轻量序列化。"""
        ...


@runtime_checkable
class ArraySerializable(Protocol):
    """可序列化为 numpy 数组的协议。"""

    def as_array(self) -> np.ndarray:
        """返回对象的 numpy 数组表示，便于数值计算与矩阵运算。"""
        ...
