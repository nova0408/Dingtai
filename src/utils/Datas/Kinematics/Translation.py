from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self, final, overload

import numpy as np

from .transform_protocol import MatrixSerializable


@final
@dataclass(frozen=True, slots=True)
class Translation(Sequence[float], MatrixSerializable):
    """三维平移向量。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # region 构造方法

    @classmethod
    def from_list(cls, data: Sequence[float]) -> Self:
        """从长度为 3 的序列构造。"""
        if len(data) != 3:
            raise ValueError(f"Translation 需要 3 个参数，收到 {len(data)} 个")
        return cls(float(data[0]), float(data[1]), float(data[2]))

    @classmethod
    def Zero(cls) -> Self:
        """返回零向量"""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def zero(cls) -> Self:
        """`Zero` 的小写别名。"""
        return cls.Zero()

    @classmethod
    def from_array(cls, arr: Sequence[float] | np.ndarray) -> Self:
        """从长度为 3 的数组或序列构造。"""
        if len(arr) != 3:
            raise ValueError("Translation.from_array 需要 3 个元素")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    # endregion

    # region 序列协议与基础方法
    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return (self.x, self.y, self.z)[index]

    def __len__(self):
        return 3

    def __array__(self, dtype=None) -> np.ndarray:
        """返回 `numpy` 一维数组表示。"""
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def __hash__(self) -> int:
        """哈希（考虑浮点精度）"""
        values = (
            round(self.x, 12),
            round(self.y, 12),
            round(self.z, 12),
        )
        return hash(tuple(values))

    # endregion

    # region 运算方法
    def __add__(self, other: Translation) -> Self:
        """向量加法：self + other"""
        if not isinstance(other, Translation):
            return NotImplemented
        return replace(self, x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Translation) -> Self:
        """向量减法：self - other"""
        if not isinstance(other, Translation):
            return NotImplemented
        return replace(self, x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __neg__(self) -> Self:
        """向量取反：-self"""
        return replace(self, x=-self.x, y=-self.y, z=-self.z)

    def __mul__(self, scalar: float | int) -> Self:
        """标量乘法：self * scalar"""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return replace(self, x=self.x * scalar, y=self.y * scalar, z=self.z * scalar)

    def __rmul__(self, scalar: float | int) -> Self:
        """标量乘法：scalar * self"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float | int) -> Self:
        """标量除法：self / scalar"""
        if scalar == 0:
            raise ZeroDivisionError("平移向量不能除以零")
        return self.__mul__(1.0 / scalar)

    # endregion

    # region 几何方法
    @property
    def magnitude(self) -> float:
        """计算向量模长（欧几里得距离）"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def distance_to(self, other: Translation) -> float:
        """计算两个平移点之间的距离"""
        return (self - other).magnitude

    def dot(self, other: Translation) -> float:
        """点积运算"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def lerp(self, other: Translation, t: float) -> Self:
        """
        线性插值
        t=0 返回 self, t=1 返回 other
        """
        return self + (other - self) * t

    # endregion

    # region 序列化与矩阵协议
    def to_list(self) -> list[float]:
        """返回列表副本"""
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple[float, float, float]:
        """返回元组（常用于库调用）"""
        return (self.x, self.y, self.z)

    def as_SE3(self) -> np.ndarray:
        """返回 4x4 SE(3) 齐次平移矩阵。"""
        return np.array(
            [
                [1.0, 0.0, 0.0, self.x],
                [0.0, 1.0, 0.0, self.y],
                [0.0, 0.0, 1.0, self.z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def as_array(self) -> np.ndarray:
        """返回 3x1 列向量"""
        return np.array([[self.x], [self.y], [self.z]], dtype=np.float64)

    def as_row_array(self) -> np.ndarray:
        """返回一维数组表示。"""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    # endregion

    # region 显示与复制
    def __str__(self) -> str:
        return f"[{self.x:.3f}, {self.y:.3f}, {self.z:.3f}]"

    def __repr__(self) -> str:
        return f"Translation(x={self.x}, y={self.y}, z={self.z})"

    def copy(self) -> Self:
        """返回当前对象的副本。"""
        return replace(self)

    # endregion
