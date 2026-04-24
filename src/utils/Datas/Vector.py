from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, final, overload

import numpy as np
from numpy.typing import NDArray

from .geometric_tolerances import LINEAR_TOLERANCES
from .kinematics.transform_protocol import MatrixSerializable

if TYPE_CHECKING:
    from .kinematics.quaternion import Quaternion
    from .kinematics.transform import Transform
    from .point import Point


@final
@dataclass(frozen=True, slots=True)
class Vector(Sequence[float]):
    """三维向量对象（不可变、不可继承）。"""

    x: float
    y: float
    z: float

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

    def __len__(self) -> int:
        return 3

    def __array__(self, dtype: Any = None) -> NDArray[np.float64]:
        """返回 `numpy` 一维数组表示。"""
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def __hash__(self) -> int:
        """哈希值，按固定精度对浮点数做归一化。"""
        values = (
            round(self.x, 12),
            round(self.y, 12),
            round(self.z, 12),
        )
        return hash(values)

    # endregion

    # region 运算方法
    def __add__(self, other: Vector) -> Vector:
        """向量加法。"""
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented

    def __sub__(self, other: Vector) -> Vector:
        """向量减法。"""
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    def __neg__(self) -> Vector:
        """向量取反。"""
        return Vector(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float | int) -> Vector:
        """标量乘法。"""
        if not isinstance(scalar, (float, int, np.number)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float | int) -> Vector:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float | int) -> Vector:
        """标量除法。"""
        if not isinstance(scalar, (float, int, np.number)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("向量不能除以零")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vector):
            return (
                abs(self.x - other.x) < LINEAR_TOLERANCES
                and abs(self.y - other.y) < LINEAR_TOLERANCES
                and abs(self.z - other.z) < LINEAR_TOLERANCES
            )
        return False

    # endregion

    # region 构造与序列化
    def to_list(self) -> list[float]:
        """返回列表表示。"""
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple[float, float, float]:
        """返回元组表示。"""
        return (self.x, self.y, self.z)

    def as_array(self) -> NDArray[np.float64]:
        """返回 `numpy` 一维数组表示。"""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def as_column_array(self) -> NDArray[np.float64]:
        """返回 3x1 列向量表示。"""
        return np.array([[self.x], [self.y], [self.z]], dtype=np.float64)

    @classmethod
    def zero(cls) -> Vector:
        """返回零向量。"""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def Zero(cls) -> Vector:
        """`zero` 的大写别名。"""
        return cls.zero()

    @classmethod
    def from_array(cls, arr: Sequence[float] | NDArray) -> Vector:
        if len(arr) != 3:
            raise ValueError("Vector.from_array 需要 3 个元素")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_list(cls, li: Sequence[float]) -> Vector:
        """从长度为 3 的列表或元组构造向量。"""
        if len(li) != 3:
            raise ValueError("Vector.from_list 需要 3 个元素")
        return cls(float(li[0]), float(li[1]), float(li[2]))

    @classmethod
    def from_points(cls, start: Point, end: Point) -> Vector:
        """计算从点 A 指向点 B 的向量。"""
        return cls(end.x - start.x, end.y - start.y, end.z - start.z)

    @classmethod
    def XAxis(cls) -> Vector:
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def YAxis(cls) -> Vector:
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def ZAxis(cls) -> Vector:
        return cls(0.0, 0.0, 1.0)

    # endregion

    # region 几何方法
    @property
    def length(self) -> float:
        """向量长度（L2 范数）。"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Vector:
        """返回单位向量；长度过小时返回零向量。"""
        l = self.length
        if l < 1e-12:
            return Vector.zero()
        return self / l

    def dot(self, other: Vector) -> float:
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector) -> Vector:
        """叉积（结果垂直于两个输入向量）。"""
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def angle_to(self, other: Vector) -> float:
        """计算与另一个向量的夹角（弧度）。"""
        m1, m2 = self.length, other.length
        if m1 < 1e-12 or m2 < 1e-12:
            return 0.0
        cos_theta = self.dot(other) / (m1 * m2)
        return math.acos(max(-1.0, min(1.0, cos_theta)))

    def lerp(self, target: Vector, t: float) -> Vector:
        """向量线性插值"""
        return self + (target - self) * t

    def negated(self) -> Vector:
        """返回向量的反方向。"""
        return Vector(-self.x, -self.y, -self.z)

    # endregion

    # region 变换方法
    def transformed(self, transformation: NDArray[np.float64] | Transform | Quaternion | MatrixSerializable) -> Vector:
        """对向量应用变换，自动忽略 4x4 变换中的平移分量。"""
        if hasattr(transformation, "as_SE3"):
            mat = np.asarray(transformation.as_SE3(), dtype=np.float64)
        elif isinstance(transformation, np.ndarray):
            mat = transformation.astype(np.float64, copy=False)
        else:
            raise TypeError(f"不支持的变换类型：{type(transformation)}")

        if mat.shape == (4, 4):
            v_homo = np.array([self.x, self.y, self.z, 0.0], dtype=np.float64)
            res = mat @ v_homo
            return Vector(float(res[0]), float(res[1]), float(res[2]))

        if mat.shape == (3, 3):
            v_raw = np.array([self.x, self.y, self.z], dtype=np.float64)
            res = mat @ v_raw
            return Vector(float(res[0]), float(res[1]), float(res[2]))

        raise ValueError(f"不受支持的矩阵形状：{mat.shape}")

    # endregion

    # region 显示与复制
    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, 0)"

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"

    def copy(self) -> Vector:
        return replace(self)

    # endregion
