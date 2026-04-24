from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self, overload

import numpy as np
from numpy.typing import NDArray

from .geometric_tolerances import LINEAR_TOLERANCES

if TYPE_CHECKING:
    from .kinematics.transform import Transform
    from .kinematics.transform_protocol import MatrixSerializable
    from .vector import Vector


@dataclass(frozen=True, slots=True)
class Point(Sequence[float]):
    """三维空间点。"""

    x: float
    y: float
    z: float

    # region 序列协议与基础方法
    def __iter__(self) -> Iterator[float]:
        """允许解包：`x, y, z = point`。"""
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
        values = (round(self.x, 8), round(self.y, 8), round(self.z, 8))
        return hash(values)

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, 1)"

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other: object) -> bool:
        """按线性容差判断两点近似相等。"""
        if isinstance(other, Point):
            return (
                abs(self.x - other.x) < LINEAR_TOLERANCES
                and abs(self.y - other.y) < LINEAR_TOLERANCES
                and abs(self.z - other.z) < LINEAR_TOLERANCES
            )
        return False

    def is_close(self, other: Point, tol: float = LINEAR_TOLERANCES) -> bool:
        """按指定容差判断两点近似相等。"""
        return abs(self.x - other.x) < tol and abs(self.y - other.y) < tol and abs(self.z - other.z) < tol

    # endregion

    # region 构造与转换
    @classmethod
    def Zero(cls) -> Self:
        """返回零点。"""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def zero(cls) -> Self:
        """`Zero` 的小写别名。"""
        return cls.Zero()

    @classmethod
    def Origin(cls) -> Self:
        """返回原点（`Zero` 的语义别名）。"""
        return cls.Zero()

    @classmethod
    def from_array(cls, arr: Sequence[float] | NDArray[np.float64]) -> Self:
        """从长度为 3 的序列构造点。"""
        if len(arr) != 3:
            raise ValueError("Point.from_array 需要 3 个元素")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_list(cls, li: Sequence[float]) -> Self:
        """从长度为 3 的列表或元组构造点。"""
        if len(li) != 3:
            raise ValueError("Point.from_list 需要 3 个元素")
        return cls(float(li[0]), float(li[1]), float(li[2]))

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

    def transformed(self, trans: NDArray[np.float64] | Transform | MatrixSerializable) -> Self:
        """对点应用 3x3 或 4x4 变换矩阵。"""
        if hasattr(trans, "as_SE3"):
            mat = np.asarray(trans.as_SE3(), dtype=np.float64)  # type: ignore
        elif isinstance(trans, np.ndarray):
            mat = trans.astype(np.float64, copy=False)
        else:
            raise TypeError("trans 必须是支持 as_SE3 的对象或 numpy 矩阵")

        if mat.shape == (4, 4):
            xyz_homo = np.array([self.x, self.y, self.z, 1.0], dtype=np.float64)
            new_xyz = mat @ xyz_homo
            return self.__class__(float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]))

        if mat.shape == (3, 3):
            xyz = np.array([self.x, self.y, self.z], dtype=np.float64)
            new_xyz = mat @ xyz
            return self.__class__(float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]))

        raise ValueError(f"不支持的矩阵形状：{mat.shape}")

    def translation(self, vec: Vector) -> Self:
        """对点应用平移向量。"""
        return self.__class__(self.x + vec.x, self.y + vec.y, self.z + vec.z)

    @staticmethod
    def _coerce_xyz(other: Any) -> tuple[float, float, float] | None:
        """将输入解析为三元坐标，无法解析时返回 `None`。"""
        if isinstance(other, Point):
            return (other.x, other.y, other.z)
        if isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            return (float(other[0]), float(other[1]), float(other[2]))
        if hasattr(other, "x") and hasattr(other, "y") and hasattr(other, "z"):
            return (float(other.x), float(other.y), float(other.z))
        return None

    # endregion

    # region 运算方法
    @overload
    def __add__(self, other: Point) -> Point: ...

    @overload
    def __add__(self, other: Vector) -> Point: ...

    @overload
    def __add__(self, other: float | int) -> Point: ...

    @overload
    def __add__(self, other: Sequence[float] | NDArray[np.float64]) -> Point: ...

    def __add__(self, other: Point | Vector | float | int | Sequence[float] | NDArray[np.float64]) -> Point:
        """点加法：支持点/向量、标量、三元序列。"""
        if isinstance(other, (int, float, np.number)):
            value = float(other)
            return Point(self.x + value, self.y + value, self.z + value)
        xyz = self._coerce_xyz(other)
        if xyz is not None:
            return Point(self.x + xyz[0], self.y + xyz[1], self.z + xyz[2])
        return NotImplemented

    def __radd__(self, other: float | int | Sequence[float] | NDArray[np.float64]) -> Point:
        return self.__add__(other)

    @overload
    def __sub__(self, other: Point) -> Point: ...

    @overload
    def __sub__(self, other: float | int) -> Point: ...

    @overload
    def __sub__(self, other: Sequence[float] | NDArray[np.float64]) -> Point: ...

    def __sub__(self, other: Point | float | int | Sequence[float] | NDArray[np.float64]) -> Point:
        """点减法：支持点、标量、三元序列。"""
        if isinstance(other, (int, float, np.number)):
            value = float(other)
            return Point(self.x - value, self.y - value, self.z - value)
        xyz = self._coerce_xyz(other)
        if xyz is not None:
            return Point(self.x - xyz[0], self.y - xyz[1], self.z - xyz[2])
        return NotImplemented

    def __rsub__(self, other: float | int | np.number | Sequence[float] | NDArray[np.float64]) -> Point:
        if isinstance(other, (int, float, np.number)):
            value = float(other)
            return Point(value - self.x, value - self.y, value - self.z)
        xyz = self._coerce_xyz(other)
        if xyz is not None:
            return Point(xyz[0] - self.x, xyz[1] - self.y, xyz[2] - self.z)
        return NotImplemented

    @overload
    def __mul__(self, other: float | int) -> Point: ...

    @overload
    def __mul__(self, other: Point) -> Point: ...

    @overload
    def __mul__(self, other: Sequence[float] | NDArray[np.float64]) -> Point: ...

    def __mul__(self, other: float | int | Point | Sequence[float] | NDArray[np.float64]) -> Point:
        """点乘法：支持标量乘和按元素乘。"""
        if isinstance(other, (int, float, np.number)):
            value = float(other)
            return Point(self.x * value, self.y * value, self.z * value)
        xyz = self._coerce_xyz(other)
        if xyz is not None:
            return Point(self.x * xyz[0], self.y * xyz[1], self.z * xyz[2])
        return NotImplemented

    # endregion

    # region 几何方法
    def __rmul__(self, other: float | int) -> Point:
        return self.__mul__(other)

    @overload
    def __truediv__(self, other: float | int) -> Point: ...

    @overload
    def __truediv__(self, other: Sequence[float] | NDArray[np.float64]) -> Point: ...

    def __truediv__(self, other: float | int | Sequence[float] | NDArray[np.float64]) -> Point:
        """点除法：支持标量除和按元素除。"""
        if isinstance(other, (int, float, np.number)):
            value = float(other)
            if value == 0:
                raise ZeroDivisionError("除数不能为零")
            return Point(self.x / value, self.y / value, self.z / value)

        if isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            ox, oy, oz = float(other[0]), float(other[1]), float(other[2])
            if ox == 0 or oy == 0 or oz == 0:
                raise ZeroDivisionError("按元素除法时除数不能为零")
            return Point(self.x / ox, self.y / oy, self.z / oz)

        return NotImplemented

    def __neg__(self) -> Point:
        """取相反数。"""
        return self.__class__(-self.x, -self.y, -self.z)

    def __abs__(self) -> Point:
        """坐标取绝对值。"""
        return self.__class__(abs(self.x), abs(self.y), abs(self.z))

    def dot(self, other: Point | Sequence[float] | NDArray[np.float64]) -> float:
        """点积。"""
        xyz = self._coerce_xyz(other)
        if xyz is None:
            raise TypeError("dot 需要 Point 或长度为 3 的序列")
        return self.x * xyz[0] + self.y * xyz[1] + self.z * xyz[2]

    def cross(self, other: Point | Sequence[float] | NDArray[np.float64]) -> Self:
        """叉积。"""
        xyz = self._coerce_xyz(other)
        if xyz is None:
            raise TypeError("cross 需要 Point 或长度为 3 的序列")
        ax, ay, az = xyz
        return self.__class__(self.y * az - self.z * ay, self.z * ax - self.x * az, self.x * ay - self.y * ax)

    def norm(self) -> float:
        """向量范数（长度）。"""
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def distance_to(self, other: Point | Sequence[float] | NDArray[np.float64]) -> float:
        """计算到另一点的欧氏距离。"""
        xyz = self._coerce_xyz(other)
        if xyz is None:
            raise TypeError("distance_to 需要 Point 或长度为 3 的序列")
        return float(np.sqrt((self.x - xyz[0]) ** 2 + (self.y - xyz[1]) ** 2 + (self.z - xyz[2]) ** 2))

    # endregion

    # region 复制
    def copy(self) -> Self:
        """返回当前点的副本。"""
        return replace(self)

    # endregion
