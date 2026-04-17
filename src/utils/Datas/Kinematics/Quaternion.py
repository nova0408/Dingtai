from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Self, final, overload

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from ..Degree import Degree
from ..Vector import Vector
from .TransformProtocol import MatrixConstructible, MatrixSerializable


class EulerSequence(Enum):
    XYZ = "XYZ"
    XZY = "XZY"
    YXZ = "YXZ"
    YZX = "YZX"
    ZXY = "ZXY"
    ZYX = "ZYX"
    XYX = "XYX"
    XZX = "XZX"
    YXY = "YXY"
    YZY = "YZY"
    ZXZ = "ZXZ"
    ZYZ = "ZYZ"


@final
@dataclass(frozen=True, slots=True)
class Quaternion(Sequence[float], MatrixSerializable, MatrixConstructible):
    """四元数表示，内部顺序遵循 `[w, x, y, z]`。"""

    q1: float  # w (实部)
    q2: float  # x (虚部)
    q3: float  # y (虚部)
    q4: float  # z (虚部)

    # region 序列协议与基础方法
    def __iter__(self) -> Iterator[float]:
        """允许解包：`q1, q2, q3, q4 = quaternion`。"""
        yield self.q1
        yield self.q2
        yield self.q3
        yield self.q4

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return (self.q1, self.q2, self.q3, self.q4)[index]

    def __len__(self) -> int:
        return 4

    def __array__(self, dtype=None) -> NDArray:
        """返回 `numpy` 一维数组表示。"""
        return np.array([self.q1, self.q2, self.q3, self.q4], dtype=dtype)

    def __hash__(self) -> int:
        """哈希（考虑浮点精度）"""
        values = (
            round(self.q1, 8),
            round(self.q2, 8),
            round(self.q3, 8),
            round(self.q4, 8),
        )
        return hash(tuple(values))

    def to_list(self) -> list[float]:
        """返回 ABB 格式列表 [w, x, y, z]"""
        return [self.q1, self.q2, self.q3, self.q4]

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.q1, self.q2, self.q3, self.q4)

    def as_array(self) -> NDArray:
        """返回 4x1 列向量"""
        return np.array([[self.q1], [self.q2], [self.q3], [self.q4]], dtype=np.float64)

    def as_row_array(self) -> NDArray:
        """返回一维数组表示。"""
        return np.array([self.q1, self.q2, self.q3, self.q4], dtype=np.float64)

    # endregion

    # region 构造方法

    @classmethod
    def Identity(cls) -> Self:
        """返回单位四元数。"""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_list(cls, li: Sequence[float]) -> Self:
        """从长度为 4 的列表或元组构造。"""
        if len(li) != 4:
            raise ValueError("Quaternion.from_list 需要 4 个元素")
        return cls(float(li[0]), float(li[1]), float(li[2]), float(li[3]))

    @classmethod
    def from_zyx(cls, z: Degree | float | int, y: Degree | float | int, x: Degree | float | int) -> Self:
        """
        从 ZYX 欧拉角构造 (内旋)。
        与 ABB RAPID 程序中 EulerZYX 的旋转定义保持严格一致。
        """
        angles = [Degree(z).value, Degree(y).value, Degree(x).value]
        # Scipy as_quat(scalar_first=True) 返回 [w, x, y, z]
        q = R.from_euler("ZYX", angles, degrees=True).as_quat(scalar_first=True)
        return cls(*q)

    @classmethod
    def from_euler(cls, sequence: EulerSequence | str, angles: Sequence[Degree | float | int]) -> Self:
        """从任意欧拉角序列构造"""
        seq_val = sequence.value if isinstance(sequence, EulerSequence) else sequence
        if len(angles) != 3:
            raise ValueError("需要 3 个欧拉角角度")

        raw_angles = [Degree(a).value for a in angles]
        q = R.from_euler(seq_val, raw_angles, degrees=True).as_quat(scalar_first=True)
        return cls(*q)

    @classmethod
    def from_SE3(cls, mat: NDArray) -> Self:
        """Construct from a 4x4 SE(3) homogeneous matrix."""
        if mat.shape != (4, 4):
            raise ValueError(f"Quaternion.from_SE3 expects a 4x4 SE(3) matrix, got {mat.shape}")
        return cls.from_SO3(mat[:3, :3])

    @classmethod
    def from_SO3(cls, mat: NDArray) -> Self:
        """Construct from a 3x3 SO(3) rotation matrix."""
        if mat.shape != (3, 3):
            raise ValueError(f"Quaternion.from_SO3 expects a 3x3 SO(3) matrix, got {mat.shape}")
        return cls(*R.from_matrix(mat).as_quat(scalar_first=True))

    @classmethod
    def from_axis_angle(cls, axis: Vector, angle: Degree | float | int) -> Self:
        """从旋转轴和角度构造 (右手定则)"""
        angle_rad = Degree(angle).as_radians()

        # 归一化轴向量
        axis_vec = np.array([axis.x, axis.y, axis.z])
        norm = np.linalg.norm(axis_vec)
        if norm < 1e-12:
            raise ValueError("旋转轴向量长度不能为零")

        axis_unit = axis_vec / norm
        half_angle = angle_rad / 2.0
        sin_half = math.sin(half_angle)

        return cls(
            q1=math.cos(half_angle), q2=axis_unit[0] * sin_half, q3=axis_unit[1] * sin_half, q4=axis_unit[2] * sin_half
        )

    @classmethod
    def from_vector2vector(cls, from_vec: Vector, to_vec: Vector) -> Self:
        """构造将 from_vec 旋转到 to_vec 的最短路径四元数"""
        v1 = np.array([from_vec.x, from_vec.y, from_vec.z])
        v2 = np.array([to_vec.x, to_vec.y, to_vec.z])

        # 归一化处理
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            raise ValueError("输入向量长度不能为零")

        v1, v2 = v1 / n1, v2 / n2
        dot = np.dot(v1, v2)

        # 情况 1: 向量相同
        if dot > 0.999999:
            return cls(1.0, 0.0, 0.0, 0.0)

        # 情况 2: 向量完全相反 (180 度)
        if dot < -0.999999:
            # 寻找一个垂直于 v1 的轴
            axis = np.cross(v1, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v1, [0, 1, 0])
            return cls.from_axis_angle(Vector(*axis), 180.0)

        # 情况 3: 一般情况
        axis = np.cross(v1, v2)
        w = 1.0 + dot
        return cls(w, axis[0], axis[1], axis[2]).normalized()

    # endregion

    # region 运算与转换

    def normalized(self) -> Self:
        """返回单位化的四元数 (保证旋转有效性)"""
        norm = math.sqrt(self.q1**2 + self.q2**2 + self.q3**2 + self.q4**2)
        if norm < 1e-12:
            return replace(self, q1=1.0, q2=0.0, q3=0.0, q4=0.0)
        return replace(self, q1=self.q1 / norm, q2=self.q2 / norm, q3=self.q3 / norm, q4=self.q4 / norm)

    def inverse(self) -> Self:
        """返回共轭四元数 (对于单位四元数即为逆旋转)"""
        return replace(self, q2=-self.q2, q3=-self.q3, q4=-self.q4)

    def __mul__(self, other: Quaternion) -> Self:
        """四元数乘法：self * other (复合旋转)"""
        if not isinstance(other, Quaternion):
            return NotImplemented
        w1, x1, y1, z1 = self.to_tuple()
        w2, x2, y2, z2 = other.to_tuple()

        return self.__class__(
            q1=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            q2=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            q3=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            q4=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ).normalized()

    def as_SE3(self) -> NDArray:
        """返回包含当前旋转的 4x4 SE(3) 矩阵（平移为零）。"""
        mat33 = R.from_quat(self.to_list(), scalar_first=True).as_matrix()
        res = np.eye(4)
        res[:3, :3] = mat33
        return res

    def as_zyx(self, degrees: bool = True) -> list[float]:
        """转换为 ZYX 欧拉角"""
        return R.from_quat(self.to_list(), scalar_first=True).as_euler("ZYX", degrees=degrees).tolist()

    def as_euler(self, sequence: EulerSequence | str, degrees: bool = True) -> list[float]:
        """转换为指定序列的欧拉角"""
        seq_val = sequence.value if isinstance(sequence, EulerSequence) else sequence
        return R.from_quat(self.to_list(), scalar_first=True).as_euler(seq_val, degrees=degrees).tolist()

    # endregion

    # region 显示与复制
    def __str__(self) -> str:
        return f"{self.q1:.6f},{self.q2:.6f},{self.q3:.6f},{self.q4:.6f}"

    def __repr__(self) -> str:
        return f"Quaternion(w={self.q1}, x={self.q2}, y={self.q3}, z={self.q4})"

    def copy(self) -> Self:
        return replace(self)

    # endregion
