from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Self, final, overload

import numpy as np
from loguru import logger

from .Quaternion import Quaternion
from .TransformProtocol import MatrixConstructible, MatrixSerializable
from .Translation import Translation


@final
@dataclass(frozen=True, slots=True)
class Transform(MatrixSerializable, MatrixConstructible):
    """SE(3) 刚体变换，由平移与旋转构成。"""

    translation: Translation = field(default_factory=lambda: Translation(0, 0, 0))
    rotation: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))

    # region 基础协议
    def __hash__(self) -> int:
        return hash((self.translation, self.rotation))

    # endregion

    # region 构造方法

    @classmethod
    def from_list(cls, li: Sequence[float]) -> Self:
        """从列表构造变换
        - 7 个参数：[x, y, z, q1, q2, q3, q4] (ABB 格式：q1 为 w)
        - 6 个参数：[x, y, z, rz, ry, rx] (ZYX 欧拉角)
        """
        if len(li) == 7:
            return cls(Translation(*li[:3]), Quaternion(*li[3:]))
        elif len(li) == 6:
            return cls(Translation(*li[:3]), Quaternion.from_zyx(li[3], li[4], li[5]))
        else:
            raise ValueError(f"Transform.from_list 需要 6 或 7 个参数，当前收到 {len(li)} 个")

    @classmethod
    def Identity(cls) -> Self:
        """返回单位变换。"""
        return cls(Translation.Zero(), Quaternion.Identity())

    @classmethod
    def from_SE3(cls, mat: np.ndarray) -> Self:
        """从 4x4 SE(3) 矩阵构造"""
        if mat.shape != (4, 4):
            raise ValueError(f"期望 4x4 矩阵，当前维度：{mat.shape}")
        return cls(Translation(*mat[:3, 3]), Quaternion.from_SO3(mat[:3, :3]))

    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        从字符串构造 Transform，兼容 ABB 格式。
        策略：提取所有浮点数并根据数量判断逻辑。
        """
        try:
            cleaned = string.replace("[", " ").replace("]", " ").replace(",", " ")
            values = [float(x) for x in cleaned.split()]

            if not values:
                return cls()

            return cls.from_list(values)
        except Exception as e:
            logger.error(f"Transform.from_str 解析失败：{e}, 输入：'{string}'")
            return cls()

    # endregion

    # region 运算方法

    @overload
    def __matmul__(self, other: Transform) -> Self: ...

    @overload
    def __matmul__(self, other: Translation) -> Self: ...

    @overload
    def __matmul__(self, other: Quaternion) -> Self: ...

    def __matmul__(self, other) -> Self:
        """实现变换复合：`self @ other`。"""
        if not isinstance(other, Transform | Translation | Quaternion):
            return NotImplemented
        return self.from_SE3(self.as_SE3() @ other.as_SE3())

    def as_SE3(self) -> np.ndarray:
        """转换为 4x4 SE(3) 齐次变换矩阵"""
        t_mat = self.translation.as_SE3()
        r_mat = self.rotation.as_SE3()
        return t_mat @ r_mat

    # endregion

    # region 序列化与显示

    def to_list(self, zyx: bool = False) -> list[float]:
        """转换为列表格式
        - 7 个参数：[x, y, z, q1, q2, q3, q4] (ABB 格式：q1 为 w)
        - 6 个参数：[x, y, z, rz, ry, rx] (ZYX 欧拉角)
        """
        base = list(self.translation.to_list())
        if zyx:
            return base + list(self.rotation.as_zyx(degrees=True))
        return base + list(self.rotation.to_list())

    def as_string(self, with_bracket: bool = False, zyx: bool = False, with_name: bool = False) -> str:
        """转换为 ABB RAPID 兼容字符串"""
        t = self.translation
        q = self.rotation
        if with_name:
            if zyx:
                rz, ry, rx = q.as_zyx(degrees=True)
                return f"x={t.x:.2f}, y={t.y:.2f}, z={t.z:.2f}, rz={rz:.1f}, ry={ry:.1f}, rx={rx:.1f}"
            return f"x={t.x:.2f}, y={t.y:.2f}, z={t.z:.2f}, q1={q.q1:.4f}, q2={q.q2:.4f}, q3={q.q3:.4f}, q4={q.q4:.4f}"
        if with_bracket:
            return f"[{t.x:.4f},{t.y:.4f},{t.z:.4f}],[{q.q1:.6f},{q.q2:.6f},{q.q3:.6f},{q.q4:.6f}]"

        if zyx:
            rz, ry, rx = q.as_zyx(degrees=True)
            return f"{t.x:.4f}, {t.y:.4f}, {t.z:.4f}, {rz:.4f}, {ry:.4f}, {rx:.4f}"

        return f"{t.x:.4f}, {t.y:.4f}, {t.z:.4f}, {q.q1:.6f}, {q.q2:.6f}, {q.q3:.6f}, {q.q4:.6f}"

    # endregion

    # region 不可变更新
    def with_component(self, value: float, axis: str) -> Self:
        """返回指定分量被替换后的新变换。"""
        x, y, z = self.translation.to_list()
        rz, ry, rx = self.rotation.as_zyx(degrees=True)

        match axis.lower():
            case "x":
                x = value
            case "y":
                y = value
            case "z":
                z = value
            case "rz":
                rz = value
            case "ry":
                ry = value
            case "rx":
                rx = value
            case _:
                raise ValueError(f"未知轴名称：{axis}")

        return self.from_list([x, y, z, rz, ry, rx])

    def copy(self) -> Self:
        """返回当前对象副本。"""
        return replace(self)

    # endregion
