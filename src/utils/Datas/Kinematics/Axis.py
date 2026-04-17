from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Self

import numpy as np

from ..Point import Point
from ..Vector import Vector
from .Transform import Transform


@dataclass(frozen=True, slots=True)
class Axis:
    """笛卡尔坐标系，由原点和三条轴向量组成。"""

    origin: Point = field(default_factory=Point.Origin)
    x_axis: Vector = field(default_factory=lambda: Vector(1, 0, 0))
    y_axis: Vector = field(default_factory=lambda: Vector(0, 1, 0))
    z_axis: Vector = field(default_factory=lambda: Vector(0, 0, 1))
    is_right_handed: bool = True

    # region 基础协议
    def __iter__(self) -> Iterator[Point | Vector]:
        """按 `origin, x_axis, y_axis, z_axis` 顺序迭代。"""
        yield self.origin
        yield self.x_axis
        yield self.y_axis
        yield self.z_axis

    def __hash__(self) -> int:
        return hash((self.origin, self.x_axis, self.y_axis, self.z_axis, self.is_right_handed))

    # endregion

    # region 构造与规范化
    def __post_init__(self):
        ux = self.x_axis.normalized()
        uy = self.z_axis.cross(ux).normalized()
        uz = ux.cross(uy).normalized()
        if not self.is_right_handed:
            uy = uy.negated()
        object.__setattr__(self, "x_axis", ux)
        object.__setattr__(self, "y_axis", uy)
        object.__setattr__(self, "z_axis", uz)

    @classmethod
    def World(cls) -> Axis:
        """返回世界坐标系。"""
        return cls()

    @classmethod
    def from_points(cls, origin: Point, point_on_x: Point, point_on_xy_plane: Point) -> Self:
        """通过三点构造坐标系。"""
        vec_x = Vector.from_points(origin, point_on_x)
        vec_xy = Vector.from_points(origin, point_on_xy_plane)
        vec_z = vec_x.cross(vec_xy)
        return cls(origin=origin, x_axis=vec_x, z_axis=vec_z)

    @classmethod
    def from_transform(cls, t: Transform) -> Self:
        """从刚体变换恢复坐标系。"""
        origin = Point(t.translation.x, t.translation.y, t.translation.z)
        mat = t.as_SE3()
        ux = Vector(mat[0, 0], mat[1, 0], mat[2, 0])
        uy = Vector(mat[0, 1], mat[1, 1], mat[2, 1])
        uz = Vector(mat[0, 2], mat[1, 2], mat[2, 2])
        return cls(origin=origin, x_axis=ux, y_axis=uy, z_axis=uz)

    # endregion

    # region 坐标变换
    def to_transform(self) -> Transform:
        """将坐标系转换为 SE(3) 变换。"""
        mat = np.eye(4, dtype=np.float64)
        mat[0:3, 0] = [self.x_axis.x, self.x_axis.y, self.x_axis.z]
        mat[0:3, 1] = [self.y_axis.x, self.y_axis.y, self.y_axis.z]
        mat[0:3, 2] = [self.z_axis.x, self.z_axis.y, self.z_axis.z]
        mat[0:3, 3] = [self.origin.x, self.origin.y, self.origin.z]
        return Transform.from_SE3(mat)

    def transformed(self, t: Transform) -> Self:
        """对坐标系整体应用刚体变换。"""
        new_origin = self.origin.transformed(t)
        new_x = self.x_axis.transformed(t)
        new_z = self.z_axis.transformed(t)
        return self.__class__(new_origin, new_x, new_z, is_right_handed=self.is_right_handed)

    def project_point(self, p_world: Point) -> Point:
        """将世界坐标系中的点投影到当前局部坐标系。"""
        relative_v = Vector.from_points(self.origin, p_world)
        return Point(relative_v.dot(self.x_axis), relative_v.dot(self.y_axis), relative_v.dot(self.z_axis))

    def point_at(self, local_x: float, local_y: float, local_z: float) -> Point:
        """由局部坐标求世界坐标点。"""
        offset = (self.x_axis * local_x) + (self.y_axis * local_y) + (self.z_axis * local_z)
        return self.origin.translation(offset)

    # endregion

    # region 显示与复制
    def __str__(self) -> str:
        return f"Axis(Origin: {self.origin}, X: {self.x_axis}, Z: {self.z_axis})"

    def copy(self) -> Axis:
        return replace(self)

    # endregion
