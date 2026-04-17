from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

from .Point import Point


@dataclass(frozen=True)
class Box:

    __slots__ = ("left_bottom_down", "right_top_up")
    left_bottom_down: Point
    right_top_up: Point

    def __hash__(self):
        # 使用元组的哈希值，并适当处理浮点数精度
        return hash((hash(self.left_bottom_down), hash(self.right_top_up)))

    def __str__(self) -> str:
        return f"{self.left_bottom_down}, {self.right_top_up}"

    def __repr__(self) -> str:
        return f"{self.left_bottom_down}, {self.right_top_up}"

    def to_list(self, as_center: bool = False):
        if as_center:
            return [*self.center.to_list(), *self.bounds]
        return [*self.left_bottom_down.to_list(), *self.right_top_up.to_list()]

    def transformed(self, trans: NDArray):
        new_left = self.left_bottom_down.transformed(trans)
        new_right = self.right_top_up.transformed(trans)
        return Box(new_left, new_right)

    def expand(self, x: float, y: float, z: float):
        """扩展盒子边界

        Parameters
        ----------
        x
            x 方向扩展量
        y
            y 方向扩展量
        z
            z 方向扩展量

        Returns
        -------
            扩展后的 Box
        """
        return Box(
            Point(self.left_bottom_down.x - x, self.left_bottom_down.y - y, self.left_bottom_down.z - z),
            Point(self.right_top_up.x + x, self.right_top_up.y + y, self.right_top_up.z + z),
        )

    @overload
    @classmethod
    def from_list(cls, p_list: Sequence[Point]) -> Self: ...
    @overload
    @classmethod
    def from_list(cls, p_list: Sequence[float]) -> Self: ...

    @classmethod
    def from_list(cls, p_list: Sequence[float | Point | Sequence[float]]):
        a, b, c, i, j, k = [0.0 for i in range(6)]

        def _resort(l: float, r: float):
            if l > r:
                l, r = r, l
            return l, r

        if len(p_list) == 2:

            def _check_point(p):
                if isinstance(p, Point):
                    return p.to_list()
                elif isinstance(p, Sequence):
                    if not len(p) == 3:
                        raise ValueError("输入不合法")
                    if not all(isinstance(x, (float, int)) for x in p):
                        raise ValueError("输入不合法")
                    return p
                raise ValueError(f"输入不合法 {type(p)}")

            p1, p2 = p_list
            a, b, c = _check_point(p1)
            i, j, k = _check_point(p2)
        elif len(p_list) == 6:
            if not all(isinstance(p, (float, int)) for p in p_list):
                raise ValueError("需要 6 个数值")
            a, b, c, i, j, k = p_list
        else:
            raise ValueError(f"需要 2 个角点或 6 个数值:{len(p_list)}")
        a, i = _resort(a, i)
        b, j = _resort(b, j)
        c, k = _resort(c, k)
        return cls(Point(a, b, c), Point(i, j, k))

    @classmethod
    def from_center(cls, center: Point | Sequence[float] | np.ndarray, bound: Sequence[float]):
        """
        从中心点和总尺寸构造 Box (min = center - bound/2, max = center + bound/2)

        Parameters
        ----------
        center
            中心点 [cx, cy, cz]，支持 Point, list, tuple, np.ndarray
        bound
            总尺寸 [width, height, depth]，支持 list, tuple, np.ndarray
        """

        # 1. 校验输入是否为可迭代对象且长度为 3
        def validate_input(val, name):
            if not hasattr(val, "__getitem__"):
                raise TypeError(f"{name} 必须是序列类型 (如 list, tuple, Point)，当前类型为：{type(val)}")
            try:
                if len(val) != 3:
                    raise ValueError(f"{name} 的长度必须为 3，当前长度为：{len(val)}")
            except TypeError:
                # 某些特殊对象可能没实现 len() 但支持索引，补充尝试索引访问
                try:
                    _ = val[0], val[1], val[2]
                except Exception:
                    raise ValueError(f"{name} 无法通过索引访问 3 个元素")

        validate_input(center, "center")
        validate_input(bound, "size")

        # 2. 尝试提取数值并检查是否为数字
        try:
            coords = [float(center[i]) for i in range(3)]
            dims = [float(bound[i]) for i in range(3)]
        except (ValueError, TypeError) as e:
            raise TypeError(f"center 和 size 中的所有元素必须为数值类型：{e}")

        # 3. 校验尺寸是否合法 (尺寸不能为负)
        if any(d < 0 for d in dims):
            raise ValueError(f"size 尺寸不能为负数：{dims}")

        # 4. 计算几何边界 (不在这里做 round，保持高精度计算)
        cx, cy, cz = coords
        sx, sy, sz = dims

        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

        return cls(Point(cx - hx, cy - hy, cz - hz), Point(cx + hx, cy + hy, cz + hz))

    @property
    def bounds(self):
        xmin, xmax = sorted([self.left_bottom_down.x, self.right_top_up.x])
        ymin, ymax = sorted([self.left_bottom_down.y, self.right_top_up.y])
        zmin, zmax = sorted([self.left_bottom_down.z, self.right_top_up.z])
        return (float(xmax - xmin), float(ymax - ymin), float(zmax - zmin))

    @property
    def center(self):
        return Point(
            (self.left_bottom_down.x + self.right_top_up.x) / 2,
            (self.left_bottom_down.y + self.right_top_up.y) / 2,
            (self.left_bottom_down.z + self.right_top_up.z) / 2,
        )

    @property
    def min_x(self):
        return self.left_bottom_down.x

    @property
    def max_x(self):
        return self.right_top_up.x

    @property
    def min_y(self):
        return self.left_bottom_down.y

    @property
    def max_y(self):
        return self.right_top_up.y

    @property
    def min_z(self):
        return self.left_bottom_down.z

    @property
    def max_z(self):
        return self.right_top_up.z


# region 转换方法
def box_to_o3d_bbox(box: Box):
    try:
        import open3d as o3d

        return o3d.geometry.AxisAlignedBoundingBox(
            min_bound=box.left_bottom_down.as_array(), max_bound=box.right_top_up.as_array()
        )
    except ImportError:
        raise ImportError("open3d 模块未安装")
