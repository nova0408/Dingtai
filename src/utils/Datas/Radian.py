from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Self, final


@final
@dataclass(frozen=True, slots=True)
class Radian:
    """
    弧度对象 (不可变、不可继承、高性能)。
    提供与角度的转换及基础三角运算。
    """

    value: float = 0.0

    # ==========================================
    # 构造工厂 (Factories)
    # ==========================================

    @classmethod
    def from_degrees(cls, degrees: float | int) -> Radian:
        """从角度构造"""
        return cls(float(degrees) * math.pi / 180.0)

    @classmethod
    def from_radians(cls, radian: float | int) -> Radian:
        """从弧度构造"""
        return cls(float(radian))

    # ==========================================
    # 转换与属性 (Conversion)
    # ==========================================

    def as_degrees(self) -> float:
        """转换为角度值"""
        return self.value * 180.0 / math.pi

    def normalized(self) -> Radian:
        """
        将弧度归一化到 [-pi, pi] 之间。
        在机器人关节限位计算中非常有用。
        """
        # 使用 math.atan2(sin, cos) 是最稳健的归一化方法
        norm_val = math.atan2(math.sin(self.value), math.cos(self.value))
        return Radian(norm_val)

    # ==========================================
    # 数学运算 (Arithmetic)
    # ==========================================

    def __add__(self, other: Radian | float | int) -> Radian:
        """弧度相加"""
        val = other.value if isinstance(other, Radian) else other
        return Radian(self.value + val)

    def __sub__(self, other: Radian | float | int) -> Radian:
        """弧度相减"""
        val = other.value if isinstance(other, Radian) else other
        return Radian(self.value - val)

    def __neg__(self) -> Radian:
        """弧度取反"""
        return Radian(-self.value)

    def __mul__(self, scalar: float | int) -> Radian:
        """标量乘法"""
        return Radian(self.value * scalar)

    def __rmul__(self, scalar: float | int) -> Radian:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float | int) -> Radian:
        """标量除法"""
        if scalar == 0:
            raise ZeroDivisionError("Radian 不能除以零")
        return Radian(self.value / scalar)

    # ==========================================
    # 三角函数封装 (方便链式调用)
    # ==========================================

    def sin(self) -> float:
        return math.sin(self.value)

    def cos(self) -> float:
        return math.cos(self.value)

    def tan(self) -> float:
        return math.tan(self.value)

    # ==========================================
    # 输出格式
    # ==========================================

    def __str__(self) -> str:
        return f"{self.value:.6f} rad"

    def __repr__(self) -> str:
        return f"Radian({self.value})"

    def copy(self) -> Radian:
        return replace(self)
