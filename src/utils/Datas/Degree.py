from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Self, final, overload

import numpy as np

# 容差定义：1 角秒
ANGULAR_TOLERANCES = 1


@final
@dataclass(frozen=True, slots=True)
class Degree:
    """
    角度对象

    注意
    ----
    1. 内部存储为整数“角秒”(1/3600 度)，彻底消除累积浮点误差。
    """

    _second_value: int = 0

    def __init__(self, value: float | int | Degree = 0):
        if isinstance(value, Degree):
            object.__setattr__(self, "_second_value", value._second_value)
        elif isinstance(value, (float, int, np.number)):
            sec = int(round(float(value) * 3600.0))
            object.__setattr__(self, "_second_value", sec)
        else:
            raise TypeError(f"Degree 不支持的输入类型：{type(value)}")

    # ==========================================
    # 1. 核心内部工厂
    # ==========================================

    @classmethod
    def _from_raw_seconds(cls, seconds: int) -> Self:
        """内部极速构造方法：跳过浮点转换，直接操作角秒整数"""
        inst = cls.__new__(cls)
        object.__setattr__(inst, "_second_value", int(seconds))
        return inst

    # ==========================================
    # 2. 增强版工厂方法 (Factories)
    # ==========================================

    @classmethod
    def from_degrees(cls, degrees: float | int) -> Self:
        return cls(degrees)

    @classmethod
    def from_radians(cls, radians: float) -> Self:
        return cls(radians * 180.0 / math.pi)

    @classmethod
    def from_minutes(cls, minutes: float | int) -> Self:
        """从“分”构造"""
        return cls._from_raw_seconds(int(round(float(minutes) * 60.0)))

    @classmethod
    def from_seconds(cls, seconds: int | float) -> Self:
        """从“秒”构造"""
        return cls._from_raw_seconds(int(round(float(seconds))))

    @classmethod
    def from_dms(cls, d: int, m: int = 0, s: float = 0) -> Self:
        """从度 (D) 分 (M) 秒 (S) 构造"""
        sign = -1 if d < 0 else 1
        total_sec = abs(d) * 3600 + abs(m) * 60 + abs(s)
        return cls._from_raw_seconds(int(round(total_sec * sign)))

    @classmethod
    def from_coordinates(cls, x: float, y: float) -> Self:
        """根据 (x, y) 坐标计算角度 (atan2)"""
        return cls.from_radians(math.atan2(y, x))

    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        鲁棒解析字符串：支持 "10.5", "10°20'30\"", "90deg" 等。
        """
        # 尝试匹配度分秒格式
        dms_match = re.search(r"(-?\d+)°\s*(\d+)'\s*([\d\.]+)\"", string)
        if dms_match:
            return cls.from_dms(int(dms_match.group(1)), int(dms_match.group(2)), float(dms_match.group(3)))

        # 尝试提取第一个浮点数
        num_match = re.search(r"(-?[\d\.]+)", string)
        if num_match:
            return cls(float(num_match.group(1)))

        raise ValueError(f"无法解析角度字符串：{string}")

    # ==========================================
    # 3. 转换属性 (Properties & Conversion)
    # ==========================================

    @property
    def value(self) -> float:
        """返回度数值"""
        return self._second_value / 3600.0

    def as_radians(self) -> float:
        """转换为弧度"""
        return (self._second_value / 3600.0) * (math.pi / 180.0)

    def as_minutes(self) -> float:
        """转换为分"""
        return self._second_value / 60.0

    def as_seconds(self) -> int:
        """转换为秒"""
        return self._second_value

    def to_dms(self) -> tuple[int, int, int]:
        """返回 (度，分，秒) 元组"""
        total = abs(self._second_value)
        d, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return (d if self._second_value >= 0 else -d), m, s

    # ==========================================
    # 4. 几何与运动学高级方法 (Advanced Geometry)
    # ==========================================

    def normalized(self) -> Self:
        """归一化到 (-180, 180]"""
        FULL, HALF = 1296000, 648000
        remainder = self._second_value % FULL
        if remainder > HALF:
            remainder -= FULL
        return self._from_raw_seconds(remainder)

    def normalized_360(self) -> Self:
        """归一化到 [0, 360) (UI 显示常用区间)"""
        return self._from_raw_seconds(self._second_value % 1296000)

    def diff_to(self, target: Degree | float | int) -> Self:
        """计算到 target 的最短有符号角度差 (结果范围 (-180, 180])"""
        target_obj = target if isinstance(target, Degree) else Degree(target)
        delta = target_obj._second_value - self._second_value
        FULL, HALF = 1296000, 648000
        delta %= FULL
        if delta > HALF:
            delta -= FULL
        return self._from_raw_seconds(delta)

    def lerp(self, target: Degree | float | int, t: float) -> Self:
        """
        角度线性插值 (最短路径插值)。
        t=0 返回 self, t=1 返回 target。
        """
        diff = self.diff_to(target)
        return self + diff * t

    def clamped(self, min_deg: Degree | float, max_deg: Degree | float) -> Self:
        """将角度限制在 [min, max] 区间内 (不考虑 360 度回绕的简单截断)"""
        v_min = min_deg._second_value if isinstance(min_deg, Degree) else int(round(min_deg * 3600))
        v_max = max_deg._second_value if isinstance(max_deg, Degree) else int(round(max_deg * 3600))
        new_val = max(v_min, min(v_max, self._second_value))
        return self._from_raw_seconds(new_val)

    def complement(self) -> Self:
        """余角 (90° - self)"""
        return (self._from_raw_seconds(324000) - self).normalized()

    def supplement(self) -> Self:
        """补角 (180° - self)"""
        return (self._from_raw_seconds(648000) - self).normalized()

    def opposite(self) -> Self:
        """相反角/对顶角 (self + 180°)"""
        return (self + 180).normalized()

    # ==========================================
    # 5. 运算符重载 (Operators)
    # ==========================================

    def __add__(self, other: Degree | float | int) -> Self:
        o_sec = other._second_value if isinstance(other, Degree) else int(round(float(other) * 3600))
        return self._from_raw_seconds(self._second_value + o_sec)

    def __radd__(self, other: float | int) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Degree | float | int) -> Self:
        o_sec = other._second_value if isinstance(other, Degree) else int(round(float(other) * 3600))
        return self._from_raw_seconds(self._second_value - o_sec)

    def __rsub__(self, other: float | int) -> Self:
        return self.__class__(other) - self

    def __mul__(self, scalar: float | int) -> Self:
        return self._from_raw_seconds(int(round(self._second_value * float(scalar))))

    def __rmul__(self, scalar: float | int) -> Self:
        return self.__mul__(scalar)

    @overload
    def __truediv__(self, other: Degree) -> float: ...
    @overload
    def __truediv__(self, other: float | int) -> Self: ...

    def __truediv__(self, other: Degree | float | int) -> float | Self:
        if isinstance(other, Degree):
            if other._second_value == 0:
                raise ZeroDivisionError
            return self._second_value / other._second_value
        if float(other) == 0:
            raise ZeroDivisionError
        return self._from_raw_seconds(int(round(self._second_value / float(other))))

    def __neg__(self) -> Self:
        return self._from_raw_seconds(-self._second_value)

    def __abs__(self) -> Self:
        return self._from_raw_seconds(abs(self._second_value))

    # ==========================================
    # 6. 比较运算
    # ==========================================

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Degree):
            return self._second_value == other._second_value
        if isinstance(other, (float, int, np.number)):
            return self._second_value == int(round(float(other) * 3600))
        return False

    def __lt__(self, other: Degree | float | int) -> bool:
        o_sec = other._second_value if isinstance(other, Degree) else int(round(float(other) * 3600))
        return self._second_value < o_sec

    def isclose(self, other: Degree, tol_seconds: int = ANGULAR_TOLERANCES, normalize: bool = True) -> bool:
        self_value = self.copy()
        if normalize:
            self_value = self.normalized()
            other = other.normalized()
        diff = abs(self_value._second_value - other._second_value)
        actual_diff = min(diff, 1296000 - diff)
        return actual_diff <= tol_seconds

    # ==========================================
    # 7. 数学辅助 (Math)
    # ==========================================

    def sin(self) -> float:
        return math.sin(self.as_radians())

    def cos(self) -> float:
        return math.cos(self.as_radians())

    def tan(self) -> float:
        return math.tan(self.as_radians())

    @classmethod
    def asin(cls, ratio: float) -> Self:
        return cls.from_radians(math.asin(ratio))

    @classmethod
    def acos(cls, ratio: float) -> Self:
        return cls.from_radians(math.acos(ratio))

    @classmethod
    def atan(cls, ratio: float) -> Self:
        return cls.from_radians(math.atan(ratio))

    def copy(self) -> Self:
        return self._from_raw_seconds(self._second_value)

    # ==========================================
    # 8. 格式化与常数
    # ==========================================

    def __str__(self) -> str:
        return f"{self.value:.2f}"

    def __repr__(self) -> str:
        return f"Degree({self.value:.4f})"

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)

    def __float__(self) -> float:
        return self.value

    def __int__(self):
        return int(round(self.value))

    @classmethod
    def Zero(cls) -> Self:
        return cls._from_raw_seconds(0)

    @classmethod
    def Right(cls) -> Self:
        return cls._from_raw_seconds(324000)

    @classmethod
    def Straight(cls) -> Self:
        return cls._from_raw_seconds(648000)

    @classmethod
    def Full(cls) -> Self:
        return cls._from_raw_seconds(1296000)
