from __future__ import annotations

from collections.abc import Iterator, Sequence
from colorsys import rgb_to_hls
from dataclasses import dataclass, replace
from itertools import cycle
from typing import TYPE_CHECKING, Any, Self, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from OCC.Core.Quantity import Quantity_Color


def _normalize_channel(value: float | int, *, name: str) -> int:
    # 统一接受 [0,1] 浮点输入以及 [0,255] 整数/浮点输入。
    if isinstance(value, bool):
        raise TypeError(f"{name} 不支持 bool 类型")
    value_float = float(value)
    if 0.0 <= value_float <= 1.0:
        return int(round(value_float * 255.0))
    if 0.0 <= value_float <= 255.0:
        return int(round(value_float))
    raise ValueError(f"{name} 超出范围，期望 [0,1] 或 [0,255]，实际 {value}")


@dataclass(frozen=True, slots=True)
class Color(Sequence[int]):
    """颜色对象（默认 RGB，Alpha 可选），支持 [0,1] 与 [0,255] 输入。"""

    r: int = 0
    g: int = 0
    b: int = 0
    a: int | None = None

    # region 基础与序列协议
    def __post_init__(self) -> None:
        object.__setattr__(self, "r", _normalize_channel(self.r, name="r"))
        object.__setattr__(self, "g", _normalize_channel(self.g, name="g"))
        object.__setattr__(self, "b", _normalize_channel(self.b, name="b"))
        if self.a is not None:
            object.__setattr__(self, "a", _normalize_channel(self.a, name="a"))

    def _components(self) -> tuple[int, ...]:
        if self.a is None:
            return (self.r, self.g, self.b)
        return (self.r, self.g, self.b, self.a)

    def __iter__(self) -> Iterator[int]:
        yield from self._components()

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        return self._components()[index]

    def __len__(self) -> int:
        return len(self._components())

    def __array__(self, dtype: Any = None) -> NDArray[np.int64] | NDArray[np.float64]:
        return np.array(self._components(), dtype=dtype)

    def __str__(self) -> str:
        if self.a is None:
            return f"({self.r}, {self.g}, {self.b})"
        return f"({self.r}, {self.g}, {self.b}, {self.a})"

    def __repr__(self) -> str:
        if self.a is None:
            return f"(r={self.r}, g={self.g}, b={self.b})"
        return f"(r={self.r}, g={self.g}, b={self.b}, a={self.a})"

    # endregion

    # region 构造方法
    @classmethod
    def from_rgb(
        cls,
        r: int | float | Sequence[int | float],
        g: int | float = 0,
        b: int | float = 0,
        a: int | float | None = None,
    ) -> Self:
        if isinstance(r, Sequence) and not isinstance(r, (str, bytes)):
            if len(r) == 3:
                return cls(r[0], r[1], r[2], a)
            if len(r) == 4:
                return cls(r[0], r[1], r[2], r[3])
            raise ValueError("from_rgb 序列输入仅支持长度 3 或 4")
        return cls(r, g, b, a)

    @classmethod
    def from_hex(cls, value: str) -> Self:
        text = value.strip().lstrip("#")
        if len(text) == 6:
            return cls(int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), None)
        if len(text) == 8:
            return cls(int(text[2:4], 16), int(text[4:6], 16), int(text[6:8], 16), int(text[0:2], 16))
        raise ValueError("from_hex 仅支持 RRGGBB 或 AARRGGBB")

    # endregion

    # region 数据访问与转换
    def copy(self) -> Self:
        return replace(self)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @property
    def rgba(self) -> tuple[int, int, int, int] | None:
        if self.a is None:
            return None
        return (self.r, self.g, self.b, self.a)

    @property
    def has_alpha(self) -> bool:
        return self.a is not None

    def with_alpha(self, alpha: int | float | None = 255) -> Self:
        return self.__class__(self.r, self.g, self.b, alpha)

    def to_list(self, normalized: bool = False, include_alpha: bool | None = None) -> list[int] | list[float]:
        return list(self.to_tuple(normalized=normalized, include_alpha=include_alpha))

    def as_array(
        self, normalized: bool = False, include_alpha: bool | None = None
    ) -> NDArray[np.int64] | NDArray[np.float64]:
        values = self.to_tuple(normalized=normalized, include_alpha=include_alpha)
        if normalized:
            return np.array(values, dtype=np.float64)
        return np.array(values, dtype=np.int64)

    def to_tuple(
        self, normalized: bool = False, include_alpha: bool | None = None
    ) -> tuple[int, ...] | tuple[float, ...]:
        if include_alpha is None:
            include_alpha = self.has_alpha

        if normalized:
            rgb = (self.r / 255.0, self.g / 255.0, self.b / 255.0)
            if include_alpha:
                alpha = (self.a / 255.0) if self.a is not None else 1.0
                return (*rgb, alpha)
            return rgb

        if include_alpha:
            alpha_int = self.a if self.a is not None else 255
            return (self.r, self.g, self.b, alpha_int)
        return (self.r, self.g, self.b)

    def to_hex(self) -> str:
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def to_argb(self) -> str:
        alpha = self.a if self.a is not None else 255
        return f"#{alpha:02X}{self.r:02X}{self.g:02X}{self.b:02X}"

    def to_hsl(self, include_alpha: bool = False) -> tuple[int, int, int] | tuple[int, int, int, int]:
        h, l, s = rgb_to_hls(self.r / 255.0, self.g / 255.0, self.b / 255.0)
        hsl = (int(round(h * 360.0)), int(round(s * 100.0)), int(round(l * 100.0)))
        if include_alpha:
            alpha = self.a if self.a is not None else 255
            return (*hsl, alpha)
        return hsl

    def to_quantity(self) -> Quantity_Color:
        try:
            from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("to_quantity 需要安装 pythonocc-core (OCC)") from exc
        return Quantity_Color(self.r / 255.0, self.g / 255.0, self.b / 255.0, Quantity_TOC_RGB)

    # endregion


# region 预置颜色
rainbow_color: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 255),
    (128, 0, 128),
    (255, 165, 0),
    (0, 0, 255),
    (0, 0, 0),
    (200, 200, 200),
]
rainbow_colors: tuple[Color, ...] = tuple(Color.from_rgb(c) for c in rainbow_color)
color_list = cycle(rainbow_colors)
# endregion
