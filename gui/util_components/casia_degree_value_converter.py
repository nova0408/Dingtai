from __future__ import annotations

from dataclasses import dataclass

from src.utils.datas import Degree

from .casia_value_converter import CasiaValueConverter


@dataclass(frozen=True, slots=True)
class DegreeValueConverter(CasiaValueConverter):
    """
    Degree 数据结构滑块转换器。

    QSlider 只能保存整数，本转换器通过 ``units_per_degree`` 定义整数值与角度值之间
    的缩放关系。例如 ``units_per_degree=10`` 表示滑块值 123 对应 12.3 度。
    """

    units_per_degree: int = 10
    display_precision: int = 1
    suffix: str = "°"

    def __post_init__(self) -> None:
        if self.units_per_degree <= 0:
            raise ValueError("units_per_degree 必须大于 0")
        if self.display_precision < 0:
            raise ValueError("display_precision 不能小于 0")

    def convert(self, value: int) -> str:
        degree = Degree(float(value) / self.units_per_degree)
        return f"{degree.value:.{self.display_precision}f}{self.suffix}"

    def convert_edit(self, value: int) -> str:
        degree = Degree(float(value) / self.units_per_degree)
        return f"{degree.value:.{self.display_precision}f}"

    def convert_back(self, text: str) -> int:
        raw_text = text.strip()
        if self.suffix and raw_text.endswith(self.suffix):
            raw_text = raw_text[: -len(self.suffix)].strip()
        degree = Degree.from_str(raw_text)
        return int(round(degree.value * self.units_per_degree))
