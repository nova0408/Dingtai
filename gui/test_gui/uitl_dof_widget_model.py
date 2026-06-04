from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DoFWidgetModel:
    """单个自由度调试控件的数据模型。"""

    name: str
    minimum: float
    maximum: float
    unit: str = ""
    feedback_value: float = 0.0
    command_value: float = 0.0
    step: float = 1.0

    def clamp(self, value: float) -> float:
        """将输入值限制在当前自由度范围内。"""

        return max(self.minimum, min(self.maximum, float(value)))

    def set_feedback_value(self, value: float) -> float:
        """更新真实反馈值并返回裁剪后的有效值。"""

        self.feedback_value = self.clamp(value)
        return self.feedback_value

    def set_command_value(self, value: float) -> float:
        """更新待发送目标值并返回裁剪后的有效值。"""

        self.command_value = self.clamp(value)
        return self.command_value

    def offset_command_from_feedback(self, delta: float) -> float:
        """以真实反馈值为基准生成增量目标值。"""

        return self.set_command_value(self.feedback_value + float(delta))

    def offset_command(self, delta: float) -> float:
        """以上一次命令值为基准生成连续增量目标值。"""

        return self.set_command_value(self.command_value + float(delta))
