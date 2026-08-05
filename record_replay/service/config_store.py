"""RecordReplay 可调运行参数的持久化存储。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ..settings import ReplayArmSettings, ReplayOffsetSettings, ReplayServiceSettings

NumericValue = float | int
MotionLevelMap = dict[str, NumericValue]
RuntimeParameterValue = NumericValue | MotionLevelMap


def _default_left_speed_levels() -> MotionLevelMap:
    return _to_json_level_map(ReplayArmSettings().left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence)


def _default_right_speed_levels() -> MotionLevelMap:
    return _to_json_level_map(ReplayArmSettings().right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence)


def _default_left_zone_levels() -> MotionLevelMap:
    return _to_json_level_map(ReplayArmSettings().left_move_abs_j_zone_mm_by_csv_sequence)


def _default_right_zone_levels() -> MotionLevelMap:
    return _to_json_level_map(ReplayArmSettings().right_move_abs_j_zone_mm_by_csv_sequence)


@dataclass(frozen=True, slots=True)
class RuntimeParameters:
    """允许本机 API 读取和修改的持久化运行参数。

    MoveAbsJ 的速度和 zone 按机械臂侧别、CSV 数字序号分别保存；键 ``-1``
    是对应侧别的默认级别，其余键覆盖指定 CSV 阶段。
    """

    left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence: MotionLevelMap = field(
        default_factory=_default_left_speed_levels
    )
    right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence: MotionLevelMap = field(
        default_factory=_default_right_speed_levels
    )
    left_move_abs_j_zone_mm_by_csv_sequence: MotionLevelMap = field(
        default_factory=_default_left_zone_levels
    )
    right_move_abs_j_zone_mm_by_csv_sequence: MotionLevelMap = field(
        default_factory=_default_right_zone_levels
    )
    agv_navigation_timeout_s: float = 600.0
    agv_navigation_poll_interval_s: float = 1.0
    non_motion_retry_count: int = 3
    non_motion_retry_delay_s: float = 0.5

    def validate(self) -> None:
        """校验全部可调参数，拒绝缺少默认级别或危险运动参数。"""

        _validate_motion_levels(
            "left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence",
            self.left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence,
            minimum=5.0,
            maximum=4000.0,
        )
        _validate_motion_levels(
            "right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence",
            self.right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence,
            minimum=5.0,
            maximum=4000.0,
        )
        _validate_motion_levels(
            "left_move_abs_j_zone_mm_by_csv_sequence",
            self.left_move_abs_j_zone_mm_by_csv_sequence,
            minimum=0.0,
        )
        _validate_motion_levels(
            "right_move_abs_j_zone_mm_by_csv_sequence",
            self.right_move_abs_j_zone_mm_by_csv_sequence,
            minimum=0.0,
        )
        if self.agv_navigation_timeout_s <= 0.0 or self.agv_navigation_poll_interval_s <= 0.0:
            raise ValueError("AGV 超时和轮询周期必须大于 0")
        if self.non_motion_retry_count <= 0 or self.non_motion_retry_delay_s < 0.0:
            raise ValueError("非运动重试次数必须大于 0，间隔不能小于 0")

    def to_service_settings(self) -> ReplayServiceSettings:
        """转换为业务层不可变配置。"""

        defaults = ReplayServiceSettings()
        return replace(
            defaults,
            arm=replace(
                ReplayArmSettings(),
                left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence=_to_level_entries(
                    self.left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence
                ),
                right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence=_to_level_entries(
                    self.right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence
                ),
                left_move_abs_j_zone_mm_by_csv_sequence=_to_level_entries(
                    self.left_move_abs_j_zone_mm_by_csv_sequence
                ),
                right_move_abs_j_zone_mm_by_csv_sequence=_to_level_entries(
                    self.right_move_abs_j_zone_mm_by_csv_sequence
                ),
            ),
            offset=ReplayOffsetSettings(),
            agv_navigation_timeout_s=self.agv_navigation_timeout_s,
            agv_navigation_poll_interval_s=self.agv_navigation_poll_interval_s,
            non_motion_retry_count=self.non_motion_retry_count,
            non_motion_retry_delay_s=self.non_motion_retry_delay_s,
        )


def _validate_motion_levels(
    name: str,
    levels: MotionLevelMap,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    """校验一个 JSON 形式的 CSV 数字序号到数值映射。"""

    if not isinstance(levels, dict) or "-1" not in levels:
        raise ValueError(f"{name} 必须是包含 -1 默认级别的 JSON object")
    for sequence_text, value in levels.items():
        if not isinstance(sequence_text, str):
            raise ValueError(f"{name} 的 CSV 序号键必须是字符串")
        try:
            sequence = int(sequence_text)
        except ValueError as error:
            raise ValueError(f"{name} 的 CSV 序号无效：{sequence_text}") from error
        if sequence < -1:
            raise ValueError(f"{name} 的 CSV 序号不能小于 -1：{sequence}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name}[{sequence_text}] 必须是数字")
        if not minimum <= float(value) or (maximum is not None and float(value) > maximum):
            limit_text = f" 至 {maximum}" if maximum is not None else ""
            raise ValueError(f"{name}[{sequence_text}] 必须在 {minimum}{limit_text} 范围内")


def _to_level_entries(levels: MotionLevelMap) -> tuple[tuple[int, float], ...]:
    """将 JSON 字符串键映射转换为业务层的有序整数键元组。"""

    return tuple(sorted((int(sequence), float(value)) for sequence, value in levels.items()))


def _to_json_level_map(entries: tuple[tuple[int, float], ...]) -> MotionLevelMap:
    """将 settings.py 的默认级别转换为 JSON object 的字符串键。"""

    return {str(sequence): value for sequence, value in entries}


class RuntimeConfigStore:
    """以 UTF-8 JSON 原子保存可调参数，修改后立即成为新默认值。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> RuntimeParameters:
        """读取配置；首次部署时创建默认配置文件。"""

        if not self._path.is_file():
            parameters = RuntimeParameters()
            self.save(parameters)
            return parameters
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("运行参数配置必须是 JSON object")
        parameters = RuntimeParameters(**payload)
        parameters.validate()
        return parameters

    def update(self, changes: dict[str, RuntimeParameterValue]) -> RuntimeParameters:
        """按字段更新配置并持久化，未知字段直接拒绝。"""

        current = self.load()
        allowed = set(asdict(current))
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"未知运行参数：{sorted(unknown)}")
        updated = replace(current, **changes)
        updated.validate()
        self.save(updated)
        return updated

    def save(self, parameters: RuntimeParameters) -> None:
        """使用同目录临时文件原子替换配置。"""

        parameters.validate()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(asdict(parameters), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._path)
