"""RecordReplay 可调运行参数的持久化存储。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..settings import ReplayArmSettings, ReplayOffsetSettings, ReplayServiceSettings


@dataclass(frozen=True, slots=True)
class RuntimeParameters:
    """允许本机 API 读取和修改的持久化运行参数。"""

    move_abs_j_end_linear_speed_mm_s: float = 1000.0
    move_abs_j_zone_mm: float = 10.0
    offset_trigger_speed_mm_s: float = 700.0
    agv_navigation_timeout_s: float = 600.0
    agv_navigation_poll_interval_s: float = 1.0
    non_motion_retry_count: int = 3
    non_motion_retry_delay_s: float = 0.5

    def validate(self) -> None:
        """校验所有可调参数，拒绝危险或无意义配置。"""

        if not 5.0 <= self.move_abs_j_end_linear_speed_mm_s <= 4000.0:
            raise ValueError("move_abs_j_end_linear_speed_mm_s 必须在 5 至 4000 之间")
        if self.move_abs_j_zone_mm < 0.0:
            raise ValueError("move_abs_j_zone_mm 不能小于 0")
        if not 5.0 <= self.offset_trigger_speed_mm_s <= 4000.0:
            raise ValueError("offset_trigger_speed_mm_s 必须在 5 至 4000 之间")
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
                move_abs_j_end_linear_speed_mm_s=self.move_abs_j_end_linear_speed_mm_s,
                move_abs_j_zone_mm=self.move_abs_j_zone_mm,
            ),
            offset=replace(
                ReplayOffsetSettings(),
                trigger_move_abs_j_end_linear_speed_mm_s=self.offset_trigger_speed_mm_s,
            ),
            agv_navigation_timeout_s=self.agv_navigation_timeout_s,
            agv_navigation_poll_interval_s=self.agv_navigation_poll_interval_s,
            non_motion_retry_count=self.non_motion_retry_count,
            non_motion_retry_delay_s=self.non_motion_retry_delay_s,
        )


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

    def update(self, changes: dict[str, float | int]) -> RuntimeParameters:
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
