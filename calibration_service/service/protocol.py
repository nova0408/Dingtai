"""手眼标定与先验记录服务的 HTTP 响应协议。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import CALIBRATION_SERVICE_VERSION


@dataclass(frozen=True, slots=True)
class CalibrationResponse:
    """服务请求的统一 JSON 响应。"""

    service_version: str = CALIBRATION_SERVICE_VERSION
    "服务功能版本。"

    state: str = "idle"
    "服务状态；仅表示当前计算任务，不表示设备状态。"

    accepted: bool = True
    "请求是否被服务接受；结果替换仍可能等待前端二次确认。"

    data: dict[str, object] = field(default_factory=dict)
    "请求成功时的结果数据。"

    error: str | None = None
    "请求失败时的可复制错误文本。"
