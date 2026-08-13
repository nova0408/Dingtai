"""RobotControl HTTP 客户端。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

_DIRECT_OPENER = build_opener(ProxyHandler({}))


class RobotControlClient:
    """访问 RobotControl 服务的同步客户端。

    GET 方法可用于只读状态检查。POST 方法保留给现场人员明确发起的控制操作；Codex
    不调用这些方法进行测试。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6500",
        timeout_s: float = 5.0,
        api_prefix: str = "",
    ) -> None:
        """创建 HTTP 客户端。"""

        self._base_url = base_url.rstrip("/")
        self._api_prefix = _normalize_prefix(api_prefix)
        self._timeout_s = float(timeout_s)

    def get_health(self) -> dict[str, Any]:
        """读取不访问硬件的服务健康状态。"""

        return self._request("GET", "/api/v1/health")

    def get_status(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """读取全部 qmlinker 与 AR5 设备状态。"""

        return self._request("GET", "/api/v1/status", timeout_s=timeout_s)

    def get_devices(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """读取设备状态别名接口。"""

        return self._request("GET", "/api/v1/devices", timeout_s=timeout_s)

    def get_agv_navigation_targets(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取 AGV 当前地图和可导航目标点。"""

        return self._request("GET", "/api/v1/agv/targets", timeout_s=timeout_s)

    def get_agv_base_state(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取 AGV 底盘状态。"""

        return self._request(
            "GET", "/api/v1/agv/base-state", timeout_s=timeout_s
        )

    def get_agv_base_mode(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取 AGV 底盘控制模式和工作模式。"""

        return self._request(
            "GET", "/api/v1/agv/base-mode", timeout_s=timeout_s
        )

    def get_agv_base_operation_state(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取 AGV 底盘原始运行状态位。"""

        return self._request(
            "GET",
            "/api/v1/agv/base-operation-state",
            timeout_s=timeout_s,
        )

    def get_agv_base_task_process(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取 AGV 当前任务和动作进度。"""

        return self._request(
            "GET",
            "/api/v1/agv/base-task-process",
            timeout_s=timeout_s,
        )

    def get_agv_base_battery(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取 AGV 底盘电量和充电状态。"""

        return self._request(
            "GET", "/api/v1/agv/base-battery", timeout_s=timeout_s
        )

    def get_ar5_soft_limits(self, side: str, timeout_s: float = 5.0) -> dict[str, Any]:
        """读取指定 AR5 七个轴的软限位。"""

        return self._request(
            "GET", f"/api/v1/ar5/{side}/soft-limits", timeout_s=timeout_s
        )

    def qmlinker_set_head(
        self,
        *,
        enable: bool | None = None,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
    ) -> dict[str, Any]:
        """显式设置头部使能或目标角度。"""

        body: dict[str, object] = {}
        if enable is not None:
            body["enable"] = enable
        if yaw_deg is not None:
            body["yaw_deg"] = yaw_deg
        if pitch_deg is not None:
            body["pitch_deg"] = pitch_deg
        if not body:
            raise ValueError("head request requires at least one field")
        return self._request("POST", "/api/v1/head", body)

    def qmlinker_set_lift(
        self,
        *,
        enable: bool | None = None,
        height_mm: float | None = None,
    ) -> dict[str, Any]:
        """显式设置升降使能或高度。"""

        body: dict[str, object] = {}
        if enable is not None:
            body["enable"] = enable
        if height_mm is not None:
            body["height_mm"] = height_mm
        if not body:
            raise ValueError("lift request requires at least one field")
        return self._request("POST", "/api/v1/lift", body)

    def qmlinker_set_waist(
        self,
        *,
        enable: bool | None = None,
        pitch_deg: float | None = None,
    ) -> dict[str, Any]:
        """显式设置腰部使能或 Pitch 目标角度。"""

        body: dict[str, object] = {}
        if enable is not None:
            body["enable"] = enable
        if pitch_deg is not None:
            body["pitch_deg"] = pitch_deg
        if not body:
            raise ValueError("waist request requires at least one field")
        return self._request("POST", "/api/v1/waist", body)

    def qmlinker_set_gripper_position(self, position: int) -> dict[str, Any]:
        """设置 qmlinker 夹爪位置。"""

        return self._request("POST", "/api/v1/gripper", {"position": position})

    def qmlinker_set_right_hand(
        self, positions: list[float] | tuple[float, ...]
    ) -> dict[str, Any]:
        """设置 qmlinker 右手全部执行器位置。"""

        return self._request(
            "POST", "/api/v1/right-hand", {"positions": positions}
        )

    def qmlinker_navigate_to(self, target: str) -> dict[str, Any]:
        """请求 qmlinker AGV 导航到指定目标。"""

        return self._request(
            "POST", "/api/v1/agv/navigate", {"target": target}
        )

    def ar5_set_power(self, side: str, enabled: bool) -> dict[str, Any]:
        """设置 AR5 上下电状态。"""

        return self._request("POST", f"/api/v1/ar5/{side}/power", {"enabled": enabled})

    def ar5_set_operate_mode(self, side: str, automatic: bool) -> dict[str, Any]:
        """设置 AR5 工作模式。"""

        return self._request(
            "POST", f"/api/v1/ar5/{side}/mode", {"automatic": automatic}
        )

    def ar5_stop(self, side: str) -> dict[str, Any]:
        """请求停止 AR5 当前运动。"""

        return self._request("POST", f"/api/v1/ar5/{side}/stop")

    def ar5_move_joints(
        self,
        side: str,
        joint_deg: list[float] | tuple[float, ...],
        speed_mm_s: float,
        zone_mm: float,
    ) -> dict[str, Any]:
        """下发 AR5 关节运动目标。"""

        return self._request(
            "POST",
            f"/api/v1/ar5/{side}/move-joints",
            {"joint_deg": joint_deg, "speed_mm_s": speed_mm_s, "zone_mm": zone_mm},
        )

    def ar5_move_cartesian(
        self,
        side: str,
        xyz_mm: list[float] | tuple[float, ...],
        rpy_deg: list[float] | tuple[float, ...],
        elbow_deg: float,
        speed_mm_s: float,
        zone_mm: float,
    ) -> dict[str, Any]:
        """下发 AR5 笛卡尔运动目标。"""

        return self._request(
            "POST",
            f"/api/v1/ar5/{side}/move-cartesian",
            {
                "xyz_mm": xyz_mm,
                "rpy_deg": rpy_deg,
                "elbow_deg": elbow_deg,
                "speed_mm_s": speed_mm_s,
                "zone_mm": zone_mm,
            },
        )

    def ar5_move_elbow(
        self,
        side: str,
        elbow_deg: float,
        speed_mm_s: float,
        zone_mm: float,
    ) -> dict[str, Any]:
        """下发 AR5 臂角运动目标。"""

        return self._request(
            "POST",
            f"/api/v1/ar5/{side}/move-elbow",
            {"elbow_deg": elbow_deg, "speed_mm_s": speed_mm_s, "zone_mm": zone_mm},
        )

    def recover_estop(self, side: str) -> dict[str, Any]:
        """请求 AR5 急停恢复；仅供现场人员明确手动调用。"""

        return self._request("POST", f"/api/v1/ar5/{side}/recover-estop")

    def get_motion_progress(self, side: str) -> dict[str, Any]:
        """读取当前 NRT 路径进度与碰撞锁存。"""

        return self._request("GET", f"/api/v1/ar5/{side}/motion-progress")

    def clear_servo_alarm(self, side: str) -> dict[str, Any]:
        """调用 xCoreSDK 清除伺服报警。"""

        return self._request("POST", f"/api/v1/ar5/{side}/clear-servo-alarm")

    def set_drag_enabled(self, side: str, enabled: bool) -> dict[str, Any]:
        """设置 AR5 拖动状态；仅供现场人员明确手动调用。"""

        return self._request(
            "POST",
            f"/api/v1/ar5/{side}/drag",
            {"enabled": enabled},
        )

    def start_jog(
        self,
        side: str,
        space: str,
        axis_index: int,
        direction_positive: bool,
        *,
        rate: float,
        step: float,
    ) -> dict[str, Any]:
        """启动 AR5 Jog；仅供现场人员明确手动调用。"""

        return self._request(
            "POST",
            f"/api/v1/ar5/{side}/jog",
            {
                "space": space,
                "axis_index": axis_index,
                "direction_positive": direction_positive,
                "rate": rate,
                "step": step,
            },
        )

    def set_gripper_enabled(self, enabled: bool) -> dict[str, Any]:
        """设置左夹爪使能；仅供现场人员明确手动调用。"""

        return self._request(
            "POST", "/api/v1/gripper/enable", {"enabled": enabled}
        )

    def calibrate_gripper(self) -> dict[str, Any]:
        """请求左夹爪校准；仅供现场人员明确手动调用。"""

        return self._request("POST", "/api/v1/gripper/calibrate")

    def set_right_hand_enabled(self, enabled: bool) -> dict[str, Any]:
        """设置右手使能；仅供现场人员明确手动调用。"""

        return self._request(
            "POST", "/api/v1/right-hand/enable", {"enabled": enabled}
        )

    def set_agv_enabled(self, enabled: bool) -> dict[str, Any]:
        """设置 AGV 使能；仅供现场人员明确手动调用。"""

        return self._request(
            "POST", "/api/v1/agv/enable", {"enabled": enabled}
        )

    def translate_agv(self, speed_mps: float, direction_deg: float) -> dict[str, Any]:
        """请求 AGV 持续平移；停止必须显式调用 ``stop_agv``。"""

        return self._request(
            "POST",
            "/api/v1/agv/translate",
            {"speed_mps": speed_mps, "direction_deg": direction_deg},
        )

    def stop_agv(self) -> dict[str, Any]:
        """停止 AGV 当前导航或实时移动；不等同硬件急停。"""

        return self._request("POST", "/api/v1/agv/stop")

    def subscribe_status(self, interval_s: float = 0.2) -> Iterator[dict[str, Any]]:
        """订阅 RobotControl 的 SSE 设备状态事件流。

        每条事件都是完整的 qmlinker 与 AR5 状态快照。迭代器结束、被调用方关闭或
        网络断开时，底层 HTTP 连接会释放；该方法只发送 GET，不发送控制请求。
        """

        if not 0.05 <= interval_s <= 5.0:
            raise ValueError("interval_s must be between 0.05 and 5.0")
        query = urlencode({"interval_s": f"{interval_s:.3f}"})
        request = Request(
            self._build_url(f"/api/v1/status/stream?{query}"),
            method="GET",
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        )
        with _DIRECT_OPENER.open(
            request,
            timeout=max(self._timeout_s, interval_s * 2.0),
        ) as response:
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line or not data_lines:
                    continue
                payload = json.loads("\n".join(data_lines))
                data_lines.clear()
                if not isinstance(payload, dict):
                    raise RuntimeError("RobotControl SSE data is not a JSON object")
                yield {str(key): value for key, value in payload.items()}

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """发送 HTTP 请求并校验 JSON object 响应。"""

        data = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        request = Request(
            self._build_url(path),
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with _DIRECT_OPENER.open(
                request,
                timeout=self._timeout_s if timeout_s is None else timeout_s,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = None
            if isinstance(error_payload, dict):
                error_text = _format_error_payload(error_payload)
                if error_text is not None:
                    raise RuntimeError(f"HTTP {exc.code}: {error_text}") from exc
            raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("RobotControl 响应不是 JSON object")
        return payload

    def _build_url(self, path: str) -> str:
        """拼接直连或 Gateway 前缀，避免重复拼接 `/api/v1`。"""

        if self._api_prefix and path.startswith("/api/v1/"):
            path = path.removeprefix("/api/v1")
        return self._base_url + self._api_prefix + path


def _normalize_prefix(value: str) -> str:
    """规范统一入口的服务 URL 前缀。"""

    normalized = value.strip().strip("/")
    return f"/{normalized}" if normalized else ""


def _format_error_payload(payload: Mapping[str, object]) -> str | None:
    """将服务端结构化错误转换为可复制的客户端异常文本。"""

    error_value = payload.get("error")
    message_value = payload.get("message")
    if isinstance(error_value, str):
        text = error_value
    elif isinstance(message_value, str):
        text = message_value
    else:
        return None

    context: list[str] = []
    for key in ("stage", "method", "path", "error_type"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            context.append(f"{key}={value}")
    if context:
        return f"{text} ({', '.join(context)})"
    return text
