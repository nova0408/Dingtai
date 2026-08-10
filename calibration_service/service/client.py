"""手眼标定与先验记录服务的 HTTP 客户端。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.request import ProxyHandler, Request, build_opener


class CalibrationServiceClient:
    """调用 Gateway 或 Orin 本机标定服务 HTTP API。

    该客户端只描述拍摄和计算接口，不提供任何设备控制方法。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6600",
        api_prefix: str = "/api/v1",
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_prefix = api_prefix.strip("/")
        self._timeout_s = timeout_s
        self._opener = build_opener(ProxyHandler({}))

    def get_status(self) -> dict[str, object]:
        """读取服务任务状态，不读取设备。"""

        return self._request("GET", "/status")

    def get_hand_eye_config(self) -> dict[str, object]:
        """读取手眼 ChArUco 默认参数和当前 OpenCV 可用字典。"""

        return self._request("GET", "/hand-eye/config")

    def update_hand_eye_config(self, payload: Mapping[str, object]) -> dict[str, object]:
        """部分更新手眼 ChArUco 默认参数。"""

        return self._request("PATCH", "/hand-eye/config", payload)

    def get_hand_eye_result(self) -> dict[str, object]:
        """读取左手眼在手上 `T_tool_cam` 结果。"""

        return self._request("GET", "/results/hand-eye")

    def get_head_eye_result(self, arm_side: str = "left") -> dict[str, object]:
        """读取头部眼在手外 `T_base_camera` 结果。"""

        return self._request("GET", f"/results/head-eye/{arm_side}")

    def get_head_prior(self) -> dict[str, object]:
        """读取头部 ChArUco 先验。"""

        return self._request("GET", "/results/prior/head")

    def get_hand_prior(self) -> dict[str, object]:
        """读取手部三球先验。"""

        return self._request("GET", "/results/prior/hand")

    def start_calibration(self, calibration_kind: str, arm_side: str = "left") -> dict[str, object]:
        """提示并开始标定采样会话。"""

        return self._request(
            "POST",
            "/start",
            {"calibration_kind": calibration_kind, "arm_side": arm_side},
        )

    def end_calibration(self, calibration_kind: str, arm_side: str = "left") -> dict[str, object]:
        """提示并结束标定采样会话。"""

        return self._request(
            "POST",
            "/end",
            {"calibration_kind": calibration_kind, "arm_side": arm_side},
        )

    def cancel(self) -> dict[str, object]:
        """取消当前采样或待确认结果，不替换正式文件。"""

        return self._request("POST", "/cancel", {})

    def confirm_replacement(
        self,
        replacement_id: str,
        confirmed: bool = True,
    ) -> dict[str, object]:
        """二次确认并替换正式结果；旧文件会保留为时间戳文件。"""

        return self._request(
            "POST",
            "/replacements/confirm",
            {"replacement_id": replacement_id, "confirmed": confirmed},
        )

    def record_head_prior(self) -> dict[str, object]:
        """触发头部 ChArUco 先验拍摄和计算。"""

        return self._request("POST", "/prior/head", {})

    def record_hand_prior(self, arm_side: str = "left") -> dict[str, object]:
        """触发手部三球先验拍摄和计算。"""

        return self._request("POST", "/prior/hand", {"arm_side": arm_side})

    def capture_hand_eye_sample(self, arm_side: str = "left") -> dict[str, object]:
        """读取当前姿态并拍摄一组手眼样本。"""

        return self._request("POST", "/hand-eye/sample", {"arm_side": arm_side})

    def capture_head_eye_sample(self, arm_side: str = "left") -> dict[str, object]:
        """读取当前姿态并拍摄头部眼在手外样本。"""

        return self._request("POST", "/head-eye/sample", {"arm_side": arm_side})

    def solve_hand_eye(self, payload: Mapping[str, object] | None = None) -> dict[str, object]:
        """计算已采集或请求中提供的手眼样本。"""

        return self._request("POST", "/hand-eye/solve", {} if payload is None else payload)

    def solve_head_eye(
        self,
        arm_side: str = "left",
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """计算头部眼在手外 `T_base_camera`。"""

        body = {} if payload is None else dict(payload)
        body["arm_side"] = arm_side
        return self._request("POST", "/head-eye/solve", body)

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        prefix = f"/{self._api_prefix}" if self._api_prefix else ""
        url = f"{self._base_url}{prefix}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(request, timeout=self._timeout_s) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("标定服务响应根节点必须是 object")
        return value
