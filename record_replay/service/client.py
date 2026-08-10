"""RecordReplay HTTP API 同步客户端。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.request import Request, ProxyHandler, build_opener

_DIRECT_OPENER = build_opener(ProxyHandler({}))


class RecordReplayClient:
    """供本机通过 Orin 管理网 IP 访问服务的 HTTP 客户端。"""

    def __init__(
        self,
        base_url: str = "http://192.168.1.128:6300",
        timeout_s: float = 5.0,
        api_prefix: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_prefix = _normalize_prefix(api_prefix)
        self._timeout_s = timeout_s

    def get_status(self) -> dict[str, object]:
        return self._request("GET", "/status")

    def get_plan(self) -> dict[str, object]:
        """读取下一轮动作、CSV 和运动参数的只读预览。"""

        return self._request("GET", "/plan")

    def get_config(self) -> dict[str, object]:
        return self._request("GET", "/config")

    def get_device_status(self, timeout_s: float = 30.0) -> dict[str, object]:
        """读取双臂、夹爪、头部与升降机构的只读诊断状态。"""

        return self._request("GET", "/device-status", timeout_s=timeout_s)

    def update_config(self, changes: Mapping[str, object]) -> dict[str, object]:
        return self._request("POST", "/config", changes)

    def upload_ball_pose_prior(self, payload: Mapping[str, object]) -> dict[str, object]:
        """上传并替换三球 JSON 先验；服务端会先备份旧文件。"""

        return self._request("POST", "/prior/ball-pose", payload)

    def upload_charuco_prior(self, payload: Mapping[str, object]) -> dict[str, object]:
        """上传并替换 ChArUco JSON 先验；服务端会先备份旧文件。"""

        return self._request("POST", "/prior/charuco", payload)

    def start(self, enable_agv_navigation: bool = False) -> dict[str, object]:
        """启动一轮回放，并显式指定本轮是否执行 AGV 导航。"""

        return self._request(
            "POST",
            "/start",
            {"enable_agv_navigation": enable_agv_navigation},
        )

    def stop(self) -> dict[str, object]:
        """请求人工停止并锁存 rapid_stop。"""

        return self._request("POST", "/stop", {})

    def reset(self) -> dict[str, object]:
        """人工处理完成后请求恢复 idle。"""

        return self._request("POST", "/reset", {})

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._build_url(path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        request_timeout_s = self._timeout_s if timeout_s is None else timeout_s
        with _DIRECT_OPENER.open(request, timeout=request_timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("RecordReplay 响应不是 JSON object")
        return payload

    def _build_url(self, path: str) -> str:
        """拼接直连或 Gateway 服务前缀。"""

        return self._base_url + self._api_prefix + path


def _normalize_prefix(value: str) -> str:
    """规范统一入口的服务 URL 前缀。"""

    normalized = value.strip().strip("/")
    return f"/{normalized}" if normalized else ""
