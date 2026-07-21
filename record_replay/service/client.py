"""RecordReplay HTTP API 同步客户端。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.request import Request, urlopen


class RecordReplayClient:
    """供本机通过 Orin 管理网 IP 访问服务的 HTTP 客户端。"""

    def __init__(self, base_url: str = "http://192.168.1.128:6300", timeout_s: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def get_status(self) -> dict[str, object]:
        return self._request("GET", "/status")

    def get_config(self) -> dict[str, object]:
        return self._request("GET", "/config")

    def get_device_status(self, timeout_s: float = 30.0) -> dict[str, object]:
        """读取双臂、夹爪、头部与升降机构的只读诊断状态。"""

        return self._request("GET", "/device-status", timeout_s=timeout_s)

    def update_config(self, changes: dict[str, float | int]) -> dict[str, object]:
        return self._request("POST", "/config", changes)

    def start(self, enable_agv_navigation: bool = True) -> dict[str, object]:
        """启动一轮回放，并显式指定本轮是否执行 AGV 导航。"""

        return self._request(
            "POST",
            "/start",
            {"enable_agv_navigation": enable_agv_navigation},
        )

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        request_timeout_s = self._timeout_s if timeout_s is None else timeout_s
        with urlopen(request, timeout=request_timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("RecordReplay 响应不是 JSON object")
        return payload
