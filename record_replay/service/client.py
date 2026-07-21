"""RecordReplay HTTP API 同步客户端。"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


class RecordReplayClient:
    """供本机通过 SSH 转发访问 Orin 服务的 HTTP 客户端。"""

    def __init__(self, base_url: str = "http://127.0.0.1:6300", timeout_s: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def get_status(self) -> dict[str, object]:
        return self._request("GET", "/status")

    def get_config(self) -> dict[str, object]:
        return self._request("GET", "/config")

    def update_config(self, changes: dict[str, float | int]) -> dict[str, object]:
        return self._request("POST", "/config", changes)

    def start(self) -> dict[str, object]:
        return self._request("POST", "/start")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, float | int] | None = None,
    ) -> dict[str, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self._timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("RecordReplay 响应不是 JSON object")
        return payload

