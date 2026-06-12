from __future__ import annotations

import time

import grpc
from google.protobuf import empty_pb2
from qmlinker import QMMoveBase

# region agv 客户端


class WujiAgvClient(QMMoveBase):
    """无际 AGV 客户端。

    职责边界：
    - 直接继承 `QMMoveBase`，负责底盘状态读取、使能与移动控制。
    - 不负责 GUI 的卡片展示逻辑，也不负责电量、里程等字段格式化。

    设计思想：
    - 底盘是独立设备域，项目侧只需要把 SDK 绑定到统一 channel。
    - 具体的控制按钮、展示格式和接口适配留给上层 GUI 或门面类处理。

    生命周期：
    - 依赖外部传入的底盘 qmlinker channel。
    - 不持有后台 worker。

    继承关系：
    - 直接继承 `QMMoveBase`。
    """

    def __init__(self, channel: object, request_timeout_s: float = 3.0) -> None:
        """创建 AGV 客户端。

        Parameters
        ----------
        channel:
            AGV 底盘使用的 qmlinker channel。
        request_timeout_s:
            单次 unary 请求超时时间，单位 s。
        """

        super().__init__(channel)
        self._request_timeout_s = float(request_timeout_s)

    def get_runtime_info(self) -> dict[str, object]:
        """读取 AGV 运行时信息。

        Returns
        -------
        dict[str, object]
            返回 AGV 当前导航状态与基础位姿、电量信息。
            `agv_navi_status` 为字符串，其余 `agv_*` 字段为浮点数。
        """

        last_error: Exception | None = None
        for _ in range(10):
            try:
                response = self.stub.GetBaseStatus(
                    empty_pb2.Empty(),
                    timeout=self._request_timeout_s,
                )
                return {
                    "agv_navi_status": str(getattr(response, "navi_status", "")),
                    "agv_x": float(getattr(response, "x", 0.0)),
                    "agv_y": float(getattr(response, "y", 0.0)),
                    "agv_yaw": float(getattr(response, "yaw", 0.0)),
                    "agv_battery": float(getattr(response, "battery", 0.0)),
                }
            except grpc.RpcError as exc:
                last_error = exc
                if exc.code() not in {
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                }:
                    raise
                time.sleep(0.5)
        raise RuntimeError("qmlinker get agv runtime info failed") from last_error

    def try_get_enable(self) -> bool | None:
        """尝试读取 AGV 使能状态。

        Returns
        -------
        bool | None
            成功时返回使能状态。
            若当前链路下 `GetEnabled` 不可用，则返回 `None`，由上层决定是否显示未知状态。
        """

        try:
            response = self.stub.GetEnabled(
                empty_pb2.Empty(),
                timeout=self._request_timeout_s,
            )
        except grpc.RpcError:
            return None
        return bool(response.status.success and response.current_state)

    @staticmethod
    def get_navigation_targets() -> list[str]:
        """返回 AGV 可导航目标点列表。

        Notes
        -----
        当前 qmlinker `BaseService` 未提供目标点清单查询 RPC，
        因此这里显式返回空列表，避免 GUI 猜测默认目标点。
        """

        return []


# endregion
