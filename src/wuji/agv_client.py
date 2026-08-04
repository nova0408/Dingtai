from __future__ import annotations

from dataclasses import dataclass
import time

import grpc
from google.protobuf import empty_pb2
from qmlinker import QMMoveBase

from . import navigation_map_pb2_grpc

# region agv 客户端


@dataclass(frozen=True, slots=True)
class WujiAgvNavigationTarget:
    """Woosh 地图中的一个可导航预设点。"""

    name: str
    id: int
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class WujiAgvNavigationMap:
    """Woosh AGV 当前地图及其可导航预设点。"""

    name: str
    id: int
    resolution: float
    targets: tuple[WujiAgvNavigationTarget, ...]


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
        self._navigation_map_stub = navigation_map_pb2_grpc.NavigationMapServiceStub(
            channel
        )

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

    def get_navigation_map(self) -> WujiAgvNavigationMap:
        """读取当前 Woosh 地图及其可导航目标点。

        Returns
        -------
        WujiAgvNavigationMap
            地图名称、ID、原始分辨率和目标点坐标。坐标单位为 m，航向单位为 rad。

        Raises
        ------
        RuntimeError
            gRPC 查询失败，或远端地图服务返回失败状态。
        """

        response = self._navigation_map_stub.GetNavigationTargets(
            empty_pb2.Empty(),
            timeout=self._request_timeout_s,
        )
        if not response.success:
            message = response.message or "navigation map service returned failure"
            raise RuntimeError(message)
        targets = tuple(
            WujiAgvNavigationTarget(
                name=target.name,
                id=int(target.id),
                x_m=float(target.x_m),
                y_m=float(target.y_m),
                yaw_rad=float(target.yaw_rad),
            )
            for target in response.targets
        )
        return WujiAgvNavigationMap(
            name=response.map_name,
            id=int(response.map_id),
            resolution=float(response.resolution),
            targets=targets,
        )

    def get_navigation_targets(self) -> list[str]:
        """返回当前地图中可用于导航请求的目标点名称。"""

        return [target.name for target in self.get_navigation_map().targets]


# endregion
