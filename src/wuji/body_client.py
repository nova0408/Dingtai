from __future__ import annotations

from typing import Any, cast

from google.protobuf import empty_pb2
from qmlinker import QMLift, QMWaist
from qmlinker.grpc_py import lift_pb2, waist_pb2

from src.wuji.client_base import WujiQmlinkerBaseClient


# region body 客户端


class WujiBodyClient:
    """无际 body 客户端。

    职责边界：
    - 直接封装 `QMLift` 和 `QMWaist`，负责升降与腰部俯仰的读写。
    - 不负责 GUI 状态同步、订阅调度或其他设备的使能逻辑。

    设计思想：
    - body 在项目语义里由两个独立执行器组成，因此这里显式持有两个 SDK 对象。
    - 只把项目侧常用的毫米/角度接口整理出来，避免上层重复处理 proto 细节。

    生命周期：
    - 依赖 `WujiQmlinkerBaseClient` 的 channel 生命周期。
    - 不持有线程，调用完成即可释放上下文。

    继承关系：
    - 不继承业务基类，避免把 lift 和 waist 强行捆成动态多态对象。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        """创建 body 客户端。"""

        self._base = base_client
        self._lift = QMLift(self._base.channel)
        self._waist = QMWaist(self._base.channel)

    def get_body_z(self) -> float:
        """读取升降高度，单位 mm。"""

        response = self._lift.stub.GetCurrentLiftPhysicalHeight(empty_pb2.Empty(), timeout=self._base.config.request_timeout_s)
        return float(response.current_height_mm)

    def get_enable(self) -> bool:
        """读取 body 使能状态。"""

        lift_enable = bool(cast(Any, self._lift).get_enable())
        waist_enable = bool(cast(Any, self._waist).get_enable())
        return bool(lift_enable and waist_enable)

    def set_body_z(self, height_mm: float) -> bool:
        """设置升降高度，单位 mm。"""

        request = lift_pb2.SetLiftPhysicalHeightRequest(height_mm=int(round(height_mm)))
        response = self._lift.stub.SetLiftPhysicalHeight(request, timeout=self._base.config.request_timeout_s)
        return bool(response.status.success)

    def get_body_ry(self) -> float:
        """读取腰部俯仰角，单位 deg。"""

        response = self._waist.stub.GetCurrentPitch(empty_pb2.Empty(), timeout=self._base.config.request_timeout_s)
        return float(response.current_pitch_deg)

    def set_body_ry(self, pitch_deg: float) -> bool:
        """设置腰部俯仰角，单位 deg。"""

        request = waist_pb2.SetWaistPitchRequest(pitch_angle_deg=float(pitch_deg))
        response = self._waist.stub.SetPitchAngle(request, timeout=self._base.config.request_timeout_s)
        return bool(response.status.success)


# endregion
