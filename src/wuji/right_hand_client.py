from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from qmlinker import QMHand
from qmlinker.grpc_py import hand_pb2

from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.right_hand_specs import RIGHT_HAND_ACTUATOR_SPECS, WujiRightHandActuatorSpec


# region 右手客户端


class WujiRightHandClient(QMHand):
    """无际右手灵巧手客户端。

    职责边界：
    - 直接继承 `QMHand`，负责右手状态读取、使能控制和状态下发。
    - 不承担左手夹爪语义，左手应由专用 gripper 客户端处理。

    设计思想：
    - 现场已经确认右手是 11 个执行器，因此这里把轴定义固定成项目侧结构体。
    - 仅保留右手语义，不再支持左手作为通用 hand。

    生命周期：
    - 依赖 `WujiQmlinkerBaseClient` 的 channel。
    - 不持有线程或任务队列。

    继承关系：
    - 直接继承 `QMHand`。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        """创建右手客户端。"""

        super().__init__(base_client.channel, cast(str, QMHand.HAND_RIGHT))
        self._base = base_client

    def get_right_hand_values(self) -> dict[str, float]:
        """读取右手执行器位置。"""

        request = hand_pb2.GetHandStateRequest()
        request.hand_id = cast(Any, self.hand_id)
        request.include_tactile = False
        response = self.stub.GetHandState(request, timeout=self._base.config.request_timeout_s)
        values: dict[str, float] = {}
        for actuator in response.actuators:
            axis_name = f"right_hand_a{int(actuator.actuator_id)}"
            values[axis_name] = float(actuator.position)
        return values

    def get_right_hand_actuator_count(self) -> int:
        """读取右手执行器数量。"""

        return len(RIGHT_HAND_ACTUATOR_SPECS)

    def get_right_hand_enable(self) -> bool:
        """读取右手使能状态。"""

        return bool(super().get_enable())

    def get_right_hand_instance_specs(self) -> tuple[WujiRightHandActuatorSpec, ...]:
        """返回右手固定执行器规格。"""

        return RIGHT_HAND_ACTUATOR_SPECS

    def set_right_hand_axis(self, actuator_id: int, target_value: float) -> bool:
        """设置右手单轴目标值。"""

        current_values = self.get_right_hand_values()
        positions = [
            float(target_value) if spec.actuator_id == int(actuator_id) else float(current_values.get(spec.axis_name, spec.minimum))
            for spec in RIGHT_HAND_ACTUATOR_SPECS
        ]
        return bool(self.set_hand_state(positions))

    def set_hand_state(self, actuator_positions: Sequence[float]) -> bool:
        """按 11 个执行器位置直接下发右手状态。"""

        actuator_commands = [
            {
                "actuator_id": spec.actuator_id,
                "position": float(position),
                "speed_ratio": 0.5,
                "force_limit": 0.0,
            }
            for spec, position in zip(RIGHT_HAND_ACTUATOR_SPECS, actuator_positions, strict=True)
        ]
        return bool(super().set_hand_state(actuator_commands))


# endregion
