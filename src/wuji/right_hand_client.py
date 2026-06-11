from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, cast

from qmlinker import QMHand
from qmlinker.grpc_py import hand_pb2

from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.right_hand_specs import RIGHT_HAND_ACTUATOR_SPECS, WujiRightHandActuatorSpec

DEFAULT_RIGHT_HAND_SPEED_RATIO = 0.5
"右手下发默认速度比例，单位 归一化比例。现场 hand_m_example.py 也是该值。"

DEFAULT_RIGHT_HAND_FORCE_LIMIT = 0.5
"右手下发默认力控上限，单位 归一化比例。现场 hand_m_example.py 也是该值。"
# 注意：
# - 现场冒烟时曾把该值写成 0.0，表现为指令看似下发但手无动作。
# - 这里固定使用 0.5，和当前 ROS 侧示例保持一致，避免后续再踩坑。
# - 调用方只需要传归一化位置，速度和力限不再暴露给上层用户。


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

    def stream_right_hand_values(self) -> Iterator[dict[str, float]]:
        """持续读取右手执行器位置。

        Notes
        -----
        该接口直接复用 qmlinker 提供的 `StreamGetHandState` 流接口，
        供 GUI 高频状态刷新线程消费。
        """

        for state in super().stream_get_hand_state(include_tactile=False):
            actuators = state.get("actuators", [])
            values: dict[str, float] = {}
            if not isinstance(actuators, list):
                continue
            for actuator in actuators:
                if not isinstance(actuator, dict):
                    continue
                actuator_id = actuator.get("actuator_id")
                position = actuator.get("position")
                if not isinstance(actuator_id, int | float) or not isinstance(position, int | float):
                    continue
                values[f"right_hand_a{int(actuator_id)}"] = float(position)
            if values:
                yield values

    def _validate_normalized_position(self, target_value: float, *, axis_name: str) -> float:
        """校验右手目标值必须是 0 到 1 的归一化值。

        Parameters
        ----------
        target_value:
            目标位置，必须是有限浮点数，且落在 `0.0` 到 `1.0` 之间。
        axis_name:
            用于错误信息的轴名。

        Returns
        -------
        float
            归一化后的目标值。

        Raises
        ------
        ValueError
            当目标值不是有限数，或超出 `0.0` 到 `1.0` 范围时抛出。
        """

        value = float(target_value)
        if not math.isfinite(value):
            raise ValueError(f"{axis_name} 目标值必须是有限数，当前为 {target_value!r}")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{axis_name} 目标值必须在 0-1 之间，当前为 {value:.6f}")
        return value

    def get_right_hand_actuator_count(self) -> int:
        """读取右手执行器数量。"""

        return len(RIGHT_HAND_ACTUATOR_SPECS)


    def get_right_hand_instance_specs(self) -> tuple[WujiRightHandActuatorSpec, ...]:
        """返回右手固定执行器规格。"""

        return RIGHT_HAND_ACTUATOR_SPECS

    def set_right_hand_axis(self, actuator_id: int, target_value: float) -> bool:
        """设置右手单轴目标值。

        Notes
        -----
        调用方只需要提供归一化目标值 `target_value`。
        速度比例和力控上限由 client 内部固定，不再由上层调用方传入。
        """

        normalized_target_value = self._validate_normalized_position(
            target_value,
            axis_name=f"right_hand_a{int(actuator_id)}",
        )
        current_values = self.get_right_hand_values()
        positions = [
            normalized_target_value
            if spec.actuator_id == int(actuator_id)
            else self._validate_normalized_position(
                self._get_required_current_value(current_values, spec.axis_name),
                axis_name=spec.axis_name,
            )
            for spec in RIGHT_HAND_ACTUATOR_SPECS
        ]
        return bool(self.set_hand_state(positions))

    def _get_required_current_value(self, current_values: dict[str, float], axis_name: str) -> float:
        """读取当前轴值，缺失时直接抛错。

        Parameters
        ----------
        current_values:
            当前右手各轴位置映射，键为轴名，值为 0 到 1 的归一化位置。
        axis_name:
            需要读取的轴名。

        Returns
        -------
        float
            当前轴位置。

        Raises
        ------
        RuntimeError
            当当前状态缺少指定轴时抛出。
        """

        if axis_name not in current_values:
            raise RuntimeError(f"当前右手状态缺少 {axis_name}，拒绝下发。")
        return float(current_values[axis_name])

    def set_hand_state(self, actuator_commands: Sequence[float]) -> bool:
        """按 11 个执行器位置直接下发右手状态。

        Notes
        -----
        调用方只需要传入 11 个归一化位置值。
        当前 `wuyou` 端右手示例链路对 `speed_ratio` 和 `force_limit` 都使用 `0.5`。
        现场曾验证过把 `force_limit` 误设为 `0.0` 时，命令可能已经到达但手部不发生有效动作。
        因此这里固定使用示例一致的默认值，不再回退到 `0.0`。
        """

        normalized_positions = [
            self._validate_normalized_position(position, axis_name=spec.axis_name)
            for spec, position in zip(RIGHT_HAND_ACTUATOR_SPECS, actuator_commands, strict=True)
        ]
        command_frames = [
            {
                "actuator_id": spec.actuator_id,
                "position": float(position),
                "speed_ratio": DEFAULT_RIGHT_HAND_SPEED_RATIO,
                "force_limit": DEFAULT_RIGHT_HAND_FORCE_LIMIT,
            }
            for spec, position in zip(RIGHT_HAND_ACTUATOR_SPECS, normalized_positions, strict=True)
        ]
        return bool(super().set_hand_state(command_frames))


# endregion
