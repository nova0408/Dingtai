from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from src.wuji.client_base import WujiQmlinkerBaseClient


# region 数据结构


@dataclass(frozen=True, slots=True)
class DahuanGripperInfo:
    """大寰夹爪当前状态快照。"""

    timestamp_ms: int
    "状态时间戳，单位 ms。"

    position: int
    "当前位置，范围通常为 `0-1000`。"

    speed: int
    "当前速度参数。"

    force: int
    "当前力参数。"

    grip_state: int
    "当前夹持状态枚举值。"

    enabled: bool | None = None
    "夹爪使能状态。"

    online: bool = True
    "夹爪在线状态。"

    calibrated: bool = False
    "夹爪是否完成校准。"


# endregion


# region 主入口


class DahuanGripperClient:
    """大寰夹爪远程适配器。

    职责边界：
    - 通过 `WujiQmlinkerBaseClient` 共享的 Orin SSH 上下文执行夹爪脚本。
    - 不创建 GUI 控件，不维护长连接线程。

    设计思想：
    - 夹爪在现场仍然通过 Orin 侧 `qmlinker` Python 环境运行，客户端只负责把脚本收口成
      项目侧稳定方法。
    - 只暴露新 SDK 直接支持的使能、校准、速度、力、位置与状态接口。

    生命周期：
    - 依赖外部传入的 `WujiQmlinkerBaseClient`。
    - 不持有后台线程。

    继承关系：
    - 不继承业务基类，避免伪装成 `QMGripper` 子类。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        self._base = base_client

    def get_gripper_info(self) -> DahuanGripperInfo:
        """读取大寰夹爪状态。"""

        payload = self._base.run_orin_python_json(self._status_script())
        return DahuanGripperInfo(
            timestamp_ms=self._int_field(payload, "timestamp_ms"),
            position=self._int_field(payload, "position"),
            speed=self._int_field(payload, "speed"),
            force=self._int_field(payload, "force"),
            grip_state=self._int_field(payload, "grip_state"),
            enabled=self._optional_bool_field(payload, "enabled"),
            online=self._bool_field(payload, "online"),
            calibrated=self._bool_field(payload, "calibrated"),
        )

    def probe(self) -> DahuanGripperInfo:
        """探测当前夹爪链路可连通性。"""

        return self.get_gripper_info()

    def set_enable(self, enabled: bool) -> None:
        """设置夹爪使能状态。"""

        self._base.run_orin_python_json(self._simple_setter_script("set_enable", int(bool(enabled))))

    def calibrate(self) -> bool:
        """执行夹爪校准。"""

        payload = self._base.run_orin_python_json(self._calibrate_script())
        return bool(payload.get("calibrated", False))

    def set_force(self, force: int) -> None:
        """设置夹爪力。"""

        self._base.run_orin_python_json(self._simple_setter_script("set_force", int(force)))

    def set_speed(self, speed: int) -> None:
        """设置夹爪速度。"""

        self._base.run_orin_python_json(self._simple_setter_script("set_speed", int(speed)))

    def set_pos(self, position: int) -> None:
        """设置夹爪位置。"""

        self._base.run_orin_python_json(self._simple_setter_script("set_pos", int(position)))

    def move_gripper_position(self, position: int) -> None:
        """按原始位置设置夹爪位置。"""

        self.set_pos(position)

    def move_gripper_ratio(self, ratio: float) -> None:
        """按归一化比例设置夹爪位置。"""

        self.set_pos(int(round(min(max(float(ratio), 0.0), 1.0) * 1000.0)))

    def _status_script(self) -> str:
        return f"""\nimport json\nfrom qmlinker import create_channel, QMGripper\nchannel = create_channel({f'{self._base.robot_network_config.base_control_ip}:{self._base.robot_network_config.gripper_port}'!r})\ngripper = QMGripper(channel)\nstatus = gripper.get_status()\nprint(json.dumps({{\n    'timestamp_ms': int(getattr(status, 'timestamp_ms', 0)),\n    'position': int(getattr(status, 'position', 0)),\n    'speed': int(getattr(status, 'speed', 0)),\n    'force': int(getattr(status, 'force', 0)),\n    'grip_state': int(getattr(status, 'state', 0)),\n    'enabled': getattr(status, 'enable', None),\n    'online': bool(getattr(status, 'online', False)),\n    'calibrated': bool(getattr(status, 'calibrated', False)),\n}}, ensure_ascii=False))\n"""

    def _simple_setter_script(self, method_name: str, value: int) -> str:
        return f"""\nimport json\nfrom qmlinker import create_channel, QMGripper\nchannel = create_channel({f'{self._base.robot_network_config.base_control_ip}:{self._base.robot_network_config.gripper_port}'!r})\ngripper = QMGripper(channel)\ngetattr(gripper, {method_name!r})({int(value)})\nprint(json.dumps({{'ok': True, 'method': {method_name!r}, 'value': {int(value)}}}, ensure_ascii=False))\n"""

    def _calibrate_script(self) -> str:
        return f"""\nimport json\nfrom qmlinker import create_channel, QMGripper\nchannel = create_channel({f'{self._base.robot_network_config.base_control_ip}:{self._base.robot_network_config.gripper_port}'!r})\ngripper = QMGripper(channel)\ncalibrated = bool(gripper.calibrate())\nprint(json.dumps({{'ok': True, 'calibrated': calibrated}}, ensure_ascii=False))\n"""

    @staticmethod
    def _int_field(payload: dict[str, object], key: str) -> int:
        value: object = payload.get(key, 0)
        return int(cast(int, value))

    @staticmethod
    def _bool_field(payload: dict[str, object], key: str) -> bool:
        value: object = payload.get(key, False)
        return bool(cast(bool, value))

    @staticmethod
    def _optional_bool_field(payload: dict[str, object], key: str) -> bool | None:
        value = payload.get(key)
        if value is None:
            return None
        return bool(cast(bool, value))


# endregion
