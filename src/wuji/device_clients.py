from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast

from src.arm.wuji_arm_protocol import ArmDeviceName
from src.wuji.agv_client import WujiAgvClient
from src.wuji.arm_client import WujiArmClient
from src.wuji.body_client import WujiBodyClient
from src.wuji.head_client import WujiHeadClient
from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.right_hand_client import WujiRightHandClient
from src.wuji.right_hand_specs import RIGHT_HAND_ACTUATOR_SPECS, WujiRightHandActuatorSpec


# region 设备组合


class WujiQmlinkerClientSet:
    """按设备组合的无际 qmlinker 客户端集合。

    职责边界：
    - 只负责把各设备专用 client 组装成一个统一入口，供 GUI 和测试调用。
    - 不再承载后台线程、订阅缓存、相机流或复杂业务逻辑。

    设计思想：
    - 让每个设备域保持独立实现，组合层只做显式转发和设备选择。
    - 避免把 `qmlinker` 再包成一个新的大而全 facade。

    生命周期：
    - 依赖 `WujiQmlinkerBaseClient` 的 channel 生命周期。
    - `close()` 仅关闭基础连接，不额外管理业务状态。

    继承关系：
    - 不继承业务基类，作为组合器使用。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        """创建设备组合器。"""

        self.base = base_client
        self.arm_left = WujiArmClient(base_client, "left_arm")
        self.arm_right = WujiArmClient(base_client, "right_arm")
        self.body = WujiBodyClient(base_client)
        self.head = WujiHeadClient(base_client)
        self.right_hand = WujiRightHandClient(base_client)
        self.agv = WujiAgvClient(base_client)

    def close(self) -> None:
        """关闭底层连接。"""

        self.arm_left.stop()
        self.arm_right.stop()
        self.base.close()

    def check_ready(self) -> None:
        """检查底层连接是否可用。"""

        self.base.check_ready()

    def get_arm_joint_count(self, device_name: ArmDeviceName) -> int:
        """读取指定机械臂的关节数。"""

        return self.arm_left.get_arm_joint_count(device_name) if device_name == "left_arm" else self.arm_right.get_arm_joint_count(device_name)

    def get_arm_joint_limits(self, device_name: ArmDeviceName):
        """读取指定机械臂的关节限位。"""

        return self.arm_left.get_arm_joint_limits(device_name) if device_name == "left_arm" else self.arm_right.get_arm_joint_limits(device_name)

    def get_joint_states(self, device_name: ArmDeviceName):
        """读取指定机械臂的关节状态。"""

        return self.arm_left.get_joint_states(device_name) if device_name == "left_arm" else self.arm_right.get_joint_states(device_name)

    def set_joint(self, device_name: ArmDeviceName, joint_index: int, target_angle_deg: float) -> bool:
        """设置指定机械臂的单个关节。"""

        return self.arm_left.set_joint(device_name, joint_index, target_angle_deg) if device_name == "left_arm" else self.arm_right.set_joint(device_name, joint_index, target_angle_deg)

    def get_right_hand_instance_specs(self) -> tuple[WujiRightHandActuatorSpec, ...]:
        """读取右手固定轴规格。"""

        return RIGHT_HAND_ACTUATOR_SPECS

    def set_enable(self, device_name: ArmDeviceName, enabled: bool) -> bool:
        """设置指定机械臂使能。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return bool(cast(Any, arm_client).set_enable(enabled))

    def get_enable(self, device_name: ArmDeviceName) -> bool:
        """读取指定机械臂使能。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return bool(cast(Any, arm_client).get_enable())

    def stop_arm(self, device_name: ArmDeviceName) -> None:
        """停止指定机械臂客户端当前发送链路。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        cast(Any, arm_client).stop()

    def get_body_z(self) -> float:
        """读取 body 升降高度。"""

        return self.body.get_body_z()

    def get_body_ry(self) -> float:
        """读取 body 腰部俯仰角。"""

        return self.body.get_body_ry()

    def set_body_z(self, height_mm: float) -> bool:
        """设置 body 升降高度。"""

        return self.body.set_body_z(height_mm)

    def set_body_ry(self, pitch_deg: float) -> bool:
        """设置 body 腰部俯仰角。"""

        return self.body.set_body_ry(pitch_deg)

    def get_head_yaw(self) -> float:
        """读取头部 yaw。"""

        return self.head.get_head_yaw()

    def set_head_yaw(self, yaw_deg: float) -> bool:
        """设置头部 yaw。"""

        return self.head.set_head_yaw(yaw_deg)

    def get_right_hand_values(self):
        """读取右手执行器值。"""

        return self.right_hand.get_right_hand_values()

    def stream_right_hand_values(self) -> Iterator[dict[str, float]]:
        """流式读取右手执行器值。"""

        return self.right_hand.stream_right_hand_values()

    def get_right_hand_actuator_count(self) -> int:
        """读取右手执行器数量。"""

        return self.right_hand.get_right_hand_actuator_count()

    def get_right_hand_enable(self) -> bool:
        """读取右手使能。"""

        return self.right_hand.get_right_hand_enable()

    def set_right_hand_enable(self, enabled: bool) -> bool:
        """设置右手使能。"""

        return bool(self.right_hand.set_enable(enabled))

    def set_right_hand_state(self, actuator_positions: Sequence[float]) -> bool:
        """下发右手状态位置。

        调用方只需要提供 11 个归一化位置值，速度比例和力控上限都由
        `WujiRightHandClient` 内部固定，不再向上层暴露。
        """

        return bool(self.right_hand.set_hand_state(actuator_positions))

    def set_right_hand_axis(self, actuator_id: int, target_value: float) -> bool:
        """设置右手单轴目标值。

        调用方只需要提供归一化目标值，速度比例和力控上限由右手 client 固化。
        """

        return self.right_hand.set_right_hand_axis(actuator_id, target_value)

    def get_module_enable(self, module_name: str) -> bool:
        """读取通用模块使能。"""

        if module_name == "body":
            return bool(cast(Any, self.body).get_enable())
        if module_name == "head":
            return bool(cast(Any, self.head).get_enable())
        if module_name == "left_arm":
            return bool(cast(Any, self.arm_left).get_enable())
        if module_name == "right_arm":
            return bool(cast(Any, self.arm_right).get_enable())
        raise ValueError(f"unsupported module: {module_name}")

    def set_module_enable(self, module_name: str, enabled: bool) -> bool:
        """设置通用模块使能。"""

        if module_name == "body":
            return bool(self.body._lift.set_enable(enabled) and self.body._waist.set_enable(enabled))
        if module_name == "head":
            return bool(self.head.set_enable(enabled))
        if module_name == "left_arm":
            return bool(self.arm_left.set_enable(enabled))
        if module_name == "right_arm":
            return bool(self.arm_right.set_enable(enabled))
        raise ValueError(f"unsupported module: {module_name}")

    def stream_get_joint_states(self, device_name: ArmDeviceName, duration_s: float) -> Iterator[tuple[Any, ...]]:
        """流式读取机械臂状态。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return cast(Any, arm_client).stream_get_joint_states(duration_s)

    def set_joints(self, device_name: ArmDeviceName, joint_angles_deg: Sequence[float], sync_threshold: int = 0) -> bool:
        """下发机械臂关节角度。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return bool(cast(Any, arm_client).set_joints(joint_angles_deg, sync_threshold=sync_threshold))

    def stream_set_joint_states(self, device_name: ArmDeviceName, command_frames: Iterable[Iterable[dict[str, float | int]]]) -> Any:
        """流式下发机械臂关节目标。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return cast(Any, arm_client).stream_set_joint_states(command_frames)

    def fk(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> Any:
        """计算正运动学。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return cast(Any, arm_client).fk(joint_angles_rad)

    def fk_fast(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> Any:
        """快速计算正运动学。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return cast(Any, arm_client).fk_fast(joint_angles_rad)

    def ik(self, device_name: ArmDeviceName, target_pose: Any, reference_joint_angles_rad: Iterable[float]) -> Any:
        """计算逆运动学。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return cast(Any, arm_client).ik(target_pose, reference_joint_angles_rad)

    def current_fk_fast(self, device_name: ArmDeviceName) -> Any:
        """计算当前位姿的快速正运动学。"""

        arm_client = self.arm_left if device_name == "left_arm" else self.arm_right
        return cast(Any, arm_client).current_fk_fast(device_name)

    def get_agv_status_values(self) -> dict[str, float]:
        """读取 AGV 状态。"""

        status = self.agv.get_base_status()
        return {
            "agv_x": float(getattr(status, "x", 0.0)),
            "agv_y": float(getattr(status, "y", 0.0)),
            "agv_yaw": float(getattr(status, "yaw", 0.0)),
            "agv_battery": float(getattr(status, "battery", 0.0)),
        }

    def get_agv_enable(self) -> bool:
        """读取 AGV 使能。"""

        return bool(self.agv.get_enable())

    def set_agv_enable(self, enabled: bool) -> bool:
        """设置 AGV 使能。"""

        return bool(self.agv.set_enable(enabled))

    def move_agv_translate(self, speed_mps: float, direction_deg: int, distance_m: float) -> bool:
        """按距离平移 AGV。"""

        return bool(self.agv.translate_with_distance_sync(speed_mps, direction_deg, distance_m))

    def move_agv_real_time_translate(self, speed_mps: float, direction_deg: int) -> bool:
        """按方向实时平移 AGV。"""

        return bool(self.agv.real_time_translate(speed_mps, direction_deg))

    def move_agv_rotate(self, angle_deg: float, direction, speed_ratio: float) -> bool:
        """按角度旋转 AGV。"""

        return bool(self.agv.rotate_with_angle_sync(angle_deg, direction, speed_ratio))

    def agv_navigate_to(self, target_name: str) -> Any:
        """发送 AGV 导航目标。"""

        return self.agv.navigate_to(target_name)

    def agv_navigate_to_charge(self) -> Any:
        """发送 AGV 去充电导航目标。"""

        return self.agv.navigate_to("charge")

    def stop_agv(self) -> bool:
        """停止 AGV。"""

        return bool(self.agv.stop())


# endregion


__all__ = ["WujiQmlinkerClientSet"]
