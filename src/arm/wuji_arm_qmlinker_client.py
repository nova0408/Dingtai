from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from qmlinker import QMArm, QMHand, QMMoveBase, create_channel
from qmlinker.grpc_py import arm_pb2, common_pb2, head_pb2, head_pb2_grpc
from qmlinker.grpc_py import lift_pb2, lift_pb2_grpc
from qmlinker.grpc_py import waist_pb2, waist_pb2_grpc
from google.protobuf import empty_pb2

from src.arm.wuji_arm_protocol import ArmDeviceName, WujiArmQmlinkerConfig, WujiModuleName
from src.hand import HandDeviceName

# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiArmJointState:
    """无际机械臂单关节状态。

    职责边界：
    - 只保存 qmlinker 返回的单关节遥测数据。
    - 不负责单位换算、GUI 展示或运动控制。

    设计思想：
    - 使用不可变 dataclass，便于跨线程从工作线程传回 GUI 线程。
    - 字段与 qmlinker 的 `arm_info.joint_states` 保持一致。

    生命周期：
    - 由一次状态读取构造，不持有网络连接或 qmlinker 对象。

    继承关系：
    - 不继承业务基类，作为协议结果数据使用。
    """

    joint_id: int
    "关节 ID，范围 1 到 6。"

    angle_deg: float
    "当前关节角度，单位 deg。"

    current_a: float
    "当前关节电流，单位 A。"

    power_w: float
    "当前关节功率，单位 W。"


# endregion


# region 主入口


class WujiArmQmlinkerClient:
    """无际机械臂 qmlinker 本机客户端。

    职责边界：
    - 负责在 DingTai 环境中通过 qmlinker SDK 访问基础控制工控机 ArmService。
    - 不执行远端 Python，不修改远端环境。
    - 不持有 GUI 控件，不发 Qt 信号。

    设计思想：
    - 直接使用接口文档提供的 `qmlinker-1.0.8-py3-none-any.whl`，避免重复实现 SDK 内部逻辑。
    - 客户端持有一个 channel 和左右臂 QMArm 实例，复用 SDK 内部状态更新线程。

    生命周期：
    - 随 GUI 后端创建和关闭。
    - `close` 只停止 SDK 内部线程，不负责关闭 GUI 会话。

    继承关系：
    - 不继承业务基类，作为硬件协议适配器使用。
    """

    def __init__(self, config: WujiArmQmlinkerConfig | None = None) -> None:
        """初始化 qmlinker 机械臂客户端。

        Parameters
        ----------
        config:
            qmlinker 连接配置，为 `None` 时使用默认配置。
        """

        self._config = WujiArmQmlinkerConfig() if config is None else config
        self._channel = create_channel(self._config.target())
        self._arms: dict[ArmDeviceName, Any] = {}
        self._hands: dict[HandDeviceName, Any] = {}
        self._default_channel = self._channel["DEFAULT"] if isinstance(self._channel, dict) else self._channel
        self._move_base = QMMoveBase(self._channel)
        self._waist_stub = waist_pb2_grpc.WaistServiceStub(self._default_channel)
        self._lift_stub = lift_pb2_grpc.LiftServiceStub(self._default_channel)
        self._head_stub = head_pb2_grpc.HeadServiceStub(self._default_channel)

    def close(self) -> None:
        """停止 qmlinker 内部状态线程。"""

        for arm in self._arms.values():
            if hasattr(arm, "running"):
                arm.running = False
            thread = getattr(arm, "thread_joint_states", None)
            if thread is not None:
                thread.join(timeout=0.5)
        self._arms.clear()
        self._hands.clear()

    def check_ready(self) -> None:
        """检查 qmlinker gRPC 通道是否可创建并进入 ready。

        Raises
        ------
        TimeoutError
            在配置时间内 gRPC 通道无法 ready。
        """

        import grpc

        grpc.channel_ready_future(self._default_channel).result(timeout=self._config.request_timeout_s)

    def get_module_enable(self, module_name: WujiModuleName) -> bool:
        """读取整机模块使能状态。

        Parameters
        ----------
        module_name:
            模块名，支持 `base`、`body`、`head`、`left_arm` 与 `right_arm`。

        Returns
        -------
        bool
            `True` 表示模块已使能。
        """

        if module_name in {"left_arm", "right_arm"}:
            return self.get_enable(self._arm_device_name(module_name))
        if module_name == "body":
            return self._get_stub_enable(self._lift_stub) and self._get_stub_enable(self._waist_stub)
        response = self._module_stub(module_name).GetEnabled(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success and response.current_state == common_pb2.MODULE_ENABLED)

    def set_module_enable(self, module_name: WujiModuleName, enabled: bool) -> bool:
        """设置整机模块使能状态。"""

        if module_name in {"left_arm", "right_arm"}:
            return self.set_enable(self._arm_device_name(module_name), enabled)
        if module_name == "body":
            return self._set_stub_enable(self._lift_stub, enabled) and self._set_stub_enable(
                self._waist_stub,
                enabled,
            )
        request = common_pb2.ModuleEnableRequest(enable=bool(enabled))
        response = self._module_stub(module_name).SetEnabled(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_enable(self, device_name: ArmDeviceName) -> bool:
        """读取机械臂使能状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。

        Returns
        -------
        bool
            `True` 表示当前状态为已使能。
        """

        return bool(self._arm(device_name).get_enable())

    def set_enable(self, device_name: ArmDeviceName, enabled: bool) -> bool:
        """设置机械臂使能状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        enabled:
            目标使能状态。

        Returns
        -------
        bool
            `True` 表示设置请求成功。
        """

        return bool(self._arm(device_name).set_enable(enabled))

    def get_joint_states(self, device_name: ArmDeviceName) -> tuple[WujiArmJointState, ...]:
        """读取指定机械臂的一帧关节状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。

        Returns
        -------
        tuple[WujiArmJointState, ...]
            关节状态序列，通常长度为 6，角度单位为 deg。
        """

        arm = self._arm(device_name)
        arm_info = arm.get_arm_info(timeout=self._config.request_timeout_s)
        if arm_info is None or not arm_info.initialized:
            raise TimeoutError(f"qmlinker joint states not ready: {device_name}")
        return tuple(
            WujiArmJointState(
                joint_id=index,
                angle_deg=float(joint.angle_deg),
                current_a=float(joint.current_a),
                power_w=float(joint.power_w),
            )
            for index, joint in enumerate(arm_info.joint_states, start=1)
        )

    def set_joint(
        self,
        device_name: ArmDeviceName,
        joint_index: int,
        target_angle_deg: float,
    ) -> bool:
        """设置单个机械臂关节目标角度。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        joint_index:
            关节索引，范围 1 到 6。
        target_angle_deg:
            目标关节角度，单位 deg。

        Returns
        -------
        bool
            `True` 表示 qmlinker 接受该关节命令。
        """

        joints = self.get_joint_states(device_name)
        commands = [
            {
                "joint_id": joint.joint_id,
                "target_angle_deg": target_angle_deg if joint.joint_id == joint_index else joint.angle_deg,
                "speed_ratio": self._config.default_speed_ratio,
            }
            for joint in joints
        ]
        return bool(self._arm(device_name).set_joints(commands))

    def set_joints(
        self,
        device_name: ArmDeviceName,
        joint_commands: Iterable[dict[str, float | int]],
        sync_threshold: float = 0.0,
    ) -> bool:
        """批量设置机械臂关节目标角度。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        joint_commands:
            关节命令序列，每个元素包含 `joint_id`、`target_angle_deg` 和 `speed_ratio`。
        sync_threshold:
            qmlinker 同步阈值，单位 deg。为 0 时不等待同步。

        Returns
        -------
        bool
            `True` 表示 qmlinker 接受该批关节命令。
        """

        commands = [dict(command) for command in joint_commands]
        return bool(self._arm(device_name).set_joints(commands, sync_threshold=sync_threshold))

    def stream_get_joint_states(
        self,
        device_name: ArmDeviceName,
        duration_s: float,
    ) -> Iterator[tuple[WujiArmJointState, ...]]:
        """在指定时长内流式读取机械臂关节状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        duration_s:
            读取持续时间，单位 s。

        Yields
        ------
        tuple[WujiArmJointState, ...]
            每帧关节状态，角度单位为 deg。
        """

        deadline = time.time() + max(0.0, duration_s)
        request = arm_pb2.GetJointStatesRequest()
        request.arm_type = self._arm_pb_type(device_name)
        for response in self._arm(device_name).stub.StreamGetJointStates(request):
            if time.time() > deadline:
                break
            yield tuple(
                WujiArmJointState(
                    joint_id=int(joint.joint_id),
                    angle_deg=float(joint.angle_deg),
                    current_a=float(joint.current_a),
                    power_w=float(joint.power_w),
                )
                for joint in response.joints
            )

    def stream_set_joint_states(
        self,
        device_name: ArmDeviceName,
        command_frames: Iterable[Iterable[dict[str, float | int]]],
    ) -> bool:
        """流式设置机械臂关节角度。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        command_frames:
            多帧关节命令。每帧为若干 `joint_id`、`target_angle_deg`、`speed_ratio` 字典。

        Returns
        -------
        bool
            `True` 表示流式发送结束。
        """

        frames = [[dict(command) for command in frame] for frame in command_frames]
        result = self._arm(device_name).stream_set_joint_states(frames)
        return result is None or bool(result)

    def fk(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> list[Any]:
        """执行 qmlinker 正向运动学。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        joint_angles_rad:
            6 轴关节角，单位 rad。

        Returns
        -------
        list[Any]
            qmlinker FK 返回的各关节变换矩阵列表。
        """

        return list(self._arm(device_name).fkik.fk(list(joint_angles_rad)))

    def fk_fast(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> Any:
        """执行 qmlinker 快速正向运动学，返回末端位姿矩阵。"""

        return self._arm(device_name).fkik.fk_fast(list(joint_angles_rad))

    def ik(
        self,
        device_name: ArmDeviceName,
        target_pose: Any,
        reference_joint_angles_rad: Iterable[float],
    ) -> tuple[float, ...]:
        """执行 qmlinker 逆向运动学。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        target_pose:
            目标 4x4 齐次变换矩阵。
        reference_joint_angles_rad:
            参考 6 轴关节角，单位 rad。

        Returns
        -------
        tuple[float, ...]
            逆解关节角，单位 rad。无解时返回空元组。
        """

        result = self._arm(device_name).fkik.ik(target_pose, list(reference_joint_angles_rad))
        return tuple(float(value) for value in result)

    def current_fk_fast(self, device_name: ArmDeviceName) -> Any:
        """读取当前关节角并计算末端位姿矩阵。"""

        joints = self.get_joint_states(device_name)
        joint_angles_rad = [float(np.deg2rad(joint.angle_deg)) for joint in joints]
        return self.fk_fast(device_name, joint_angles_rad)

    def get_body_z(self) -> float:
        """读取身体 z 轴升降高度。

        Returns
        -------
        float
            当前升降高度，单位 mm。
        """

        response = self._lift_stub.GetCurrentLiftPhysicalHeight(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return float(response.current_height_mm)

    def set_body_z(self, height_mm: float) -> bool:
        """设置身体 z 轴升降高度，单位 mm。"""

        request = lift_pb2.SetLiftPhysicalHeightRequest(height_mm=int(round(height_mm)))
        response = self._lift_stub.SetLiftPhysicalHeight(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_body_ry(self) -> float:
        """读取身体 Ry 俯仰角，单位 deg。"""

        response = self._waist_stub.GetCurrentPitch(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return float(response.current_pitch_deg)

    def set_body_ry(self, pitch_deg: float) -> bool:
        """设置身体 Ry 俯仰角，单位 deg。"""

        request = waist_pb2.SetWaistPitchRequest(pitch_angle_deg=float(pitch_deg))
        response = self._waist_stub.SetPitchAngle(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_head_yaw(self) -> float:
        """读取头部旋转轴角度，单位 deg。"""

        response = self._head_stub.GetHeadYaw(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return float(response.current_yaw_deg)

    def set_head_yaw(self, yaw_deg: float) -> bool:
        """设置头部旋转轴角度，单位 deg。"""

        request = head_pb2.SetHeadYawRequest(yaw_angle_deg=float(yaw_deg))
        response = self._head_stub.SetHeadYaw(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_agv_status_values(self) -> dict[str, float]:
        """读取 AGV 底盘基础状态值。"""

        status = self._move_base.get_base_status()
        if not isinstance(status, dict):
            raise RuntimeError("qmlinker get base status failed")
        return {
            "agv_x": float(status.get("x", 0.0)),
            "agv_y": float(status.get("y", 0.0)),
            "agv_yaw": float(status.get("yaw", 0.0)),
            "agv_battery": float(status.get("battery", 0.0)),
        }

    def get_hand_values(self, device_name: HandDeviceName) -> dict[str, float]:
        """读取指定手部执行器位置。"""

        state = self._hand(device_name).get_hand_state(include_tactile=False)
        if not isinstance(state, dict) or not isinstance(state.get("actuators"), list):
            raise RuntimeError(f"qmlinker get hand state failed: {device_name}")
        values: dict[str, float] = {}
        for actuator in state["actuators"]:
            if not isinstance(actuator, dict):
                continue
            actuator_id = int(actuator.get("actuator_id", -1))
            if actuator_id < 0:
                continue
            values[f"{device_name}_a{actuator_id}"] = float(actuator.get("position", 0.0))
        return values

    def _arm(self, device_name: ArmDeviceName) -> Any:
        """返回指定机械臂的 qmlinker QMArm 实例。"""

        arm = self._arms.get(device_name)
        if arm is not None:
            return arm
        arm_type: Any = QMArm.ARM_LEFT if device_name == "left_arm" else QMArm.ARM_RIGHT
        arm = QMArm(self._channel, arm_type)
        self._arms[device_name] = arm
        return arm

    def _hand(self, device_name: HandDeviceName) -> Any:
        """返回指定手部的 qmlinker QMHand 实例。"""

        hand = self._hands.get(device_name)
        if hand is not None:
            return hand
        hand_id: Any = QMHand.HAND_LEFT if device_name == "left_hand" else QMHand.HAND_RIGHT
        hand = QMHand(self._channel, hand_id)
        self._hands[device_name] = hand
        return hand

    def _arm_pb_type(self, device_name: ArmDeviceName) -> Any:
        """返回 qmlinker proto 中的机械臂枚举值。"""

        if device_name == "left_arm":
            return arm_pb2.ArmType.ARM_LEFT
        return arm_pb2.ArmType.ARM_RIGHT

    def _module_stub(self, module_name: WujiModuleName) -> Any:
        """返回非机械臂模块的 qmlinker gRPC stub。"""

        if module_name == "body":
            return self._lift_stub
        if module_name == "head":
            return self._head_stub
        raise ValueError(f"unsupported non-arm module: {module_name}")

    def _arm_device_name(self, module_name: WujiModuleName) -> ArmDeviceName:
        """将整机模块名收窄为机械臂设备名。"""

        if module_name == "left_arm":
            return "left_arm"
        if module_name == "right_arm":
            return "right_arm"
        raise ValueError(f"module is not an arm device: {module_name}")

    def _get_stub_enable(self, stub: Any) -> bool:
        """读取通用模块 stub 的使能状态。"""

        response = stub.GetEnabled(empty_pb2.Empty(), timeout=self._config.request_timeout_s)
        return bool(response.status.success and response.current_state == common_pb2.MODULE_ENABLED)

    def _set_stub_enable(self, stub: Any, enabled: bool) -> bool:
        """设置通用模块 stub 的使能状态。"""

        request = common_pb2.ModuleEnableRequest(enable=bool(enabled))
        response = stub.SetEnabled(request, timeout=self._config.request_timeout_s)
        return bool(response.status.success)


# endregion
