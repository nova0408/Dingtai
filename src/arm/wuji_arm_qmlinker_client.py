from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from qmlinker import QMArm, create_channel
from qmlinker.grpc_py import arm_pb2

from src.arm.wuji_arm_protocol import ArmDeviceName, WujiArmQmlinkerConfig

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
    - 不通过 SSH 执行远端 Python，不修改远端环境。
    - 不持有 GUI 控件，不发 Qt 信号。

    设计思想：
    - 直接使用接口文档提供的 `qmlinker-1.0.8-py3-none-any.whl`，避免重复实现 SDK 内部逻辑。
    - 客户端持有一个 channel 和左右臂 QMArm 实例，复用 SDK 内部状态更新线程。

    生命周期：
    - 随 GUI 后端创建和关闭。
    - `close` 只停止 SDK 内部线程，不负责关闭 GUI 或 SSH 会话。

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

    def close(self) -> None:
        """停止 qmlinker 内部状态线程。"""

        for arm in self._arms.values():
            if hasattr(arm, "running"):
                arm.running = False
            thread = getattr(arm, "thread_joint_states", None)
            if thread is not None:
                thread.join(timeout=0.5)
        self._arms.clear()

    def check_ready(self) -> None:
        """检查 qmlinker gRPC 通道是否可创建并进入 ready。

        Raises
        ------
        TimeoutError
            在配置时间内 gRPC 通道无法 ready。
        """

        channel = self._channel["DEFAULT"] if isinstance(self._channel, dict) else self._channel
        import grpc

        grpc.channel_ready_future(channel).result(timeout=self._config.request_timeout_s)

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

    def _arm(self, device_name: ArmDeviceName) -> Any:
        """返回指定机械臂的 qmlinker QMArm 实例。"""

        arm = self._arms.get(device_name)
        if arm is not None:
            return arm
        arm_type: Any = QMArm.ARM_LEFT if device_name == "left_arm" else QMArm.ARM_RIGHT
        arm = QMArm(self._channel, arm_type)
        self._arms[device_name] = arm
        return arm

    def _arm_pb_type(self, device_name: ArmDeviceName) -> Any:
        """返回 qmlinker proto 中的机械臂枚举值。"""

        if device_name == "left_arm":
            return arm_pb2.ArmType.ARM_LEFT
        return arm_pb2.ArmType.ARM_RIGHT


# endregion
