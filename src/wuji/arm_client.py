from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import atan2, pi, sqrt
from time import monotonic, sleep
from typing import Any, Iterator, cast

import numpy as np
from qmlinker import QMArm
from qmlinker.grpc_py import arm_pb2

from src.arm.wuji_arm_protocol import ArmDeviceName, WujiArmJointLimit
from src.wuji.client_base import WujiQmlinkerBaseClient


# region 机械臂客户端


class WujiArmClient(QMArm):
    """无际机械臂运行时客户端。

    职责边界：
    - 直接继承 `QMArm`，负责机械臂的使能、关节状态、关节控制和运动学查询。
    - 不负责 GUI 组装、订阅调度、相机链路或底盘控制。

    设计思想：
    - 只把 `qmlinker` SDK 中对项目不够固定的部分收口成项目侧方法。
    - 关节状态读取直接走服务端流式 RPC，不再依赖本地缓存快照，避免单轴控制拼接时
      使用过期值。
    - 机械臂左右臂由 `device_name` 显式区分，不做动态分发。
    - 只保存必要的 base 引用，用于读取统一超时和默认速度比例。

    生命周期：
    - 随 `WujiQmlinkerBaseClient` 的 channel 生命周期创建和关闭。
    - 不额外持有线程或后台任务，可跨调用复用。

    继承关系：
    - 直接继承 `QMArm`，不再通过中间厚 client 间接代理。
    """

    # region 初始化

    def __init__(self, base_client: WujiQmlinkerBaseClient, device_name: ArmDeviceName) -> None:
        """创建机械臂客户端。

        Parameters
        ----------
        base_client:
            提供 `qmlinker` channel 和统一配置的基础客户端。
        device_name:
            机械臂设备名，当前仅支持 `left_arm` 和 `right_arm`。
        """

        super().__init__(
            base_client.channel,
            cast(str, QMArm.ARM_LEFT if device_name == "left_arm" else QMArm.ARM_RIGHT),
        )
        self._base = base_client
        self._device_name = device_name

    # endregion

    # region 关节状态读取

    def get_arm_joint_count(self, device_name: ArmDeviceName) -> int:
        """返回机械臂关节数。

        Parameters
        ----------
        device_name:
            机械臂设备名。

        Returns
        -------
        int
            当前机械臂关节数量。
        """

        return len(self._read_joint_states())

    def get_joint_states(self, device_name: ArmDeviceName) -> tuple[Any, ...]:
        """返回当前关节状态。

        Parameters
        ----------
        device_name:
            机械臂设备名。

        Returns
        -------
        tuple[Any, ...]
            按关节编号排序后的关节状态元组。

            每个关节状态对象来自 `qmlinker` SDK，通常包含：

            - `joint_id`：关节编号；
            - `angle_deg`：当前关节角度，单位 deg；
            - 其他 SDK 内部字段。

        Notes
        -----
        该接口每次都会通过服务端流式 RPC 读取当前状态，不依赖本地缓存。
        """

        return self._read_joint_states()

    # endregion

    # region 关节控制

    def set_joint(self, device_name: ArmDeviceName, joint_index: int, target_angle_deg: float) -> bool:
        """设置单个关节目标角度。

        Parameters
        ----------
        device_name:
            机械臂设备名。

            当前客户端实例在初始化时已经绑定左臂或右臂，因此该参数主要用于保持
            项目侧统一接口形式。

        joint_index:
            关节编号，从 1 开始，与 `qmlinker` 关节命名和 GUI 轴名保持一致。

            例如：

            - `joint_index=1` 表示 J1；
            - `joint_index=2` 表示 J2；
            - `joint_index=6` 表示 J6。

        target_angle_deg:
            目标角度，单位 deg。

            该值是真实关节角度，不是 0~1 归一化值。

        Returns
        -------
        bool
            底层整臂 `set_joints()` 调用结果。

        Notes
        -----
        底层 qmlinker 的单次控制接口使用整臂关节命令列表，因此这里的实现方式是：

        1. 先读取当前 6 个关节角度；
        2. 只替换指定 `joint_index` 对应关节的目标角度；
        3. 其余关节保持当前角度；
        4. 调用底层 `QMArm.set_joints()` 下发整臂目标。

        这样可以在项目侧提供“单关节控制”的语义，同时避免调用方手动拼接
        6 个关节值。
        """

        current_joints = list(self._read_joint_states())
        if len(current_joints) != 6:
            return False

        target_joint_id = int(joint_index)
        if not 1 <= target_joint_id <= len(current_joints):
            return False

        joint_commands: list[dict[str, float | int]] = []
        for current_joint_id, joint in enumerate(current_joints, start=1):
            joint_commands.append(
                {
                    "joint_id": current_joint_id,
                    "target_angle_deg": float(target_angle_deg)
                    if current_joint_id == target_joint_id
                    else float(getattr(joint, "angle_deg", 0.0)),
                    "speed_ratio": float(self._base.config.default_speed_ratio),
                }
            )

        return bool(super().set_joints(joint_commands))

    def set_joints(self, joint_commands: Sequence[float], sync_threshold: int = 0) -> bool:
        """下发整臂关节目标角度。

        Parameters
        ----------
        joint_commands:
            整臂关节目标角度序列，单位 deg。

            序列顺序必须与机械臂关节顺序一致：

            - 第 0 个元素对应 J1；
            - 第 1 个元素对应 J2；
            - 第 5 个元素对应 J6。

            注意：这里输入的是真实角度 deg，不是 0~1 归一化值。

        sync_threshold:
            同步阈值，直接透传给底层 `QMArm.set_joints()`。

            默认值为 `0`。具体含义由 qmlinker SDK 定义，项目侧不额外解释或转换。

        Returns
        -------
        bool
            底层 `QMArm.set_joints()` 调用结果。

        Raises
        ------
        ValueError
            当 `joint_angles_deg` 为空时抛出。
        """

        if len(joint_commands) == 0:
            raise ValueError("joint_commands 不能为空。")

        return bool(
            super().set_joints(
                self._build_joint_commands(joint_commands),
                sync_threshold=sync_threshold,
            )
        )

    # endregion

    # region 运动学计算

    def current_fk_fast(self, device_name: ArmDeviceName) -> Any:
        """基于当前关节角快速计算正运动学。

        Parameters
        ----------
        device_name:
            机械臂设备名。

            当前客户端实例已经在初始化时绑定具体机械臂，因此该参数主要用于保持
            项目侧统一接口形式。

        Returns
        -------
        Any
            底层 `fk_fast()` 返回的 4x4 齐次变换矩阵。

            当前 qmlinker SDK 返回值通常可转换为 `numpy.ndarray`，其矩阵结构为：

            - 左上角 3x3：旋转矩阵；
            - 右上角 3x1：平移向量；
            - 最后一行：齐次坐标行 `[0, 0, 0, 1]`。

        Notes
        -----
        该方法保持原始语义：返回 4x4 位姿矩阵。

        如果需要直接得到 `x, y, z, roll, pitch, yaw`，请调用
        `current_fk_xyzrpy()`，不要修改本方法的返回类型，以免破坏已有调用。
        """

        joint_states = self._read_joint_states()
        if len(joint_states) != 6:
            raise RuntimeError(f"机械臂关节状态尚未就绪，当前关节数={len(joint_states)}")

        joint_angles_rad = [float(np.deg2rad(joint.angle_deg)) for joint in joint_states]
        return self.fkik.fk_fast(joint_angles_rad)

    def current_fk_xyzrpy(self, device_name: ArmDeviceName) -> tuple[float, float, float, float, float, float]:
        """基于当前关节角计算末端 xyzrpy。

        Parameters
        ----------
        device_name:
            机械臂设备名。

        Returns
        -------
        tuple[float, float, float, float, float, float]
            当前末端位姿，格式为：

            ``(x, y, z, roll_deg, pitch_deg, yaw_deg)``

            其中：

            - `x` / `y` / `z` 单位为 m；
            - `roll_deg` / `pitch_deg` / `yaw_deg` 单位为 deg；
            - RPY 使用 ZYX 欧拉角约定；
            - 对应旋转矩阵关系为
              `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

        Raises
        ------
        RuntimeError
            当 FK 返回空值、矩阵尺寸异常，或矩阵中存在非有限数值时抛出。

        Notes
        -----
        该方法是 `current_fk_fast()` 的便捷封装，不改变底层 FK 的计算逻辑。
        """

        transform = self.current_fk_fast(device_name)
        return self._pose_matrix_to_xyzrpy_deg(transform)

    def fk(self, joint_angles_rad: Iterable[float]) -> Any:
        """计算机械臂正运动学。

        Parameters
        ----------
        joint_angles_rad:
            关节角度序列，单位 rad。

        Returns
        -------
        Any
            底层 `fk()` 返回的位姿结果。
        """

        return self.fkik.fk(list(joint_angles_rad))

    def fk_fast(self, joint_angles_rad: Iterable[float]) -> Any:
        """快速计算机械臂正运动学。

        Parameters
        ----------
        joint_angles_rad:
            关节角度序列，单位 rad。

        Returns
        -------
        Any
            底层 `fk_fast()` 返回的位姿结果。
        """

        return self.fkik.fk_fast(list(joint_angles_rad))

    def ik(self, target_pose: Any, reference_joint_angles_rad: Iterable[float]) -> Any:
        """计算机械臂逆运动学。

        Parameters
        ----------
        target_pose:
            目标末端位姿。

            通常为 4x4 齐次变换矩阵，具体格式由 qmlinker SDK 的 `fkik.ik()`
            接口定义。

        reference_joint_angles_rad:
            参考关节角度序列，单位 rad。

            IK 求解通常需要参考当前关节位姿或附近位姿，以便选择更接近当前姿态的解。

        Returns
        -------
        Any
            底层 `ik()` 返回的关节角度结果。

            当前项目中通常返回 6 个关节角，单位 rad。
        """

        return self.fkik.ik(target_pose, list(reference_joint_angles_rad))

    # endregion

    # region 关节限位

    def get_arm_joint_limits(self, device_name: ArmDeviceName) -> tuple[WujiArmJointLimit, ...]:
        """返回机械臂关节限位。

        Parameters
        ----------
        device_name:
            机械臂设备名。

        Returns
        -------
        tuple[WujiArmJointLimit, ...]
            机械臂关节限位列表。

            每个元素包含：

            - 关节名；
            - 最小角度，单位 deg；
            - 最大角度，单位 deg。

        Notes
        -----
        底层 `qmlinker` 的 `joint_min` / `joint_max` 以弧度存储，这里统一转换为 deg。
        """

        joint_mins = list(getattr(self.fkik, "joint_min", ()))
        joint_maxs = list(getattr(self.fkik, "joint_max", ()))
        joint_count = min(len(joint_mins), len(joint_maxs))

        return tuple(
            WujiArmJointLimit(
                f"j{index}",
                float(joint_mins[index - 1] * 180.0 / pi),
                float(joint_maxs[index - 1] * 180.0 / pi),
            )
            for index in range(1, joint_count + 1)
        )

    # endregion

    # region 停止与生命周期清理

    def stop(self) -> None:
        """停止机械臂后台线程。

        Notes
        -----
        这一级 `stop()` 只负责终止项目侧的后台读取和发送链路，不等同于硬件急停。

        当前 `wuyou` 服务端链路为：

        - GUI / client 通过 `qmlinker` 调用机械臂接口；
        - `grpc_bridge_v2` 只提供 `SetJointStates`、`SetArmPose`、`SetEnabled` 等转发接口；
        - `actuators_arm` 将关节命令写入 `ArmBridge.move_arm()`；
        - `ArmBridge` 再把目标角度写入 `ArmControl` 控制结构；
        - 服务端目前没有单独暴露 `Stop` / `EmergencyStop` RPC。

        因此这里的停止语义是：

        - 结束客户端侧后台轮询和发送线程；
        - 避免继续向服务端发送新的关节目标；
        - 不能替代机械臂底层的硬件急停或刹车指令。

        上游 `QMArm.stop()` 会无条件等待 `thread_arm_pose`，但当前 SDK 版本并未启动该线程。
        这里显式补齐属性，避免在 `close()` / 析构时抛出属性错误。

        如果后续在 `actuators_arm` 或更底层驱动中补出明确的急停接口，应优先调用
        该接口，再执行这里的线程清理逻辑。
        """

        self.running = False
        joint_thread = getattr(self, "thread_joint_states", None)
        if joint_thread is not None:
            joint_thread.join(timeout=1)

    # endregion

    # region 内部工具方法

    def _read_joint_states(self) -> tuple[Any, ...]:
        """读取机械臂当前关节状态。

        Returns
        -------
        tuple[Any, ...]
            按 `joint_id` 排序后的关节状态元组。

        Notes
        -----
        这里优先读取 `QMArm` 内部流式线程持续更新的 `arm_info.joint_states`。

        这样可以避免：

        - 反复创建和取消 `StreamGetJointStates` 流；
        - GUI 高刷场景下重复建立 gRPC 流带来的抖动；
        - 订阅能力已存在时又额外走主动拉取。
        """
        _ = arm_pb2
        arm_info = getattr(self, "arm_info", None)
        arm_info_lock = getattr(self, "arm_info_lock", None)
        if arm_info is None or arm_info_lock is None:
            return tuple()

        with arm_info_lock:
            if not bool(getattr(arm_info, "initialized", False)):
                return tuple()
            joint_states = tuple(getattr(arm_info, "joint_states", ()))

        return joint_states

    def _build_joint_commands(self, joint_angles_deg: Sequence[float], start_joint_id: int = 1) -> list[dict[str, float | int]]:
        """将角度序列封装为 qmlinker 关节命令。

        Parameters
        ----------
        joint_angles_deg:
            关节目标角度序列，单位 deg。

        start_joint_id:
            起始关节编号。

            默认值为 `1`，表示输入序列第 0 个元素对应 qmlinker 的 `joint_id=1`。

        Returns
        -------
        list[dict[str, float | int]]
            qmlinker `set_joints()` 所需的关节命令列表。

            每个命令包含：

            - `joint_id`；
            - `target_angle_deg`；
            - `speed_ratio`。
        """

        return [
            {
                "joint_id": start_joint_id + index,
                "target_angle_deg": float(angle_deg),
                "speed_ratio": float(self._base.config.default_speed_ratio),
            }
            for index, angle_deg in enumerate(joint_angles_deg)
        ]

    def _pose_matrix_to_xyzrpy_deg(self, transform: Any) -> tuple[float, float, float, float, float, float]:
        """将 4x4 齐次变换矩阵转换为 xyzrpy。

        Parameters
        ----------
        transform:
            4x4 齐次变换矩阵。

            矩阵结构应为：

            - `transform[:3, :3]`：旋转矩阵；
            - `transform[:3, 3]`：平移向量；
            - `transform[3, :]`：齐次坐标行。

            平移单位保持底层 FK 返回值的单位。当前项目中通常为 m。

        Returns
        -------
        tuple[float, float, float, float, float, float]
            末端位姿，格式为：

            ``(x, y, z, roll_deg, pitch_deg, yaw_deg)``

            其中：

            - `x` / `y` / `z` 为平移分量；
            - `roll_deg` 为绕 X 轴旋转角；
            - `pitch_deg` 为绕 Y 轴旋转角；
            - `yaw_deg` 为绕 Z 轴旋转角。

        Raises
        ------
        RuntimeError
            当输入矩阵不是 4x4，或包含非有限数值时抛出。

        Notes
        -----
        RPY 使用 ZYX 欧拉角约定：

        ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``

        常规情况下分解公式为：

        - `roll  = atan2(R[2, 1], R[2, 2])`
        - `pitch = atan2(-R[2, 0], sqrt(R[0, 0]^2 + R[1, 0]^2))`
        - `yaw   = atan2(R[1, 0], R[0, 0])`

        当 `pitch` 接近 ±90 deg 时会出现万向节锁，`roll` 和 `yaw`
        不再唯一。这里采用固定 `roll=0` 的稳定输出策略。
        """

        matrix = np.asarray(transform, dtype=float)

        if matrix.shape != (4, 4):
            raise RuntimeError(f"FK 位姿矩阵尺寸异常：shape={matrix.shape}")

        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("FK 位姿矩阵中存在非有限数值。")

        x = float(matrix[0, 3])
        y = float(matrix[1, 3])
        z = float(matrix[2, 3])

        r00 = float(matrix[0, 0])
        r10 = float(matrix[1, 0])
        r20 = float(matrix[2, 0])
        r21 = float(matrix[2, 1])
        r22 = float(matrix[2, 2])

        # ZYX 欧拉角分解。
        # cy 表示 cos(pitch) 的绝对尺度。
        # 当 cy 接近 0 时，pitch 接近 ±90 deg，此时会出现万向节锁。
        cy = sqrt(r00 * r00 + r10 * r10)

        if cy > 1e-9:
            roll_rad = atan2(r21, r22)
            pitch_rad = atan2(-r20, cy)
            yaw_rad = atan2(r10, r00)
        else:
            # 万向节锁附近，roll 和 yaw 不再唯一。
            # 这里固定 roll=0，并把剩余旋转量放到 yaw 中，保证输出稳定。
            roll_rad = 0.0
            pitch_rad = atan2(-r20, cy)
            yaw_rad = atan2(-float(matrix[0, 1]), float(matrix[1, 1]))

        return (
            x,
            y,
            z,
            float(np.rad2deg(roll_rad)),
            float(np.rad2deg(pitch_rad)),
            float(np.rad2deg(yaw_rad)),
        )

    # endregion


# endregion
