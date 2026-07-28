from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import RLock
from typing import Literal

from sdk.xcoresdk import xCoreSDK_python

# region 数据结构

Ar5Side = Literal["left", "right"]
Ar5JogSpace = Literal["cartesian", "joint"]


@dataclass(frozen=True, slots=True)
class Ar5ConnectionConfig:
    """单台 AR5 的连接配置。

    该结构只描述 SDK 连接端点与预期侧别，不持有 SDK 或网络资源。左右臂型号在建立
    连接后由控制器上报值再次校验，防止本地转发地址接反。
    """

    side: Ar5Side
    "预期机械臂侧别。"

    robot_ip: str
    "SDK 连接地址；直连时为控制器地址，SSH 转发时为独立 loopback 地址。"

    local_ip: str | None = None
    "SDK 实时数据绑定地址；不指定时由 SDK 自动选择。"


@dataclass(frozen=True, slots=True)
class Ar5Snapshot:
    """AR5 一帧只读状态。

    该结构是硬件适配层与 GUI 之间的数据契约。所有角度和长度均已转换为适合人工
    查看与输入的 deg/mm，SDK 原始 m/rad 不会从该对象回灌到计算链路。
    """

    robot_type: str
    "控制器上报的机器人型号。"

    robot_uid: str
    "控制器上报的机器人唯一标识。"

    operation_state: str
    "操作状态名称，例如 idle、moving、jogging 或 drag。"

    operate_mode: str
    "操作模式名称，例如 manual 或 automatic。"

    power_state: str
    "电机状态名称，例如 on、off、estop 或 gstop。"

    joint_deg: tuple[float, ...]
    "七个机械臂关节角，单位 deg。"

    xyz_mm: tuple[float, float, float]
    "当前工具末端位置，单位 mm，顺序为 X/Y/Z。"

    rpy_deg: tuple[float, float, float]
    "SDK 当前工具末端姿态，单位 deg，使用小写外禀 xyz 约定。"

    elbow_deg: float
    "当前七轴臂角，单位 deg。"

    has_elbow: bool
    "当前笛卡尔点位是否携带臂角约束。"


# endregion


# region 固定配置

AR5_EXPECTED_TYPES: dict[Ar5Side, str] = {
    "left": "AR5-5_0.8L-W4C1C9-ZY2",
    "right": "AR5-5_0.8R-W4C1C9-ZY2",
}
"现场左右 AR5 控制器应上报的完整型号。"

AR5_DIRECT_IPS: dict[Ar5Side, str] = {
    "left": "192.168.100.161",
    "right": "192.168.100.160",
}
"平板直连时两台 AR5 控制器的固定地址。"

AR5_TUNNEL_IPS: dict[Ar5Side, str] = {
    "left": "127.0.0.2",
    "right": "127.0.0.3",
}
"本机 SSH 转发时用于区分两台 SDK 固定端口的 loopback 地址。"

AR5_SSH_FORWARD_PORTS: tuple[int, ...] = (5050, 4567, 6666, 7777)
"xCoreSDK 已验证需要转发的固定端口。"

DEFAULT_MOVE_SPEED_MM_S = 1000.0
"MoveAbsJ 与 MoveL 默认末端线速度，单位 mm/s。"

DEFAULT_MOVE_ZONE_MM = 10.0
"非实时运动默认转弯区半径，单位 mm。"

POWER_STATE_TIMEOUT_S = 3.0
"等待控制器确认上下电状态的最长时间，单位 s。"


# endregion


# region AR5 客户端


class Ar5Client:
    """基于官方 xCoreSDK 的单台 AR5 硬件客户端。

    职责边界：
    - 负责 SDK 对象、错误码、单位换算和机器人控制状态切换
    - 提供状态、Move、Jog、拖动以及独立 elbow 调整
    - 不创建 SSH 隧道，不依赖 Qt，也不保存 GUI 表单状态

    设计思想：
    - 用同一把可重入锁串行化 GUI 刷新线程与控制线程对 SDK 的访问
    - SDK 内部始终保留 m/rad，只有接口边界转换为 mm/deg
    - 独立 elbow 调整通过当前 TCP + 指定臂角求逆解，再执行 MoveAbsJ

    生命周期：
    - 构造时连接并校验机器人型号
    - ``close`` 时先停止运动和拖动，再断开 SDK
    - 可由多个 GUI 后台线程调用，但同一实例上的 SDK 调用会被串行化

    继承关系：
    - 不继承业务基类，作为官方 SDK 的窄适配器使用
    """

    def __init__(self, config: Ar5ConnectionConfig) -> None:
        """连接并校验一台 AR5。

        Parameters
        ----------
        config:
            SDK 端点和预期左右侧配置。

        Raises
        ------
        RuntimeError
            SDK 返回错误码或控制器型号与预期侧别不匹配。
        """

        self._config = config
        self._lock = RLock()
        self._ec: dict[str, object] = {}
        if config.local_ip is None:
            self._robot = xCoreSDK_python.xMateErProRobot(config.robot_ip)
        else:
            self._robot = xCoreSDK_python.xMateErProRobot(config.robot_ip, config.local_ip)
        robot_info = self._call_value("robotInfo", self._robot.robotInfo)
        expected_type = AR5_EXPECTED_TYPES[config.side]
        if robot_info.type != expected_type:
            self.close()
            raise RuntimeError(
                f"AR5 型号与侧别不匹配: expected={expected_type}, actual={robot_info.type}, ip={config.robot_ip}"
            )
        self._robot_type = str(robot_info.type)
        self._robot_uid = str(robot_info.id)

    @property
    def side(self) -> Ar5Side:
        """返回机械臂侧别。"""

        return self._config.side

    def read_snapshot(self) -> Ar5Snapshot:
        """读取一帧 AR5 状态并转换为 GUI 单位。

        Returns
        -------
        snapshot:
            关节、TCP、elbow、电机和运行模式的同一调用周期快照。

        Notes
        -----
        ``CartesianPosition.pos`` 在当前 SDK 路径可能为空，因此只读取 ``trans`` 和
        ``rpy``。两者原始单位分别为 m 和 rad。
        """

        with self._lock:
            operation_state = self._call_value("operationState", self._robot.operationState)
            operate_mode = self._call_value("operateMode", self._robot.operateMode)
            power_state = self._call_value("powerState", self._robot.powerState)
            cartesian_pose = self._call_value(
                "cartPosture(endInRef)",
                lambda ec: self._robot.cartPosture(xCoreSDK_python.endInRef, ec),
            )
            joint_rad = tuple(self._call_value("jointPos", self._robot.jointPos))
            translation_m = tuple(float(value) for value in cartesian_pose.trans)
            sdk_rpy_rad = tuple(float(value) for value in cartesian_pose.rpy)
            return Ar5Snapshot(
                robot_type=self._robot_type,
                robot_uid=self._robot_uid,
                operation_state=operation_state.name,
                operate_mode=operate_mode.name,
                power_state=power_state.name,
                joint_deg=tuple(math.degrees(float(value)) for value in joint_rad[:7]),
                xyz_mm=(
                    translation_m[0] * 1000.0,
                    translation_m[1] * 1000.0,
                    translation_m[2] * 1000.0,
                ),
                rpy_deg=(
                    math.degrees(sdk_rpy_rad[0]),
                    math.degrees(sdk_rpy_rad[1]),
                    math.degrees(sdk_rpy_rad[2]),
                ),
                elbow_deg=math.degrees(float(cartesian_pose.elbow)),
                has_elbow=bool(cartesian_pose.hasElbow),
            )

    def set_power(self, enabled: bool) -> None:
        """设置电机上下电状态。

        Parameters
        ----------
        enabled:
            ``True`` 上电，``False`` 下电。切换前先停止现有运动。
        """

        with self._lock:
            self._call_none("stop", self._robot.stop)
            self._call_none(
                "setMotionControlMode(NrtCommandMode)",
                lambda ec: self._robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec),
            )
            self._call_none("setPowerState", lambda ec: self._robot.setPowerState(enabled, ec))
            target_state = (
                xCoreSDK_python.PowerState.on
                if enabled
                else xCoreSDK_python.PowerState.off
            )
            self._wait_power_state_locked(target_state)

    def set_operate_mode(self, automatic: bool) -> None:
        """设置机器人手动或自动工作模式。

        Parameters
        ----------
        automatic:
            ``True`` 切换到自动模式，``False`` 切换到手动模式。

        Notes
        -----
        该调用与已验证 CLI 一致，直接使用官方 SDK 的 ``setOperateMode``。页面负责
        根据最新状态决定切换方向，本方法不读取或猜测当前模式。
        """

        target_mode = (
            xCoreSDK_python.OperateMode.automatic
            if automatic
            else xCoreSDK_python.OperateMode.manual
        )
        with self._lock:
            self._call_none(
                f"setOperateMode({target_mode.name})",
                lambda ec: self._robot.setOperateMode(target_mode, ec),
            )

    def recover_estop(self) -> None:
        """请求控制器执行急停恢复。

        Notes
        -----
        调用参数 ``1`` 来自项目中已经过硬件验证的 AR5 CLI。该接口只发送恢复请求，
        不会自动上电；现场急停按钮仍处于按下状态时，控制器会返回真实错误。
        """

        with self._lock:
            self._call_none(
                "recoverState(1)",
                lambda ec: self._robot.recoverState(1, ec),
            )

    def move_joints_deg(
        self,
        joint_deg: tuple[float, ...],
        *,
        speed_mm_s: float = DEFAULT_MOVE_SPEED_MM_S,
        zone_mm: float = DEFAULT_MOVE_ZONE_MM,
    ) -> None:
        """执行七关节绝对运动。

        Parameters
        ----------
        joint_deg:
            七个目标关节角，单位 deg。
        speed_mm_s:
            MoveAbsJ 末端线速度，单位 mm/s。
        zone_mm:
            转弯区半径，单位 mm。
        """

        if len(joint_deg) != 7:
            raise ValueError(f"AR5 关节目标必须为 7 维，实际为 {len(joint_deg)}")
        target_joint_rad = [math.radians(float(value)) for value in joint_deg]
        command = xCoreSDK_python.MoveAbsJCommand(target_joint_rad, speed_mm_s, zone_mm)
        self._execute_move(command)

    def move_cartesian(
        self,
        xyz_mm: tuple[float, float, float],
        rpy_deg: tuple[float, float, float],
        elbow_deg: float,
        *,
        speed_mm_s: float = DEFAULT_MOVE_SPEED_MM_S,
        zone_mm: float = DEFAULT_MOVE_ZONE_MM,
    ) -> None:
        """执行带 elbow 约束的笛卡尔直线运动。

        Parameters
        ----------
        xyz_mm:
            目标 TCP 平移，单位 mm，顺序 X/Y/Z。
        rpy_deg:
            目标 TCP 欧拉角，单位 deg，使用小写外禀 xyz 约定。
        elbow_deg:
            七轴臂角约束，单位 deg。
        speed_mm_s:
            MoveL 末端线速度，单位 mm/s。
        zone_mm:
            转弯区半径，单位 mm。
        """

        target_pose = self._build_cartesian_target(xyz_mm, rpy_deg, elbow_deg)
        command = xCoreSDK_python.MoveLCommand(target_pose, speed_mm_s, zone_mm)
        self._execute_move(command)

    def move_elbow_deg(
        self,
        elbow_deg: float,
        *,
        speed_mm_s: float = DEFAULT_MOVE_SPEED_MM_S,
        zone_mm: float = DEFAULT_MOVE_ZONE_MM,
    ) -> None:
        """保持当前 TCP，单独调整 elbow。

        Parameters
        ----------
        elbow_deg:
            目标臂角，单位 deg。
        speed_mm_s:
            MoveAbsJ 末端线速度，单位 mm/s。
        zone_mm:
            转弯区半径，单位 mm。

        Notes
        -----
        当前 TCP 直接使用 SDK 的 m/rad 构造 ``CartesianPosition``，设置 elbow 后通过
        当前 toolset 求逆解。显示用 mm/deg 不参与计算。
        """

        with self._lock:
            current_pose = self._call_value(
                "cartPosture(endInRef)",
                lambda ec: self._robot.cartPosture(xCoreSDK_python.endInRef, ec),
            )
            target_pose = xCoreSDK_python.CartesianPosition(
                [float(value) for value in current_pose.trans],
                [float(value) for value in current_pose.rpy],
            )
            target_pose.hasElbow = True
            target_pose.elbow = math.radians(elbow_deg)
            toolset = self._call_value("toolset", self._robot.toolset)
            target_joint_rad = self._call_value(
                "calcIk(elbow)",
                lambda ec: self._robot.model().calcIk(target_pose, toolset, ec),
            )
            command = xCoreSDK_python.MoveAbsJCommand(
                [float(value) for value in target_joint_rad],
                speed_mm_s,
                zone_mm,
            )
            self._execute_move_locked(command)

    def start_jog(
        self,
        space: Ar5JogSpace,
        axis_index: int,
        direction_positive: bool,
        *,
        rate: float,
        step: float,
    ) -> None:
        """启动单轴 Jog。

        Parameters
        ----------
        space:
            ``cartesian`` 或 ``joint``。
        axis_index:
            零基轴索引；笛卡尔空间为 X/Y/Z/Rx/Ry/Rz，关节空间为 J1-J7。
        direction_positive:
            ``True`` 正向，``False`` 负向。
        rate:
            SDK Jog 速率，范围 0.01-1.00。
        step:
            笛卡尔平移轴单位 mm；旋转轴和关节轴单位 deg。
        """

        if not 0.01 <= rate <= 1.0:
            raise ValueError("Jog rate 必须位于 0.01-1.00")
        max_axis_index = 5 if space == "cartesian" else 6
        if not 0 <= axis_index <= max_axis_index:
            raise ValueError(f"{space} Jog 轴索引无效: {axis_index}")
        with self._lock:
            self._prepare_jog_locked()
            sdk_space = (
                xCoreSDK_python.JogOptSpace.baseFrame
                if space == "cartesian"
                else xCoreSDK_python.JogOptSpace.jointSpace
            )
            self._call_none(
                f"startJog({space}, {axis_index})",
                lambda ec: self._robot.startJog(
                    sdk_space,
                    rate,
                    step,
                    axis_index,
                    direction_positive,
                    ec,
                ),
            )

    def stop(self) -> None:
        """停止当前 Move 或 Jog。"""

        with self._lock:
            self._call_none("stop", self._robot.stop)

    def set_drag_enabled(self, enabled: bool) -> None:
        """开启或关闭笛卡尔自由拖动。

        Parameters
        ----------
        enabled:
            ``True`` 开启无需末端按键的自由拖动并自动下电，``False`` 关闭拖动并
            自动上电。

        Raises
        ------
        RuntimeError
            开启拖动时机器人不是手动模式，或 SDK 拒绝切换使能状态。
        """

        with self._lock:
            if not enabled:
                self._call_none("disableDrag", self._robot.disableDrag)
                power_state = self._call_value(
                    "powerState",
                    self._robot.powerState,
                )
                if power_state != xCoreSDK_python.PowerState.on:
                    self._call_none(
                        "setPowerState(True)",
                        lambda ec: self._robot.setPowerState(True, ec),
                    )
                    self._wait_power_state_locked(xCoreSDK_python.PowerState.on)
                return
            operate_mode = self._call_value(
                "operateMode",
                self._robot.operateMode,
            )
            if operate_mode != xCoreSDK_python.OperateMode.manual:
                raise RuntimeError("开启拖动前请先将工作模式切换为手动")
            self._call_none(
                "setMotionControlMode(NrtCommandMode)",
                lambda ec: self._robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec),
            )
            self._call_none("setPowerState(False)", lambda ec: self._robot.setPowerState(False, ec))
            self._wait_power_state_locked(xCoreSDK_python.PowerState.off)
            self._call_none("moveReset", self._robot.moveReset)
            self._call_none(
                "enableDrag(cartesian, freely)",
                lambda ec: self._robot.enableDrag(
                    int(xCoreSDK_python.DragParameterSpace.cartesianSpace),
                    int(xCoreSDK_python.DragParameterType.freely),
                    ec,
                    enable_drag_button=False,
                ),
            )

    def close(self) -> None:
        """停止并断开机器人连接，不主动改变现场电机状态。"""

        with self._lock:
            try:
                self._call_none("stop", self._robot.stop)
            except Exception:
                pass
            try:
                self._call_none("disableDrag", self._robot.disableDrag)
            except Exception:
                pass
            try:
                self._call_none("disconnectFromRobot", self._robot.disconnectFromRobot)
            except Exception:
                pass

    # region 内部 SDK 调用

    def _prepare_jog_locked(self) -> None:
        """在已持锁状态下验证 Jog 所需模式和使能，不主动切换状态。"""

        operate_mode = self._call_value("operateMode", self._robot.operateMode)
        if operate_mode != xCoreSDK_python.OperateMode.manual:
            raise RuntimeError("Jog 前请先将工作模式切换为手动")
        power_state = self._call_value("powerState", self._robot.powerState)
        if power_state != xCoreSDK_python.PowerState.on:
            raise RuntimeError("Jog 前请先将使能状态切换为上电")

    def _prepare_move_locked(self) -> None:
        """在已持锁状态下验证 Move 所需模式和使能，不主动切换状态。"""

        operate_mode = self._call_value("operateMode", self._robot.operateMode)
        if operate_mode != xCoreSDK_python.OperateMode.automatic:
            raise RuntimeError("Move 前请先将工作模式切换为自动")
        power_state = self._call_value("powerState", self._robot.powerState)
        if power_state != xCoreSDK_python.PowerState.on:
            raise RuntimeError("Move 前请先将使能状态切换为上电")

    def _wait_power_state_locked(self, target_state: object) -> None:
        """在已持锁状态下等待控制器确认目标电机状态。"""

        deadline = time.monotonic() + POWER_STATE_TIMEOUT_S
        while time.monotonic() < deadline:
            current_state = self._call_value("powerState", self._robot.powerState)
            if current_state == target_state:
                return
            time.sleep(0.1)
        raise RuntimeError(f"等待电机状态超时: target={target_state}")

    def _execute_move(
        self,
        command: xCoreSDK_python.MoveAbsJCommand | xCoreSDK_python.MoveLCommand,
    ) -> None:
        """串行下发单条非实时运动命令。"""

        with self._lock:
            self._execute_move_locked(command)

    def _execute_move_locked(
        self,
        command: xCoreSDK_python.MoveAbsJCommand | xCoreSDK_python.MoveLCommand,
    ) -> None:
        """在已持锁状态下下发单条非实时运动命令。"""

        self._prepare_move_locked()
        self._call_none("moveReset", self._robot.moveReset)
        command_id = xCoreSDK_python.PyString()
        if isinstance(command, xCoreSDK_python.MoveAbsJCommand):
            self._call_none(
                "moveAppend(MoveAbsJ)",
                lambda ec: self._robot.moveAppend([command], command_id, ec),
            )
        else:
            self._call_none(
                "moveAppend(MoveL)",
                lambda ec: self._robot.moveAppend([command], command_id, ec),
            )
        self._call_none("moveStart", self._robot.moveStart)

    @staticmethod
    def _build_cartesian_target(
        xyz_mm: tuple[float, float, float],
        rpy_deg: tuple[float, float, float],
        elbow_deg: float,
    ) -> xCoreSDK_python.CartesianPosition:
        """把 GUI 的 mm/deg 目标转换为 SDK 的 m/rad 点位。"""

        target_pose = xCoreSDK_python.CartesianPosition(
            [float(value) / 1000.0 for value in xyz_mm],
            [math.radians(float(value)) for value in rpy_deg],
        )
        target_pose.hasElbow = True
        target_pose.elbow = math.radians(elbow_deg)
        return target_pose

    def _call_none(self, action: str, callback) -> None:  # noqa: ANN001
        """调用一个无返回值 SDK 方法并检查错误码。"""

        self._ec.clear()
        callback(self._ec)
        self._raise_for_error(action)

    def _call_value(self, action: str, callback):  # noqa: ANN001, ANN202
        """调用一个有返回值 SDK 方法并检查错误码。"""

        self._ec.clear()
        value = callback(self._ec)
        self._raise_for_error(action)
        return value

    def _raise_for_error(self, action: str) -> None:
        """把 SDK 错误码转换为带动作上下文的异常。"""

        error_code = self._ec.get("ec", 0)
        if error_code == 0:
            return
        raise RuntimeError(
            f"{action} 失败: ec={error_code}, message={self._ec.get('message', '')}"
        )

    # endregion


# endregion
