from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock, RLock
from typing import Literal

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

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

    该结构是硬件适配层与 GUI 之间的数据契约。人工查看字段使用 deg/mm；
    ``pose_matrix_m`` 在读取 SDK 原始 m/rad 后直接构造，供标定计算链路使用，
    避免把展示单位回灌到齐次矩阵。
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

    pose_matrix_m: tuple[tuple[float, ...], ...]
    "当前工具末端齐次变换，形状为 (4, 4)，平移单位 m，姿态使用小写外禀 xyz 约定。"

    xyz_mm: tuple[float, float, float]
    "当前工具末端位置，单位 mm，顺序为 X/Y/Z。"

    rpy_deg: tuple[float, float, float]
    "SDK 当前工具末端姿态，单位 deg，使用小写外禀 xyz 约定。"

    elbow_deg: float
    "当前七轴臂角，单位 deg。"

    has_elbow: bool
    "当前笛卡尔点位是否携带臂角约束。"

    conf_data: tuple[int, ...]
    "当前笛卡尔点位的控制器构型数据。"


@dataclass(frozen=True, slots=True)
class Ar5SoftLimitSnapshot:
    """AR5 七个轴的软限位只读快照。

    软限位上下限沿用 xCoreSDK 的原始单位 rad。``enabled`` 表示控制器当前是否
    启用软限位功能；该结构不包含任何写入控制器的操作。
    """

    enabled: bool
    "控制器当前是否启用软限位。"

    limits_rad: tuple[tuple[float, float], ...]
    "七个轴的下限和上限，顺序与控制器轴号一致，单位 rad。"


@dataclass(frozen=True, slots=True)
class Ar5MotionProgress:
    """当前 NRT 批量轨迹的事件进度与碰撞锁存。"""

    command_id: str | None = None
    "xCoreSDK 为最近一次 ``moveAppend`` 分配的路径 ID。"
    target_count: int = 0
    "当前路径包含的 waypoint 数量。"
    last_reached_waypoint_index: int | None = None
    "事件明确确认到达的最后一个 waypoint 下标，从 0 开始。"
    collision_detected: bool = False
    "当前路径是否收到控制器碰撞事件。"
    collision_code: int | None = None
    "控制器碰撞错误码；``collision_fc`` 为 30400。"
    collision_detail: str | None = None
    "控制器碰撞详情。"


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

OPERATE_MODE_TIMEOUT_S = 3.0
"等待控制器确认手动或自动模式的最长时间，单位 s。"

AR5_DEFAULT_TOOL_NAME = "g_tool_0"
"AR5 已验证默认工具坐标系名称。"

AR5_DEFAULT_WOBJ_NAME = "g_wobj_0"
"AR5 已验证默认工件坐标系名称。"


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

    def __init__(
        self,
        config: Ar5ConnectionConfig,
        *,
        initialize_toolset: bool = False,
    ) -> None:
        """连接并校验一台 AR5。

        Parameters
        ----------
        config:
            SDK 端点和预期左右侧配置。
        initialize_toolset:
            是否在连接阶段写入默认工具/工件坐标系。默认值 ``False``，控制客户端
            不因建立连接而改变控制器上下文；RecordReplay 等需要固定坐标系的业务
            必须显式传入 ``True`` 或执行自己的显式配置流程。

        Raises
        ------
        RuntimeError
            SDK 返回错误码或控制器型号与预期侧别不匹配。
        """

        self._config = config
        self._lock = RLock()
        self._event_lock = Lock()
        self._ec: dict[str, object] = {}
        self._soft_limit_cache: Ar5SoftLimitSnapshot | None = None
        self._motion_progress = Ar5MotionProgress()
        self._event_callbacks: dict[
            xCoreSDK_python.Event, Callable[[dict[str, object]], None]
        ] = {}
        logger.info(
            "AR5 SDK connection requested: side={} robot_ip={} local_ip={}",
            config.side,
            config.robot_ip,
            config.local_ip,
        )
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
        try:
            self._register_event_watchers_locked()
            if initialize_toolset:
                self._apply_default_toolset_locked()
        except Exception:
            logger.error(
                "AR5 SDK connection initialization failed: side={} robot_ip={}",
                config.side,
                config.robot_ip,
            )
            self.close()
            raise
        logger.success(
            "AR5 SDK connection ready: side={} robot_ip={} robot_type={} robot_uid={}",
            config.side,
            config.robot_ip,
            self._robot_type,
            self._robot_uid,
        )

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
            # pose_matrix_m: (4, 4) float64；直接使用 SDK 原始 m/rad 构造计算矩阵。
            pose_matrix_m = np.eye(4, dtype=np.float64)
            pose_matrix_m[:3, :3] = Rotation.from_euler(
                "xyz",
                sdk_rpy_rad,
                degrees=False,
            ).as_matrix()
            pose_matrix_m[:3, 3] = translation_m
            return Ar5Snapshot(
                robot_type=self._robot_type,
                robot_uid=self._robot_uid,
                operation_state=operation_state.name,
                operate_mode=operate_mode.name,
                power_state=power_state.name,
                joint_deg=tuple(math.degrees(float(value)) for value in joint_rad[:7]),
                pose_matrix_m=tuple(tuple(float(value) for value in row) for row in pose_matrix_m),
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
                conf_data=tuple(int(value) for value in cartesian_pose.confData),
            )

    def sdk_robot_info(self) -> tuple[str, str]:
        """返回原 SDK ``robotInfo`` 的型号与 UID。"""

        return self._robot_type, self._robot_uid

    def sdk_set_motion_control_mode_nrt(self) -> None:
        """一对一封装 ``setMotionControlMode(NrtCommandMode)``。"""

        with self._lock:
            self._call_none(
                "setMotionControlMode(NrtCommandMode)",
                lambda ec: self._robot.setMotionControlMode(
                    xCoreSDK_python.MotionControlMode.NrtCommandMode, ec
                ),
            )

    def sdk_set_operate_mode_automatic(self) -> None:
        """一对一封装 ``setOperateMode(automatic)``。"""

        with self._lock:
            self._call_none(
                "setOperateMode(automatic)",
                lambda ec: self._robot.setOperateMode(
                    xCoreSDK_python.OperateMode.automatic, ec
                ),
            )

    def sdk_set_power_state(self, enabled: bool) -> None:
        """一对一封装 ``setPowerState``，不附加模式切换。"""

        with self._lock:
            self._call_none(
                f"setPowerState({enabled})",
                lambda ec: self._robot.setPowerState(enabled, ec),
            )

    def sdk_set_default_conf_opt(self, enabled: bool) -> None:
        """一对一封装 ``setDefaultConfOpt``。"""

        with self._lock:
            self._call_none(
                f"setDefaultConfOpt({enabled})",
                lambda ec: self._robot.setDefaultConfOpt(enabled, ec),
            )

    def sdk_set_default_speed(self, speed_mm_s: float) -> None:
        """一对一封装 ``setDefaultSpeed``。"""

        with self._lock:
            self._call_none(
                f"setDefaultSpeed({speed_mm_s})",
                lambda ec: self._robot.setDefaultSpeed(speed_mm_s, ec),
            )

    def sdk_set_default_zone(self, zone_mm: float) -> None:
        """一对一封装 ``setDefaultZone``。"""

        with self._lock:
            self._call_none(
                f"setDefaultZone({zone_mm})",
                lambda ec: self._robot.setDefaultZone(zone_mm, ec),
            )

    def sdk_set_toolset(self, tool_name: str, wobj_name: str) -> None:
        """一对一封装命名 ``setToolset``。"""

        with self._lock:
            self._apply_named_toolset_locked(tool_name, wobj_name)

    def sdk_operation_state(self) -> str:
        """返回原 SDK ``operationState`` 的枚举名称。"""

        with self._lock:
            return self._call_value("operationState", self._robot.operationState).name

    def sdk_joint_position(self) -> tuple[float, ...]:
        """返回原 SDK ``jointPos`` 的七轴实时关节位置，单位 rad。"""

        with self._lock:
            joint_rad = tuple(
                float(value) for value in self._call_value("jointPos", self._robot.jointPos)
            )
        if len(joint_rad) != 7:
            raise RuntimeError(
                f"jointPos 返回关节数异常: expected=7 actual={len(joint_rad)}"
            )
        return joint_rad

    def sdk_operate_mode(self) -> str:
        """返回原 SDK ``operateMode`` 的枚举名称。"""

        with self._lock:
            return self._call_value("operateMode", self._robot.operateMode).name

    def sdk_power_state(self) -> str:
        """返回原 SDK ``powerState`` 的枚举名称。"""

        with self._lock:
            return self._call_value("powerState", self._robot.powerState).name

    def sdk_cart_posture(self) -> dict[str, object]:
        """返回原 SDK ``cartPosture(endInRef)`` 的显式可序列化字段。"""

        with self._lock:
            pose = self._call_value(
                "cartPosture(endInRef)",
                lambda ec: self._robot.cartPosture(xCoreSDK_python.endInRef, ec),
            )
            return {
                "trans_m": [float(value) for value in pose.trans],
                "rpy_rad": [float(value) for value in pose.rpy],
                "has_elbow": bool(pose.hasElbow),
                "elbow_rad": float(pose.elbow),
                "conf_data": [int(value) for value in pose.confData],
            }

    def sdk_calc_ik(
        self,
        *,
        trans_m: tuple[float, ...],
        rpy_rad: tuple[float, ...],
        has_elbow: bool,
        elbow_rad: float,
        conf_data: tuple[int, ...],
    ) -> tuple[float, ...]:
        """一对一封装当前 toolset 下的 ``model().calcIk``，返回 rad。"""

        with self._lock:
            pose = xCoreSDK_python.CartesianPosition(list(trans_m), list(rpy_rad))
            pose.hasElbow = has_elbow
            pose.elbow = elbow_rad
            pose.confData = list(conf_data)
            toolset = self._call_value("toolset", self._robot.toolset)
            result = self._call_value(
                "model().calcIk",
                lambda ec: self._robot.model().calcIk(pose, toolset, ec),
            )
            joints_rad = tuple(float(value) for value in result)
            if len(joints_rad) != 7:
                raise RuntimeError(
                    f"calcIk 返回关节数异常: expected=7 actual={len(joints_rad)}"
                )
            return joints_rad

    def sdk_move_reset(self) -> None:
        """一对一封装 ``moveReset``。"""

        with self._lock:
            self._call_none("moveReset", self._robot.moveReset)

    def sdk_move_append_abs_j(
        self, targets: tuple[tuple[tuple[float, ...], float, float], ...]
    ) -> Ar5MotionProgress:
        """一对一封装一组 ``MoveAbsJCommand`` 的 ``moveAppend``。"""

        if not targets:
            raise ValueError("moveAppend targets 不能为空")
        commands = []
        append_log: list[dict[str, object]] = []
        for target_index, (joints_rad, speed_mm_s, zone_mm) in enumerate(targets):
            if len(joints_rad) != 7:
                raise ValueError(f"MoveAbsJ 关节数必须为 7，实际为 {len(joints_rad)}")
            commands.append(
                xCoreSDK_python.MoveAbsJCommand(list(joints_rad), speed_mm_s, zone_mm)
            )
            append_log.append(
                {
                    "target_index": target_index,
                    "joints_rad": list(joints_rad),
                    "joints_deg": [math.degrees(value) for value in joints_rad],
                    "speed_mm_s": speed_mm_s,
                    "zone_mm": zone_mm,
                }
            )
        with self._lock:
            command_id = xCoreSDK_python.PyString()
            logger.info(
                "AR5 SDK moveAppend actual targets: side={} robot_ip={} targets={}",
                self._config.side,
                self._config.robot_ip,
                append_log,
            )
            self._call_none(
                f"moveAppend(MoveAbsJ,count={len(commands)})",
                lambda ec: self._robot.moveAppend(commands, command_id, ec),
            )
            progress = Ar5MotionProgress(
                command_id=str(command_id.content()),
                target_count=len(commands),
            )
            with self._event_lock:
                self._motion_progress = progress
        return progress

    def sdk_move_start(self) -> None:
        """一对一封装 ``moveStart``。"""

        with self._lock:
            self._call_none("moveStart", self._robot.moveStart)

    def sdk_motion_progress(self) -> Ar5MotionProgress:
        """返回事件回调维护的当前 NRT 路径进度快照。"""

        with self._event_lock:
            return self._motion_progress

    def sdk_clear_servo_alarm(self) -> None:
        """调用 ``clearServoAlarm``，成功后清除当前路径碰撞锁存。"""

        with self._lock:
            self._call_none("clearServoAlarm", self._robot.clearServoAlarm)
            with self._event_lock:
                self._motion_progress = replace(
                    self._motion_progress,
                    collision_detected=False,
                    collision_code=None,
                    collision_detail=None,
                )

    def sdk_disable_drag(self) -> None:
        """一对一封装 ``disableDrag``。"""

        with self._lock:
            self._call_none("disableDrag", self._robot.disableDrag)

    def read_soft_limits(self) -> Ar5SoftLimitSnapshot:
        """从 SDK 读取并覆盖当前客户端缓存的七轴软限位。

        Returns
        -------
        snapshot:
            七个轴的软限位上下限和控制器软限位使能状态。上下限单位为 rad。

        Raises
        ------
        RuntimeError
            xCoreSDK 返回错误，或控制器返回的软限位不是完整七轴数据。

        Notes
        -----
        每次调用均执行 xCoreSDK ``getSoftLimit`` 并覆盖缓存，供 RecordReplay 在每次
        ``moveStart`` 前获取最新快照；不会切换电源、工作模式或拖动状态。
        """

        with self._lock:
            self._soft_limit_cache = self._load_soft_limits_locked()
            logger.info(
                "AR5 soft limits refreshed and cached: side={} robot_ip={} enabled={} limits_rad={}",
                self._config.side,
                self._config.robot_ip,
                self._soft_limit_cache.enabled,
                self._soft_limit_cache.limits_rad,
            )
            return self._soft_limit_cache

    def _load_soft_limits_locked(self) -> Ar5SoftLimitSnapshot:
        """在已持锁状态下从 SDK 读取并校验一份软限位快照。"""

        limits = xCoreSDK_python.PyTypeVectorArrayDouble2()
        enabled = bool(
            self._call_value(
                "getSoftLimit",
                lambda ec: self._robot.getSoftLimit(limits, ec),
            )
        )
        raw_limits = limits.content()
        if len(raw_limits) != 7:
            raise RuntimeError(f"AR5 软限位数量异常: expected=7 actual={len(raw_limits)}")

        parsed_limits: list[tuple[float, float]] = []
        for axis_index, pair in enumerate(raw_limits):
            if len(pair) != 2:
                raise RuntimeError(
                    "AR5 软限位轴数据异常: "
                    f"axis_index={axis_index} expected=2 actual={len(pair)}"
                )
            lower_rad = float(pair[0])
            upper_rad = float(pair[1])
            if (
                not math.isfinite(lower_rad)
                or not math.isfinite(upper_rad)
                or lower_rad > upper_rad
            ):
                raise RuntimeError(
                    "AR5 软限位范围异常: "
                    f"axis_index={axis_index} lower_rad={lower_rad} upper_rad={upper_rad}"
                )
            parsed_limits.append((lower_rad, upper_rad))
        return Ar5SoftLimitSnapshot(enabled, tuple(parsed_limits))

    def _register_event_watchers_locked(self) -> None:
        """注册原始控制器日志和非实时运动执行事件回调。"""

        def handle_log_reporter(event_info: dict[str, object]) -> None:
            logger.error(
                "AR5 SDK raw controller log: side={} robot_ip={} event={}",
                self._config.side,
                self._config.robot_ip,
                event_info,
            )
            code = event_info.get("ecode")
            detail = event_info.get("edetail")
            if code == 30400 or detail == "30400.collision_fc":
                with self._event_lock:
                    self._motion_progress = replace(
                        self._motion_progress,
                        collision_detected=True,
                        collision_code=30400,
                        collision_detail="30400.collision_fc",
                    )

        def handle_move_execution(event_info: dict[str, object]) -> None:
            logger.info(
                "AR5 SDK raw move execution event: side={} robot_ip={} event={}",
                self._config.side,
                self._config.robot_ip,
                event_info,
            )
            command_id = event_info.get("cmdID")
            waypoint_index = event_info.get("wayPointIndex")
            reach_target = event_info.get("reachTarget")
            if (
                not isinstance(command_id, str)
                or isinstance(waypoint_index, bool)
                or not isinstance(waypoint_index, int)
                or reach_target is not True
            ):
                return
            with self._event_lock:
                progress = self._motion_progress
                if (
                    command_id != progress.command_id
                    or waypoint_index < 0
                    or waypoint_index >= progress.target_count
                ):
                    return
                last_reached = progress.last_reached_waypoint_index
                if last_reached is None or waypoint_index > last_reached:
                    self._motion_progress = replace(
                        progress,
                        last_reached_waypoint_index=waypoint_index,
                    )

        callbacks = {
            xCoreSDK_python.Event.logReporter: handle_log_reporter,
            xCoreSDK_python.Event.moveExecution: handle_move_execution,
        }
        for event_type, callback in callbacks.items():
            self._call_none(
                f"setEventWatcher({event_type.name})",
                lambda ec, selected_event=event_type, selected_callback=callback: (
                    self._robot.setEventWatcher(selected_event, selected_callback, ec)
                ),
            )
        self._event_callbacks = callbacks

    def configure_motion_context(self) -> None:
        """为后续运动显式配置已验证的默认工具与工件坐标系。

        Notes
        -----
        该方法会向控制器写入配置，只能由人工确认后的控制流程调用。只读状态路径
        不应调用本方法。
        """

        with self._lock:
            self._apply_default_toolset_locked()

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
            self._call_none(
                f"setPowerState({enabled})",
                lambda ec: self._robot.setPowerState(enabled, ec),
            )
            target_state = xCoreSDK_python.PowerState.on if enabled else xCoreSDK_python.PowerState.off
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

        target_mode = xCoreSDK_python.OperateMode.automatic if automatic else xCoreSDK_python.OperateMode.manual
        with self._lock:
            self._call_none(
                f"setOperateMode({target_mode.name})",
                lambda ec: self._robot.setOperateMode(target_mode, ec),
            )
            self._wait_operate_mode_locked(target_mode)

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
            SDK 拒绝切换工具、模式、使能或拖动状态。
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
            self._apply_default_toolset_locked()
            self._call_none(
                "setMotionControlMode(NrtCommandMode)",
                lambda ec: self._robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec),
            )
            self._call_none("setPowerState(False)", lambda ec: self._robot.setPowerState(False, ec))
            self._wait_power_state_locked(xCoreSDK_python.PowerState.off)
            self._call_none(
                "setOperateMode(manual)",
                lambda ec: self._robot.setOperateMode(
                    xCoreSDK_python.OperateMode.manual,
                    ec,
                ),
            )
            self._wait_operate_mode_locked(xCoreSDK_python.OperateMode.manual)
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
            for event_type in tuple(self._event_callbacks):
                try:
                    self._call_none(
                        f"setNoneEventWatcher({event_type.name})",
                        lambda ec, selected_event=event_type: (
                            self._robot.setNoneEventWatcher(selected_event, ec)
                        ),
                    )
                except Exception:
                    pass
            self._event_callbacks.clear()
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

    def _wait_operate_mode_locked(self, target_mode: object) -> None:
        """在已持锁状态下等待控制器确认目标工作模式。"""

        deadline = time.monotonic() + OPERATE_MODE_TIMEOUT_S
        while time.monotonic() < deadline:
            current_mode = self._call_value("operateMode", self._robot.operateMode)
            if current_mode == target_mode:
                return
            time.sleep(0.1)
        raise RuntimeError(f"等待工作模式超时: target={target_mode}")

    def _apply_default_toolset_locked(self) -> None:
        """固定已验证的工具和工件坐标系，避免双臂继承不同控制器上下文。"""

        self._call_control_value(
            (f"setToolset({AR5_DEFAULT_TOOL_NAME}, " f"{AR5_DEFAULT_WOBJ_NAME})"),
            lambda ec: self._robot.setToolset(
                AR5_DEFAULT_TOOL_NAME,
                AR5_DEFAULT_WOBJ_NAME,
                ec,
            ),
        )

    def _apply_named_toolset_locked(self, tool_name: str, wobj_name: str) -> None:
        """设置调用方显式给出的工具和工件坐标系。"""

        if not tool_name.strip() or not wobj_name.strip():
            raise ValueError("tool_name 与 wobj_name 不能为空")
        self._call_control_value(
            f"setToolset({tool_name}, {wobj_name})",
            lambda ec: self._robot.setToolset(tool_name, wobj_name, ec),
        )

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
        """调用无返回值 SDK 方法，并记录目标臂、耗时和完整错误码。"""

        started_at = time.monotonic()
        logger.info(
            "AR5 SDK request: side={} robot_ip={} action={}",
            self._config.side,
            self._config.robot_ip,
            action,
        )
        self._ec.clear()
        try:
            callback(self._ec)
            self._raise_for_error(action)
        except Exception:
            logger.error(
                "AR5 SDK request failed: side={} robot_ip={} action={} " "elapsed_ms={:.3f} ec={}",
                self._config.side,
                self._config.robot_ip,
                action,
                (time.monotonic() - started_at) * 1000.0,
                self._ec,
            )
            raise
        logger.info(
            "AR5 SDK response: side={} robot_ip={} action={} " "elapsed_ms={:.3f} ec={}",
            self._config.side,
            self._config.robot_ip,
            action,
            (time.monotonic() - started_at) * 1000.0,
            self._ec,
        )

    def _call_value(self, action: str, callback):  # noqa: ANN001, ANN202
        """调用只读或本地计算 SDK 方法；失败时记录 SDK 原始错误字典。"""

        self._ec.clear()
        try:
            value = callback(self._ec)
            self._raise_for_error(action)
        except Exception:
            logger.error(
                "AR5 SDK value request failed: side={} robot_ip={} action={} raw_ec={}",
                self._config.side,
                self._config.robot_ip,
                action,
                self._ec,
            )
            raise
        return value

    def _call_control_value(self, action: str, callback):  # noqa: ANN001, ANN202
        """调用带返回值的控制指令，并记录请求、响应、耗时和错误码。"""

        started_at = time.monotonic()
        logger.info(
            "AR5 SDK request: side={} robot_ip={} action={}",
            self._config.side,
            self._config.robot_ip,
            action,
        )
        self._ec.clear()
        try:
            value = callback(self._ec)
            self._raise_for_error(action)
        except Exception:
            logger.error(
                "AR5 SDK request failed: side={} robot_ip={} action={} " "elapsed_ms={:.3f} ec={}",
                self._config.side,
                self._config.robot_ip,
                action,
                (time.monotonic() - started_at) * 1000.0,
                self._ec,
            )
            raise
        logger.info(
            "AR5 SDK response: side={} robot_ip={} action={} " "elapsed_ms={:.3f} ec={} value={}",
            self._config.side,
            self._config.robot_ip,
            action,
            (time.monotonic() - started_at) * 1000.0,
            self._ec,
            _summarize_sdk_value(value),
        )
        return value

    def _raise_for_error(self, action: str) -> None:
        """把 SDK 错误码转换为带动作上下文的异常。"""

        error_code = self._ec.get("ec", 0)
        if error_code == 0:
            return
        raise RuntimeError(f"{action} 失败: ec={error_code}, message={self._ec.get('message', '')}")

    # endregion


def _summarize_sdk_value(value: object, max_length: int = 500) -> str:
    """生成适合日志记录的 SDK 返回值摘要。

    Parameters
    ----------
    value:
        SDK 返回对象，只读取其字符串表示，不改变对象内容。
    max_length:
        日志摘要最大字符数，避免状态对象异常膨胀日志。

    Returns
    -------
    str
        单行返回值摘要；超过上限时以省略号截断。
    """

    text = repr(value).replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}…"


# endregion
