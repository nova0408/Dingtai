"""现场设备连接与只读状态诊断。"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from qmlinker import QMGripper, QMHead, QMLift
from sdk.xcoresdk import xCoreSDK_python

from .arm_gateway import retry_non_motion_call
from .settings import ReplayDeviceConnection, ReplayServiceSettings

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class ArmDeviceStatus:
    """单台机械臂的连接、身份与运行状态。"""

    connected: bool
    error: str | None
    ip: str
    expected_type: str
    robot_type: str | None = None
    robot_uid: str | None = None
    operate_mode: str | None = None
    operation_state: str | None = None
    power_state: str | None = None
    powered_on: bool | None = None


@dataclass(frozen=True, slots=True)
class GripperDeviceStatus:
    """左夹爪的连接、在线、标定与使能状态。"""

    connected: bool
    error: str | None
    online: bool | None = None
    calibrated: bool | None = None
    enabled: bool | None = None
    position: int | None = None
    state: str | None = None


@dataclass(frozen=True, slots=True)
class HeadDeviceStatus:
    """头部的连接、使能与当前角度状态。"""

    connected: bool
    error: str | None
    enabled: bool | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None


@dataclass(frozen=True, slots=True)
class LiftDeviceStatus:
    """升降机构的连接、使能与当前高度状态。"""

    connected: bool
    error: str | None
    enabled: bool | None = None
    height_mm: float | None = None


@dataclass(frozen=True, slots=True)
class DeviceStatusResponse:
    """一次现场设备只读诊断的完整结果。"""

    all_connected: bool
    left_arm: ArmDeviceStatus
    right_arm: ArmDeviceStatus
    gripper: GripperDeviceStatus
    head: HeadDeviceStatus
    lift: LiftDeviceStatus


class _RobotInfoProtocol(Protocol):
    """xCoreSDK robotInfo 诊断所需的最小字段。"""

    type: str
    id: str


class DeviceStatusReader:
    """通过 xCoreSDK 与 qmlinker 原生对象读取设备状态。"""

    def __init__(
        self,
        connection: ReplayDeviceConnection,
        settings: ReplayServiceSettings,
        gripper: QMGripper,
        head: QMHead,
        lift: QMLift,
    ) -> None:
        self._connection = connection
        self._settings = settings
        self._gripper = gripper
        self._head = head
        self._lift = lift

    def read(self) -> DeviceStatusResponse:
        """读取全部设备；单项失败不遮蔽其他设备结果。"""

        left_arm = self._read_arm(
            "left",
            self._connection.left_arm_ip,
            self._settings.arm.left_arm_type,
        )
        right_arm = self._read_arm(
            "right",
            self._connection.right_arm_ip,
            self._settings.arm.right_arm_type,
        )
        gripper = self._read_gripper()
        head = self._read_head()
        lift = self._read_lift()
        all_connected = all(
            item.connected
            for item in (left_arm, right_arm, gripper, head, lift)
        )
        return DeviceStatusResponse(all_connected, left_arm, right_arm, gripper, head, lift)

    def _read_arm(self, arm_side: str, robot_ip: str, expected_type: str) -> ArmDeviceStatus:
        """连接机械臂并校验 IP 对应的控制器型号。"""

        robot: xCoreSDK_python.xMateErProRobot | None = None
        robot_type: str | None = None
        robot_uid: str | None = None
        operate_mode: str | None = None
        operation_state: str | None = None
        power_state: str | None = None
        powered_on: bool | None = None
        ec: dict[str, object] = {}
        try:
            robot = xCoreSDK_python.xMateErProRobot(robot_ip)
            info = cast(
                _RobotInfoProtocol,
                self._arm_call(robot, ec, f"{arm_side}.robotInfo", lambda: robot.robotInfo(ec)),
            )
            robot_type = info.type
            robot_uid = info.id
            if robot_type != expected_type:
                raise RuntimeError(
                    f"机型不匹配 expected={expected_type!r} actual={robot_type!r}"
                )
            operate_mode_value = self._arm_call(
                robot,
                ec,
                f"{arm_side}.operateMode",
                lambda: robot.operateMode(ec),
            )
            operation_state_value = self._arm_call(
                robot,
                ec,
                f"{arm_side}.operationState",
                lambda: robot.operationState(ec),
            )
            power_state_value = self._arm_call(
                robot,
                ec,
                f"{arm_side}.powerState",
                lambda: robot.powerState(ec),
            )
            operate_mode = str(operate_mode_value)
            operation_state = str(operation_state_value)
            power_state = str(power_state_value)
            powered_on = power_state_value == xCoreSDK_python.PowerState.on
            return ArmDeviceStatus(
                True,
                None,
                robot_ip,
                expected_type,
                robot_type,
                robot_uid,
                operate_mode,
                operation_state,
                power_state,
                powered_on,
            )
        except Exception as exc:
            return ArmDeviceStatus(
                False,
                f"{type(exc).__name__}: {exc}",
                robot_ip,
                expected_type,
                robot_type,
                robot_uid,
                operate_mode,
                operation_state,
                power_state,
                powered_on,
            )
        finally:
            if robot is not None:
                try:
                    robot.disconnectFromRobot({})
                except Exception:
                    pass

    def _arm_call(
        self,
        robot: xCoreSDK_python.xMateErProRobot,
        ec: dict[str, object],
        label: str,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        """读取机械臂并检查 SDK 错误上下文。"""

        del robot
        ec.clear()
        result = retry_non_motion_call(
            label,
            operation,
            self._settings.non_motion_retry_count,
            self._settings.non_motion_retry_delay_s,
        )
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"{label} SDK 错误：{ec}")
        return result

    def _read_gripper(self) -> GripperDeviceStatus:
        """读取夹爪原始状态，避免第三方 GripperInfo 字段不一致。"""

        try:
            payload = self._retry(
                "gripper.status",
                lambda: self._gripper._send_control(QMGripper.STATUS),
            )
            if not isinstance(payload, Mapping):
                raise TypeError(f"夹爪状态必须是 mapping，实际为 {type(payload).__name__}")
            online = bool(payload.get("online", True))
            if not online:
                raise RuntimeError("夹爪返回 online=false")
            return GripperDeviceStatus(
                connected=True,
                error=None,
                online=online,
                calibrated=bool(payload.get("calibrated", False)),
                enabled=bool(payload.get("enable", False)),
                position=self._optional_int(payload.get("position")),
                state=str(payload.get("state", "")),
            )
        except Exception as exc:
            return GripperDeviceStatus(False, f"{type(exc).__name__}: {exc}")

    def _read_head(self) -> HeadDeviceStatus:
        """读取头部使能、yaw 与 pitch。"""

        try:
            enabled = self._require_bool(self._retry("head.get_enable", self._head.get_enable))
            yaw_deg = self._require_finite_number(
                self._retry("head.get_head_yaw", self._head.get_head_yaw),
                "head yaw",
            )
            pitch_deg = self._require_finite_number(
                self._retry("head.get_head_pitch", self._head.get_head_pitch),
                "head pitch",
            )
            return HeadDeviceStatus(True, None, enabled, yaw_deg, pitch_deg)
        except Exception as exc:
            return HeadDeviceStatus(False, f"{type(exc).__name__}: {exc}")

    def _read_lift(self) -> LiftDeviceStatus:
        """读取升降使能与物理高度。"""

        try:
            enabled = self._require_bool(self._retry("lift.get_enable", self._lift.get_enable))
            height_mm = self._lift_height(
                self._retry("lift.get_lift_physical_height", self._lift.get_lift_physical_height)
            )
            if height_mm < 0.0:
                raise RuntimeError(f"lift 返回无效高度 {height_mm:.1f} mm")
            return LiftDeviceStatus(True, None, enabled, height_mm)
        except Exception as exc:
            return LiftDeviceStatus(False, f"{type(exc).__name__}: {exc}")

    def _retry(self, label: str, operation: Callable[[], _ResultT]) -> _ResultT:
        """按服务统一重试参数执行只读 qmlinker 调用。"""

        return retry_non_motion_call(
            label,
            operation,
            self._settings.non_motion_retry_count,
            self._settings.non_motion_retry_delay_s,
        )

    @staticmethod
    def _require_bool(value: object) -> bool:
        """校验设备使能状态为明确布尔值。"""

        if not isinstance(value, bool):
            raise TypeError(f"使能状态必须是 bool，实际为 {value!r}")
        return value

    @staticmethod
    def _require_finite_number(value: object, label: str) -> float:
        """校验状态数值为有限数。"""

        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{label} 必须是数值，实际为 {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} 必须是有限数，实际为 {value!r}")
        return number

    @classmethod
    def _lift_height(cls, value: object) -> float:
        """解析 qmlinker lift 标量或二元返回值。"""

        if isinstance(value, tuple) and len(value) == 2:
            return cls._require_finite_number(value[0], "lift height")
        return cls._require_finite_number(value, "lift height")

    @staticmethod
    def _optional_int(value: object) -> int | None:
        """解析可选夹爪整数状态。"""

        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"夹爪位置必须是数值，实际为 {value!r}")
        return int(value)
