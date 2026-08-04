"""GUI 使用的 RobotControl HTTP 适配器。

本模块只依赖 RobotControl 的 HTTP/SSE 合同，不创建 qmlinker channel、xCoreSDK
对象或 SSH 转发。状态读取统一来自 ``RobotControlStatusStream``，因此各页面不会
重复轮询完整 ``/status``。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import Condition, Event, Thread
from typing import Literal

from robot_control.service.client import RobotControlClient

Ar5Side = Literal["left", "right"]
DEFAULT_AR5_MOVE_SPEED_MM_S = 1000.0
DEFAULT_AR5_MOVE_ZONE_MM = 10.0


@dataclass(frozen=True, slots=True)
class Ar5Snapshot:
    """GUI 展示与标定使用的 AR5 状态快照。"""

    robot_type: str
    robot_uid: str
    operation_state: str
    operate_mode: str
    power_state: str
    joint_deg: tuple[float, ...]
    pose_matrix_m: tuple[tuple[float, ...], ...]
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    elbow_deg: float
    has_elbow: bool


@dataclass(frozen=True, slots=True)
class GripperInfo:
    """夹爪状态。"""

    online: bool
    calibrated: bool
    enable: bool
    position: int
    state: int


@dataclass(frozen=True, slots=True)
class RightHandActuatorSpec:
    """右手执行器的 GUI 轴定义。"""

    axis_name: str
    actuator_id: int
    label: str
    minimum: float = 0.0
    maximum: float = 1.0


class RobotControlStatusStream:
    """维护一个共享的 RobotControl 状态 SSE 连接。"""

    def __init__(self, client: RobotControlClient, interval_s: float = 0.2) -> None:
        self._client = client
        self._interval_s = interval_s
        self._condition = Condition()
        self._ready = Event()
        self._stop = Event()
        self._snapshot: dict[str, object] | None = None
        self._version = 0
        self._thread: Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        """启动共享 SSE 读取线程。"""

        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, name="robot-control-status-sse", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout_s: float) -> None:
        """等待第一帧状态；连接失败时抛出可读异常。"""

        if not self._ready.wait(timeout_s):
            if self._error is not None:
                raise RuntimeError(f"RobotControl 状态 SSE 失败：{self._error}") from self._error
            raise TimeoutError("RobotControl 状态 SSE 首帧超时")
        if self._error is not None and self._snapshot is None:
            raise RuntimeError(f"RobotControl 状态 SSE 失败：{self._error}") from self._error

    def snapshot(self) -> dict[str, object]:
        """返回最近一帧完整状态。"""

        with self._condition:
            if self._snapshot is None:
                if self._error is not None:
                    raise RuntimeError(f"RobotControl 状态不可用：{self._error}") from self._error
                raise RuntimeError("RobotControl 状态尚未就绪")
            return self._snapshot

    def iter_snapshots(self) -> Iterator[dict[str, object]]:
        """迭代共享 SSE 的后续状态帧，不创建第二条连接。"""

        version = 0
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._version > version or self._stop.is_set(),
                    timeout=1.0,
                )
                if self._stop.is_set():
                    return
                if self._snapshot is None or self._version <= version:
                    continue
                version = self._version
                snapshot = self._snapshot
            yield snapshot

    def close(self) -> None:
        """停止共享 SSE 线程。"""

        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            for payload in self._client.subscribe_status(self._interval_s):
                if self._stop.is_set():
                    return
                with self._condition:
                    self._snapshot = payload
                    self._version += 1
                    self._ready.set()
                    self._condition.notify_all()
        except Exception as exc:  # noqa: BLE001
            with self._condition:
                self._error = exc
                self._ready.set()
                self._condition.notify_all()


class _RobotControlDeviceClient:
    """共享状态流设备适配器的公共实现。"""

    def __init__(self, client: RobotControlClient, status_stream: RobotControlStatusStream) -> None:
        self._client = client
        self._status_stream = status_stream

    def _device_data(self, name: str) -> dict[str, object]:
        status = self._status_stream.snapshot()
        devices = status.get("devices")
        if not isinstance(devices, list):
            raise RuntimeError("RobotControl status.devices 不是数组")
        for item in devices:
            if not isinstance(item, dict) or item.get("name") != name:
                continue
            if item.get("connected") is not True:
                error = item.get("error") or "设备未连接"
                raise RuntimeError(f"{name}: {error}")
            data = item.get("data")
            if not isinstance(data, dict):
                raise RuntimeError(f"{name}: data 不是对象")
            return {str(key): value for key, value in data.items()}
        raise RuntimeError(f"RobotControl 未返回设备：{name}")


class RobotControlAr5Client(_RobotControlDeviceClient):
    """AR5 的 RobotControl HTTP 适配器。"""

    def __init__(self, side: Ar5Side, client: RobotControlClient, status_stream: RobotControlStatusStream) -> None:
        super().__init__(client, status_stream)
        self.side = side
        self._device_name = f"ar5_{side}"

    def read_snapshot(self) -> Ar5Snapshot:
        data = self._device_data(self._device_name)
        joints = _number_tuple(data.get("joint_deg"), 7, "joint_deg")
        xyz_values = _number_tuple(data.get("xyz_mm"), 3, "xyz_mm")
        rpy_values = _number_tuple(data.get("rpy_deg"), 3, "rpy_deg")
        matrix_value = data.get("pose_matrix_m")
        if not isinstance(matrix_value, list | tuple):
            raise RuntimeError("pose_matrix_m 不是矩阵")
        matrix = tuple(_number_tuple(row, 4, "pose_matrix_m row") for row in matrix_value)
        if len(matrix) != 4:
            raise RuntimeError("pose_matrix_m 必须是 4x4")
        return Ar5Snapshot(
            robot_type=_string_value(data, "robot_type"),
            robot_uid=_string_value(data, "robot_uid"),
            operation_state=_string_value(data, "operation_state"),
            operate_mode=_string_value(data, "operate_mode"),
            power_state=_string_value(data, "power_state"),
            joint_deg=joints,
            pose_matrix_m=matrix,
            xyz_mm=(xyz_values[0], xyz_values[1], xyz_values[2]),
            rpy_deg=(rpy_values[0], rpy_values[1], rpy_values[2]),
            elbow_deg=_float_value(data, "elbow_deg"),
            has_elbow=bool(data.get("has_elbow", True)),
        )

    def set_power(self, enabled: bool) -> dict[str, object]:
        return self._client.ar5_set_power(self.side, enabled)

    def set_operate_mode(self, automatic: bool) -> dict[str, object]:
        return self._client.ar5_set_operate_mode(self.side, automatic)

    def stop(self) -> dict[str, object]:
        return self._client.ar5_stop(self.side)

    def move_joints_deg(
        self,
        joint_deg: Sequence[float],
        *,
        speed_mm_s: float = DEFAULT_AR5_MOVE_SPEED_MM_S,
        zone_mm: float = DEFAULT_AR5_MOVE_ZONE_MM,
    ) -> dict[str, object]:
        return self._client.ar5_move_joints(self.side, tuple(joint_deg), speed_mm_s, zone_mm)

    def move_cartesian(
        self,
        xyz_mm: Sequence[float],
        rpy_deg: Sequence[float],
        elbow_deg: float,
        *,
        speed_mm_s: float = DEFAULT_AR5_MOVE_SPEED_MM_S,
        zone_mm: float = DEFAULT_AR5_MOVE_ZONE_MM,
    ) -> dict[str, object]:
        return self._client.ar5_move_cartesian(
            self.side,
            tuple(xyz_mm),
            tuple(rpy_deg),
            elbow_deg,
            speed_mm_s,
            zone_mm,
        )

    def move_elbow_deg(
        self,
        elbow_deg: float,
        *,
        speed_mm_s: float = DEFAULT_AR5_MOVE_SPEED_MM_S,
        zone_mm: float = DEFAULT_AR5_MOVE_ZONE_MM,
    ) -> dict[str, object]:
        return self._client.ar5_move_elbow(self.side, elbow_deg, speed_mm_s, zone_mm)


class RobotControlHeadClient(_RobotControlDeviceClient):
    """头部 qmlinker HTTP 适配器。"""

    def get_enable(self) -> bool:
        return bool(self._device_data("qmlinker_head").get("enabled"))

    def set_enable(self, enabled: bool) -> dict[str, object]:
        return self._client.qmlinker_set_head(enable=enabled)

    def get_head_yaw(self) -> float:
        return _float_value(self._device_data("qmlinker_head"), "yaw_deg")

    def set_head_yaw(self, value: float) -> dict[str, object]:
        return self._client.qmlinker_set_head(yaw_deg=value)

    def get_head_pitch(self) -> float:
        return _float_value(self._device_data("qmlinker_head"), "pitch_deg")

    def set_head_pitch(self, value: float) -> dict[str, object]:
        return self._client.qmlinker_set_head(pitch_deg=value)


class _RobotControlLiftClient(_RobotControlDeviceClient):
    """升降机构 HTTP 适配器。"""

    def get_enable(self) -> bool:
        return bool(self._device_data("qmlinker_lift").get("enabled"))

    def set_enable(self, enabled: bool) -> dict[str, object]:
        return self._client.qmlinker_set_lift(enable=enabled)

    def get_lift_height(self) -> tuple[float]:
        return (_float_value(self._device_data("qmlinker_lift"), "height_mm"),)

    def set_lift_physical_height(self, value: int) -> dict[str, object]:
        return self._client.qmlinker_set_lift(height_mm=float(value))


class RobotControlBodyClient:
    """仅暴露 RobotControl API 明确支持的升降能力，不提供腰部控制。"""

    def __init__(self, client: RobotControlClient, status_stream: RobotControlStatusStream) -> None:
        self.lift = _RobotControlLiftClient(client, status_stream)


class RobotControlGripperClient(_RobotControlDeviceClient):
    """夹爪 HTTP 适配器。"""

    def get_status(self) -> GripperInfo:
        data = self._device_data("qmlinker_gripper")
        return GripperInfo(
            online=bool(data.get("online")),
            calibrated=bool(data.get("calibrated")),
            enable=bool(data.get("enabled")),
            position=_int_value(data, "position"),
            state=_int_value(data, "state"),
        )

    def set_enable(self, enabled: bool) -> dict[str, object]:
        return self._client.set_gripper_enabled(enabled)

    def calibrate(self) -> dict[str, object]:
        return self._client.calibrate_gripper()

    def set_pos(self, position: int) -> dict[str, object]:
        return self._client.qmlinker_set_gripper_position(position)


class RobotControlAgvClient(_RobotControlDeviceClient):
    """AGV HTTP 适配器，仅保留状态、使能和目标点导航。"""

    def get_runtime_info(self) -> dict[str, object]:
        data = self._device_data("qmlinker_agv")
        runtime = data.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("qmlinker_agv.runtime 不是对象")
        return {str(key): value for key, value in runtime.items()}

    def try_get_enable(self) -> bool | None:
        data = self._device_data("qmlinker_agv")
        value = data.get("enabled")
        return value if isinstance(value, bool) else None

    def set_enable(self, enabled: bool) -> dict[str, object]:
        return self._client.set_agv_enabled(enabled)

    def navigate_to(self, target: str) -> dict[str, object]:
        return self._client.qmlinker_navigate_to(target)

    @staticmethod
    def get_navigation_targets() -> list[str]:
        return []


class RobotControlRightHandClient(_RobotControlDeviceClient):
    """右手 HTTP 适配器，状态流复用共享 SSE。"""

    def get_enable(self) -> bool:
        return bool(self._device_data("qmlinker_right_hand").get("enabled"))

    def set_enable(self, enabled: bool) -> dict[str, object]:
        return self._client.set_right_hand_enabled(enabled)

    def get_right_hand_instance_specs(self) -> tuple[RightHandActuatorSpec, ...]:
        count = _int_value(self._device_data("qmlinker_right_hand"), "actuator_count")
        if count <= 0:
            raise RuntimeError("右手执行器数量无效")
        return tuple(
            RightHandActuatorSpec(f"right_hand_a{index}", index, f"A{index}")
            for index in range(count)
        )

    def get_right_hand_values(self) -> dict[str, float]:
        positions = self._device_data("qmlinker_right_hand").get("positions")
        if isinstance(positions, dict):
            return {
                str(key): float(value)
                for key, value in positions.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        if isinstance(positions, list | tuple):
            return {f"right_hand_a{index}": float(value) for index, value in enumerate(positions)}
        raise RuntimeError("qmlinker_right_hand.positions 不是对象或数组")

    def stream_right_hand_values(self) -> Iterator[dict[str, float]]:
        try:
            for snapshot in self._status_stream.iter_snapshots():
                data = _device_data_from_status(snapshot, "qmlinker_right_hand")
                positions = data.get("positions")
                if isinstance(positions, dict):
                    yield {
                        str(key): float(value)
                        for key, value in positions.items()
                        if isinstance(value, int | float) and not isinstance(value, bool)
                    }
                elif isinstance(positions, list | tuple):
                    yield {
                        f"right_hand_a{index}": float(value)
                        for index, value in enumerate(positions)
                    }
        except GeneratorExit:
            # 页面切换停止状态流时，GeneratorExit 是正常的迭代器关闭信号。
            return

    def set_right_hand_axis(self, actuator_id: int, value: float) -> dict[str, object]:
        positions = self.get_right_hand_values()
        positions[f"right_hand_a{actuator_id}"] = value
        ordered = [positions[key] for key in sorted(positions, key=_right_hand_index)]
        return self._client.qmlinker_set_right_hand(ordered)


def _device_data_from_status(status: Mapping[str, object], name: str) -> dict[str, object]:
    devices = status.get("devices")
    if not isinstance(devices, list):
        raise RuntimeError("RobotControl status.devices 不是数组")
    for item in devices:
        if isinstance(item, dict) and item.get("name") == name:
            data = item.get("data")
            if isinstance(data, dict):
                return {str(key): value for key, value in data.items()}
    raise RuntimeError(f"RobotControl 未返回设备：{name}")


def _number_tuple(value: object, expected_length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != expected_length:
        raise RuntimeError(f"{name} 长度不是 {expected_length}")
    result = tuple(float(item) for item in value if isinstance(item, int | float) and not isinstance(item, bool))
    if len(result) != expected_length:
        raise RuntimeError(f"{name} 包含非数值字段")
    return result


def _string_value(data: Mapping[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str):
        raise RuntimeError(f"{name} 不是字符串")
    return value


def _float_value(data: Mapping[str, object], name: str) -> float:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"{name} 不是数值")
    return float(value)


def _int_value(data: Mapping[str, object], name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"{name} 不是整数")
    return int(value)


def _right_hand_index(value: str) -> int:
    suffix = value.removeprefix("right_hand_a")
    return int(suffix) if suffix.isdigit() else 0
