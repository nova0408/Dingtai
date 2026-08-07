"""qmlinker 与 AR5 的统一设备网关。

本模块只负责第三方客户端的生命周期、状态读取和显式控制方法，不负责 HTTP 路由。
硬件对象采用延迟创建，导入模块和构造网关均不会连接现场设备。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from .config import RobotControlSettings
from .protocol import API_VERSION, DeviceState, JsonValue, RobotControlStatus
from . import ROBOT_CONTROL_VERSION


class RobotControlGateway:
    """持有 qmlinker 与 AR5 客户端的统一设备网关。

    职责边界：
    - 只在第一次 GET 或人工控制请求到达时创建对应硬件客户端。
    - 将状态读取和控制参数收窄为项目侧明确的 Python 类型。
    - 用一把进程内锁串行化跨设备请求，避免服务内并发访问第三方 SDK。

    不负责：
    - HTTP 请求解析、认证和路由。
    - 急停安全回路或动作完成后的物理安全判断。

    生命周期：
    - 通常由一个 HTTP 服务进程持有一份。
    - ``close`` 只释放本进程创建的 SDK/channel 资源，不替代现场急停。
    """

    def __init__(self, settings: RobotControlSettings) -> None:
        """创建尚未连接硬件的网关。"""

        self._settings = settings
        self._lock = threading.RLock()
        self._qmlinker_channel: Any | None = None
        self._agv_channel: Any | None = None
        self._gripper_channel: Any | None = None
        self._qmlinker_clients: dict[str, Any] = {}
        self._ar5_clients: dict[str, Any] = {}

    # region 只读状态

    def read_status(self) -> RobotControlStatus:
        """读取 qmlinker 与 AR5 的完整只读状态。

        Returns
        -------
        RobotControlStatus
            每个设备单独包含连接状态和错误摘要；单项失败不会中止其它设备读取。

        Notes
        -----
        本方法只调用状态读取接口。AR5 使用 ``initialize_toolset=False``，不会因 GET
        请求写入工具或工件坐标系。
        """

        devices = [
            self._read_device("qmlinker_head", "qmlinker", self._read_qmlinker_head),
            self._read_device("qmlinker_lift", "qmlinker", self._read_qmlinker_lift),
            self._read_device(
                "qmlinker_gripper", "qmlinker", self._read_qmlinker_gripper
            ),
            self._read_device(
                "qmlinker_right_hand", "qmlinker", self._read_qmlinker_right_hand
            ),
            self._read_device("qmlinker_agv", "qmlinker", self._read_qmlinker_agv),
            self._read_device("ar5_left", "xcoresdk", lambda: self._read_ar5("left")),
            self._read_device("ar5_right", "xcoresdk", lambda: self._read_ar5("right")),
        ]
        if self._settings.qmlinker_waist_available:
            devices.insert(
                2,
                self._read_device(
                    "qmlinker_waist", "qmlinker", self._read_qmlinker_waist
                ),
            )
        return RobotControlStatus(ROBOT_CONTROL_VERSION, API_VERSION, tuple(devices))

    def _read_device(
        self,
        name: str,
        backend: str,
        reader: Callable[[], dict[str, JsonValue]],
    ) -> DeviceState:
        """隔离单个设备的只读异常。"""

        try:
            with self._lock:
                return DeviceState(name, backend, True, None, reader())
        except Exception as exc:
            return DeviceState(name, backend, False, f"{type(exc).__name__}: {exc}", {})

    def _read_qmlinker_head(self) -> dict[str, JsonValue]:
        """读取 qmlinker 头部状态。"""

        client = self._qmlinker_client("head")
        return {
            "enabled": bool(client.get_enable()),
            "yaw_deg": float(client.get_head_yaw()),
            "pitch_deg": float(client.get_head_pitch()),
        }

    def _read_qmlinker_lift(self) -> dict[str, JsonValue]:
        """读取 qmlinker 升降状态。"""

        lift = self._qmlinker_client("body").lift
        height = lift.get_lift_physical_height()
        if isinstance(height, tuple):
            height_mm = float(height[0])
        else:
            height_mm = float(height)
        return {"enabled": bool(lift.get_enable()), "height_mm": height_mm}

    def _read_qmlinker_waist(self) -> dict[str, JsonValue]:
        """读取 qmlinker 腰部 Pitch 状态；不提供腰部控制接口。"""

        waist = self._qmlinker_client("body").waist
        pitch_deg = waist.get_waist_pitch()
        if pitch_deg is None:
            raise RuntimeError("waist pitch unavailable")
        return {
            "enabled": bool(waist.get_enable()),
            "pitch_deg": float(pitch_deg),
        }

    def _read_qmlinker_gripper(self) -> dict[str, JsonValue]:
        """读取 qmlinker 夹爪状态。"""

        info = self._qmlinker_client("gripper").get_status()
        return {
            "online": bool(info.online),
            "calibrated": bool(info.calibrated),
            "enabled": bool(info.enable),
            "position": int(info.position),
            "state": int(info.state),
        }

    def _read_qmlinker_right_hand(self) -> dict[str, JsonValue]:
        """读取 qmlinker 右手执行器状态。"""

        client = self._qmlinker_client("right_hand")
        actuator_specs = client.get_right_hand_instance_specs()
        values = client.get_right_hand_values()
        expected_names = tuple(spec.axis_name for spec in actuator_specs)
        expected_name_set = set(expected_names)
        missing_names = tuple(name for name in expected_names if name not in values)
        unexpected_names = tuple(name for name in values if name not in expected_name_set)
        if missing_names or unexpected_names or len(values) != len(expected_names):
            raise RuntimeError(
                "右手执行器状态不完整: "
                f"expected={len(expected_names)} actual={len(values)} "
                f"missing={list(missing_names)} unexpected={list(unexpected_names)}"
            )
        return {
            "actuator_count": len(expected_names),
            "enabled": bool(client.get_enable()),
            "positions": {name: float(values[name]) for name in expected_names},
        }

    def _read_qmlinker_agv(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 状态。"""

        client = self._qmlinker_client("agv")
        runtime_info = client.get_runtime_info()
        return {
            "enabled": client.try_get_enable(),
            "runtime": {
                key: cast(JsonValue, value) for key, value in runtime_info.items()
            },
        }

    def read_qmlinker_agv_navigation_map(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 当前地图和可导航目标点。"""

        with self._lock:
            navigation_map = self._qmlinker_client("agv").get_navigation_map()
        return {
            "map": {
                "name": navigation_map.name,
                "id": navigation_map.id,
                "resolution": navigation_map.resolution,
            },
            "targets": [
                {
                    "name": target.name,
                    "id": target.id,
                    "x_m": target.x_m,
                    "y_m": target.y_m,
                    "yaw_rad": target.yaw_rad,
                }
                for target in navigation_map.targets
            ],
        }

    def read_qmlinker_agv_base_state(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 底盘状态。"""

        with self._lock:
            return cast(
                dict[str, JsonValue],
                self._qmlinker_client("agv").get_base_state(),
            )

    def read_qmlinker_agv_base_mode(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 底盘控制模式和工作模式。"""

        with self._lock:
            return cast(
                dict[str, JsonValue],
                self._qmlinker_client("agv").get_base_mode(),
            )

    def read_qmlinker_agv_base_operation_state(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 底盘原始运行状态位。"""

        with self._lock:
            return cast(
                dict[str, JsonValue],
                self._qmlinker_client("agv").get_base_operation_state(),
            )

    def read_qmlinker_agv_base_task_process(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 当前任务和动作进度。"""

        with self._lock:
            return cast(
                dict[str, JsonValue],
                self._qmlinker_client("agv").get_base_task_process(),
            )

    def read_qmlinker_agv_base_battery(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 电量和充电状态。"""

        with self._lock:
            return cast(
                dict[str, JsonValue],
                self._qmlinker_client("agv").get_base_battery(),
            )

    def _read_ar5(self, side: str) -> dict[str, JsonValue]:
        """读取结构化 AR5 状态，按关节、TCP、臂角和控制器状态分组。"""

        snapshot = self._ar5_client(side).read_snapshot()
        return {
            "identity": {
                "robot_type": snapshot.robot_type,
                "robot_uid": snapshot.robot_uid,
            },
            "joints": {
                "count": len(snapshot.joint_deg),
                "angle_deg": list(snapshot.joint_deg),
            },
            "tcp": {
                "pose_matrix_m": [list(row) for row in snapshot.pose_matrix_m],
                "xyz_mm": list(snapshot.xyz_mm),
                "rpy_deg": list(snapshot.rpy_deg),
            },
            "elbow": {
                "angle_deg": snapshot.elbow_deg,
                "available": snapshot.has_elbow,
            },
            "status": {
                "operation_state": snapshot.operation_state,
                "operate_mode": snapshot.operate_mode,
                "power_state": snapshot.power_state,
            },
        }

    def read_ar5_soft_limits(self, side: str) -> dict[str, JsonValue]:
        """读取指定 AR5 七个轴的软限位上下限和使能状态。"""

        with self._lock:
            snapshot = self._ar5_client(side).read_soft_limits()
        return {
            "side": side,
            "enabled": snapshot.enabled,
            "axis_count": len(snapshot.limits_rad),
            "limits_rad": [
                {
                    "axis_index": axis_index,
                    "lower_rad": lower_rad,
                    "upper_rad": upper_rad,
                }
                for axis_index, (lower_rad, upper_rad) in enumerate(snapshot.limits_rad)
            ],
        }

    # endregion

    # region qmlinker 控制

    def qmlinker_set_head(
        self, *, enable: bool | None, yaw_deg: float | None, pitch_deg: float | None
    ) -> None:
        """显式设置 qmlinker 头部使能或角度。"""

        client = self._qmlinker_client("head")
        if enable is not None:
            client.set_enable(enable)
        if yaw_deg is not None:
            client.set_head_yaw(float(yaw_deg))
        if pitch_deg is not None:
            client.set_head_pitch(float(pitch_deg))

    def qmlinker_set_lift(
        self, *, enable: bool | None, height_mm: float | None
    ) -> None:
        """显式设置 qmlinker 升降使能或目标高度。"""

        lift = self._qmlinker_client("body").lift
        if enable is not None:
            lift.set_enable(enable)
        if height_mm is not None:
            lift.set_lift_physical_height(float(height_mm))

    def qmlinker_set_gripper_position(self, position: int) -> None:
        """下发左夹爪位置。"""

        self._qmlinker_client("gripper").set_pos(int(position))

    def qmlinker_set_gripper_enabled(self, enabled: bool) -> None:
        """设置左夹爪使能状态。"""

        self._qmlinker_client("gripper").set_enable(bool(enabled))

    def qmlinker_calibrate_gripper(self) -> None:
        """请求左夹爪校准。"""

        self._qmlinker_client("gripper").calibrate()

    def qmlinker_set_right_hand(self, positions: Sequence[float]) -> None:
        """下发右手归一化执行器位置。"""

        self._qmlinker_client("right_hand").set_hand_state(
            tuple(float(value) for value in positions)
        )

    def qmlinker_set_right_hand_enabled(self, enabled: bool) -> None:
        """设置右手使能状态。"""

        self._qmlinker_client("right_hand").set_enable(bool(enabled))

    def qmlinker_navigate_to(self, target: str) -> None:
        """请求 AGV 导航到指定目标点。"""

        self._qmlinker_client("agv").navigate_to(str(target))

    def qmlinker_set_agv_enabled(self, enabled: bool) -> None:
        """设置 AGV 使能状态。"""

        self._qmlinker_client("agv").set_enable(bool(enabled))

    def qmlinker_translate_agv(self, speed_mps: float, direction_deg: float) -> None:
        """请求 AGV 持续平移；停止必须显式调用 ``qmlinker_stop_agv``。"""

        if speed_mps <= 0.0:
            raise ValueError("AGV translate speed_mps must be positive")
        self._qmlinker_client("agv").real_time_translate(
            float(speed_mps),
            float(direction_deg),
        )

    def qmlinker_stop_agv(self) -> None:
        """停止 AGV 当前导航或实时移动请求；不等同硬件急停。"""

        self._qmlinker_client("agv").stop()

    # endregion

    # region AR5 控制

    def ar5_set_power(self, side: str, enabled: bool) -> None:
        """设置 AR5 电机上下电状态。"""

        self._ar5_client(side).set_power(bool(enabled))

    def ar5_set_operate_mode(self, side: str, automatic: bool) -> None:
        """设置 AR5 手动或自动工作模式。"""

        self._ar5_client(side).set_operate_mode(bool(automatic))

    def ar5_recover_estop(self, side: str) -> None:
        """请求 AR5 控制器恢复急停状态；不自动上电。"""

        self._ar5_client(side).recover_estop()

    def ar5_set_drag_enabled(self, side: str, enabled: bool) -> None:
        """设置 AR5 笛卡尔自由拖动状态。"""

        self._ar5_client(side).set_drag_enabled(bool(enabled))

    def ar5_start_jog(
        self,
        side: str,
        space: str,
        axis_index: int,
        direction_positive: bool,
        rate: float,
        step: float,
    ) -> None:
        """启动 AR5 单轴 Jog。"""

        if space not in {"cartesian", "joint"}:
            raise ValueError(f"不支持的 AR5 Jog 空间：{space}")
        self._ar5_client(side).start_jog(
            cast(Literal["cartesian", "joint"], space),
            int(axis_index),
            bool(direction_positive),
            rate=float(rate),
            step=float(step),
        )

    def ar5_move_joints(
        self, side: str, joint_deg: Sequence[float], speed_mm_s: float, zone_mm: float
    ) -> None:
        """执行 AR5 七关节绝对运动，角度单位 deg。"""

        client = self._ar5_client(side)
        client.move_joints_deg(
            tuple(float(value) for value in joint_deg),
            speed_mm_s=speed_mm_s,
            zone_mm=zone_mm,
        )

    def ar5_move_cartesian(
        self,
        side: str,
        xyz_mm: Sequence[float],
        rpy_deg: Sequence[float],
        elbow_deg: float,
        speed_mm_s: float,
        zone_mm: float,
    ) -> None:
        """执行 AR5 笛卡尔直线运动，平移单位 mm，姿态单位 deg。"""

        client = self._ar5_client(side)
        client.move_cartesian(
            tuple(float(value) for value in xyz_mm),
            tuple(float(value) for value in rpy_deg),
            float(elbow_deg),
            speed_mm_s=float(speed_mm_s),
            zone_mm=float(zone_mm),
        )

    def ar5_move_elbow(
        self, side: str, elbow_deg: float, speed_mm_s: float, zone_mm: float
    ) -> None:
        """保持 AR5 TCP 并调整臂角，角度单位 deg。"""

        client = self._ar5_client(side)
        client.move_elbow_deg(float(elbow_deg), speed_mm_s=speed_mm_s, zone_mm=zone_mm)

    def ar5_stop(self, side: str) -> None:
        """请求停止 AR5 当前运动或 Jog。"""

        self._ar5_client(side).stop()

    # endregion

    # region 客户端生命周期

    def close(self) -> None:
        """释放本进程创建的 SDK 和 qmlinker channel。"""

        with self._lock:
            for client in self._ar5_clients.values():
                client.close()
            self._ar5_clients.clear()
            for channel in (
                self._qmlinker_channel,
                self._agv_channel,
                self._gripper_channel,
            ):
                if channel is None:
                    continue
                if isinstance(channel, dict):
                    for item in channel.values():
                        item.close()
                else:
                    channel.close()
            self._qmlinker_channel = None
            self._agv_channel = None
            self._gripper_channel = None
            self._qmlinker_clients.clear()

    # endregion

    # region 客户端创建与校验

    def _qmlinker_client(self, name: str) -> Any:
        """按固定名称返回 qmlinker 客户端。"""

        with self._lock:
            self._ensure_qmlinker_clients()
            return self._qmlinker_clients[name]

    def _ensure_qmlinker_clients(self) -> None:
        """延迟创建全部 qmlinker 客户端。"""

        if self._qmlinker_clients:
            return
        from qmlinker import create_channel

        from src.wuji.agv_client import WujiAgvClient
        from src.wuji.body_client import WujiBodyClient
        from src.wuji.dahuan_gripper_client import DahuanGripperClient
        from src.wuji.head_client import WujiHeadClient
        from src.wuji.right_hand_client import WujiRightHandClient

        self._qmlinker_channel = create_channel(
            f"{self._settings.qmlinker_host}:{self._settings.qmlinker_port}"
        )
        self._agv_channel = create_channel(
            f"{self._settings.agv_host}:{self._settings.qmlinker_port}"
        )
        self._gripper_channel = create_channel(
            f"{self._settings.qmlinker_host}:{self._settings.gripper_port}"
        )
        self._qmlinker_clients.update(
            {
                "body": WujiBodyClient(self._qmlinker_channel),
                "head": WujiHeadClient(self._qmlinker_channel),
                "gripper": DahuanGripperClient(self._gripper_channel),
                "right_hand": WujiRightHandClient(self._qmlinker_channel),
                "agv": WujiAgvClient(self._agv_channel),
            }
        )

    def _ar5_client(self, side: str) -> Any:
        """延迟创建并返回指定侧 AR5 客户端。"""

        if side not in {"left", "right"}:
            raise ValueError(f"不支持的 AR5 侧别：{side}")
        with self._lock:
            client = self._ar5_clients.get(side)
            if client is not None:
                return client
            from src.wuji.ar5_client import Ar5Client, Ar5ConnectionConfig

            robot_ip = (
                self._settings.left_ar5_ip
                if side == "left"
                else self._settings.right_ar5_ip
            )
            client = Ar5Client(
                Ar5ConnectionConfig(cast(Any, side), robot_ip),
                initialize_toolset=False,
            )
            self._ar5_clients[side] = client
            return client

    # endregion
