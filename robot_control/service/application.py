"""把 HTTP 请求桥接到 RobotControl 网关。"""

from __future__ import annotations

from collections.abc import Sequence

from ..gateway import RobotControlGateway
from ..protocol import ActionResponse, JsonValue, RobotControlStatus, health_payload
from .. import ROBOT_CONTROL_VERSION


class RobotControlApplication:
    """统一管理 qmlinker 与 AR5 的 HTTP 业务边界。

    控制方法只提供服务能力，不由本类自动调用。Codex 的验证范围限定为健康检查和
    GET 状态读取；任何 POST 控制请求必须由现场人员手动发起。
    """

    def __init__(self, gateway: RobotControlGateway) -> None:
        """创建应用门面。"""

        self._gateway = gateway

    def health(self) -> dict[str, JsonValue]:
        """返回不访问硬件的服务健康状态。"""

        return health_payload()

    def status(self) -> RobotControlStatus:
        """读取全部设备状态。"""

        return self._gateway.read_status()

    def qmlinker_get_agv_targets(self) -> dict[str, JsonValue]:
        """读取 qmlinker AGV 当前地图和可导航目标点。"""

        return self._gateway.read_qmlinker_agv_navigation_map()

    def qmlinker_set_joints(
        self, device_name: str, joint_deg: Sequence[float]
    ) -> ActionResponse:
        """转发 qmlinker 整臂关节控制请求。"""

        self._gateway.qmlinker_set_joints(device_name, joint_deg)
        return self._accepted("qmlinker_set_joints", device_name)

    def qmlinker_set_joint(
        self, device_name: str, joint_index: int, target_angle_deg: float
    ) -> ActionResponse:
        """转发 qmlinker 单关节控制请求。"""

        self._gateway.qmlinker_set_joint(device_name, joint_index, target_angle_deg)
        return self._accepted("qmlinker_set_joint", device_name)

    def qmlinker_set_head(
        self, enable: bool | None, yaw_deg: float | None, pitch_deg: float | None
    ) -> ActionResponse:
        """转发 qmlinker 头部控制请求。"""

        if enable is None and yaw_deg is None and pitch_deg is None:
            raise ValueError("head 请求至少需要一个控制字段")
        self._gateway.qmlinker_set_head(
            enable=enable, yaw_deg=yaw_deg, pitch_deg=pitch_deg
        )
        return self._accepted("qmlinker_set_head", "head")

    def qmlinker_set_lift(
        self, enable: bool | None, height_mm: float | None
    ) -> ActionResponse:
        """转发 qmlinker 升降控制请求。"""

        if enable is None and height_mm is None:
            raise ValueError("lift 请求至少需要一个控制字段")
        self._gateway.qmlinker_set_lift(enable=enable, height_mm=height_mm)
        return self._accepted("qmlinker_set_lift", "lift")

    def qmlinker_set_gripper_position(self, position: int) -> ActionResponse:
        """转发左夹爪控制请求。"""

        self._gateway.qmlinker_set_gripper_position(position)
        return self._accepted("qmlinker_set_gripper_position", "gripper")

    def qmlinker_set_gripper_enabled(self, enabled: bool) -> ActionResponse:
        """转发左夹爪使能请求。"""

        self._gateway.qmlinker_set_gripper_enabled(enabled)
        return self._accepted("qmlinker_set_gripper_enabled", "gripper")

    def qmlinker_calibrate_gripper(self) -> ActionResponse:
        """转发左夹爪校准请求。"""

        self._gateway.qmlinker_calibrate_gripper()
        return self._accepted("qmlinker_calibrate_gripper", "gripper")

    def qmlinker_set_right_hand(self, positions: Sequence[float]) -> ActionResponse:
        """转发右手控制请求。"""

        self._gateway.qmlinker_set_right_hand(positions)
        return self._accepted("qmlinker_set_right_hand", "right_hand")

    def qmlinker_set_right_hand_enabled(self, enabled: bool) -> ActionResponse:
        """转发右手使能请求。"""

        self._gateway.qmlinker_set_right_hand_enabled(enabled)
        return self._accepted("qmlinker_set_right_hand_enabled", "right_hand")

    def qmlinker_navigate_to(self, target: str) -> ActionResponse:
        """转发 AGV 导航控制请求。"""

        self._gateway.qmlinker_navigate_to(target)
        return self._accepted("qmlinker_navigate_to", "agv")

    def qmlinker_set_agv_enabled(self, enabled: bool) -> ActionResponse:
        """转发 AGV 使能请求。"""

        self._gateway.qmlinker_set_agv_enabled(enabled)
        return self._accepted("qmlinker_set_agv_enabled", "agv")

    def qmlinker_translate_agv(
        self, speed_mps: float, direction_deg: float
    ) -> ActionResponse:
        """转发 AGV 持续平移请求。"""

        self._gateway.qmlinker_translate_agv(speed_mps, direction_deg)
        return self._accepted("qmlinker_translate_agv", "agv")

    def qmlinker_stop_agv(self) -> ActionResponse:
        """转发 AGV 实时停止请求。"""

        self._gateway.qmlinker_stop_agv()
        return self._accepted("qmlinker_stop_agv", "agv")

    def ar5_set_power(self, side: str, enabled: bool) -> ActionResponse:
        """转发 AR5 上下电请求。"""

        self._gateway.ar5_set_power(side, enabled)
        return self._accepted("ar5_set_power", f"ar5_{side}")

    def ar5_set_operate_mode(self, side: str, automatic: bool) -> ActionResponse:
        """转发 AR5 工作模式请求。"""

        self._gateway.ar5_set_operate_mode(side, automatic)
        return self._accepted("ar5_set_operate_mode", f"ar5_{side}")

    def ar5_recover_estop(self, side: str) -> ActionResponse:
        """转发 AR5 急停恢复请求。"""

        self._gateway.ar5_recover_estop(side)
        return self._accepted("ar5_recover_estop", f"ar5_{side}")

    def ar5_set_drag_enabled(self, side: str, enabled: bool) -> ActionResponse:
        """转发 AR5 拖动开关请求。"""

        self._gateway.ar5_set_drag_enabled(side, enabled)
        return self._accepted("ar5_set_drag_enabled", f"ar5_{side}")

    def ar5_start_jog(
        self,
        side: str,
        space: str,
        axis_index: int,
        direction_positive: bool,
        rate: float,
        step: float,
    ) -> ActionResponse:
        """转发 AR5 单轴 Jog 请求。"""

        self._gateway.ar5_start_jog(
            side,
            space,
            axis_index,
            direction_positive,
            rate,
            step,
        )
        return self._accepted("ar5_start_jog", f"ar5_{side}")

    def ar5_move_joints(
        self, side: str, joint_deg: Sequence[float], speed_mm_s: float, zone_mm: float
    ) -> ActionResponse:
        """转发 AR5 七关节运动请求。"""

        self._gateway.ar5_move_joints(side, joint_deg, speed_mm_s, zone_mm)
        return self._accepted("ar5_move_joints", f"ar5_{side}")

    def ar5_move_cartesian(
        self,
        side: str,
        xyz_mm: Sequence[float],
        rpy_deg: Sequence[float],
        elbow_deg: float,
        speed_mm_s: float,
        zone_mm: float,
    ) -> ActionResponse:
        """转发 AR5 笛卡尔运动请求。"""

        self._gateway.ar5_move_cartesian(
            side, xyz_mm, rpy_deg, elbow_deg, speed_mm_s, zone_mm
        )
        return self._accepted("ar5_move_cartesian", f"ar5_{side}")

    def ar5_move_elbow(
        self, side: str, elbow_deg: float, speed_mm_s: float, zone_mm: float
    ) -> ActionResponse:
        """转发 AR5 臂角运动请求。"""

        self._gateway.ar5_move_elbow(side, elbow_deg, speed_mm_s, zone_mm)
        return self._accepted("ar5_move_elbow", f"ar5_{side}")

    def ar5_stop(self, side: str) -> ActionResponse:
        """转发 AR5 停止请求。"""

        self._gateway.ar5_stop(side)
        return self._accepted("ar5_stop", f"ar5_{side}")

    @staticmethod
    def _accepted(action: str, device: str) -> ActionResponse:
        """构造控制请求接受响应。"""

        return ActionResponse(
            ROBOT_CONTROL_VERSION,
            "1",
            True,
            {"action": action, "device": device},
        )

    def close(self) -> None:
        """释放网关硬件资源。"""

        self._gateway.close()
