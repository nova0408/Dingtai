from __future__ import annotations

from qmlinker import QMHead
from qmlinker.grpc_py import common_pb2, head_pb2
from google.protobuf import empty_pb2

# region head 客户端


class WujiHeadClient(QMHead):
    """无际头部客户端。

    职责边界：
    - 直接继承 `QMHead`，负责头部 yaw 读取、使能读取和角度控制。
    - 不负责 body、机械臂、相机或 GUI 逻辑。

    设计思想：
    - 显式调用 qmlinker 已生成的 gRPC stub，避免上游封装吞掉业务失败消息。
    - 对所有写接口统一检查 ``status.success``，让 GUI 能展示服务端真实错误。

    生命周期：
    - 依赖外部传入的 qmlinker channel。
    - 不额外持有后台资源。

    继承关系：
    - 直接继承 `QMHead`。
    """

    def __init__(self, channel: object) -> None:
        """创建头部客户端。"""

        super().__init__(channel)

    def set_enable(self, enable: bool) -> bool:
        """设置头部模块使能并校验服务端业务结果。

        Parameters
        ----------
        enable:
            ``True`` 使能头部模块，``False`` 禁用头部模块。

        Returns
        -------
        success:
            服务端确认成功后固定返回 ``True``。

        Raises
        ------
        RuntimeError
            服务端返回失败状态，错误消息会原样保留供 GUI 显示。

        Notes
        -----
        当前 qmlinker 协议只提供一个头部模块级使能接口，没有 Yaw/Pitch 独立使能
        字段。GUI 可以分别显示两个电机，但物理使能仍由该共同接口控制。
        """

        response = self.stub.SetEnabled(
            common_pb2.ModuleEnableRequest(enable=enable)
        )
        self._require_success(
            "head.set_enable",
            bool(response.status.success),
            str(response.status.message),
        )
        return True

    def get_enable(self) -> bool:
        """读取头部模块共同使能状态。

        Returns
        -------
        enabled:
            ``True`` 表示头部模块已使能，``False`` 表示已禁用。

        Raises
        ------
        RuntimeError
            服务端未能完成状态查询。
        """

        response = self.stub.GetEnabled(empty_pb2.Empty())
        self._require_success(
            "head.get_enable",
            bool(response.status.success),
            str(response.status.message),
        )
        return response.current_state == common_pb2.MODULE_ENABLED

    def set_head_yaw(self, yaw_angle_deg: float) -> bool:
        """设置头部 Yaw 电机目标角度并校验业务结果。

        Parameters
        ----------
        yaw_angle_deg:
            Yaw 目标角度，单位 deg，正值向左。

        Returns
        -------
        success:
            服务端确认成功后固定返回 ``True``。

        Raises
        ------
        RuntimeError
            服务端拒绝目标角或电机当前不可控制。
        """

        response = self.stub.SetHeadYaw(
            head_pb2.SetHeadYawRequest(yaw_angle_deg=yaw_angle_deg)
        )
        self._require_success(
            "head.set_head_yaw",
            bool(response.status.success),
            str(response.status.message),
        )
        return True

    def get_head_yaw(self) -> float:
        """读取头部 Yaw 电机当前角度。

        Returns
        -------
        yaw_angle_deg:
            Yaw 当前角度，单位 deg。
        """

        response = self.stub.GetHeadYaw(empty_pb2.Empty())
        return float(response.current_yaw_deg)

    def set_head_pitch(self, pitch_angle_deg: float) -> bool:
        """设置头部 Pitch 电机目标角度并校验业务结果。

        Parameters
        ----------
        pitch_angle_deg:
            Pitch 目标角度，单位 deg。

        Returns
        -------
        success:
            服务端确认成功后固定返回 ``True``。

        Raises
        ------
        RuntimeError
            服务端拒绝目标角或电机当前不可控制。
        """

        response = self.stub.SetHeadPitch(
            head_pb2.SetHeadPitchRequest(pitch_angle_deg=pitch_angle_deg)
        )
        self._require_success(
            "head.set_head_pitch",
            bool(response.status.success),
            str(response.status.message),
        )
        return True

    def get_head_pitch(self) -> float:
        """读取头部 Pitch 电机当前角度。

        Returns
        -------
        pitch_angle_deg:
            Pitch 当前角度，单位 deg。
        """

        response = self.stub.GetHeadPitch(empty_pb2.Empty())
        return float(response.current_pitch_deg)

    @staticmethod
    def _require_success(action: str, success: bool, message: str) -> None:
        """校验头部服务业务状态并保留服务端错误消息。

        Parameters
        ----------
        action:
            当前接口名称，用于错误定位。
        success:
            服务端 ``status.success`` 字段。
        message:
            服务端 ``status.message`` 字段。

        Raises
        ------
        RuntimeError
            ``success`` 为 ``False``。
        """

        if success:
            return
        detail = message.strip() or "服务端未提供错误详情"
        raise RuntimeError(f"{action} 失败：{detail}")


# endregion
