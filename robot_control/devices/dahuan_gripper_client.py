from __future__ import annotations

from collections.abc import Mapping

from qmlinker import GripperInfo, QMGripper

# region 主入口


class DahuanGripperClient(QMGripper):
    """大寰夹爪客户端。

    职责边界：
    - 直接绑定已准备好的夹爪 channel。
    - 继承 `QMGripper` 复用已有使能、校准与位置控制接口。
    - 只重写 `get_status()`，不走原始 `QMGripper.get_status()`。

    设计思想：
    - 当前现场稳定链路是“本机 qmlinker -> Orin SSH 转发 -> 工控机夹爪端口”。
    - 本机环境中 `QMGripper.get_status()` 会因 `GripperInfo` 字段不匹配报错，因此这里
      直接调用 `_send_control(QMGripper.STATUS)` 读取原始字典，再手动封装为 `GripperInfo`。

    生命周期：
    - 本类不拥有 tunnel 生命周期；若调用方通过 SSH 转发暴露本地端口，应由调用方释放。

    继承关系：
    - 直接继承 `QMGripper`，不再额外包装控制接口。
    """

    def __init__(
        self,
        channel: object,
    ) -> None:
        """初始化夹爪客户端。

        Parameters
        ----------
        channel:
            已准备好的夹爪 qmlinker channel。可为直连 channel，也可为调用方基于本地
            SSH 转发目标创建的 channel。
        """

        super().__init__(channel)

    def get_status(self) -> GripperInfo:
        """读取夹爪状态并封装为 `GripperInfo`。

        Returns
        -------
        GripperInfo
            手动封装后的夹爪状态对象。

        Raises
        ------
        RuntimeError
            当底层返回值不是状态字典时抛出。
        """

        payload = self._send_control(QMGripper.STATUS)
        if not isinstance(payload, dict):
            raise RuntimeError(f"状态读取失败: {payload!r}")

        info = GripperInfo()
        self._assign_int_field(info, "position", payload, "position", 0)
        self._assign_int_field(info, "state", payload, "state", 0)
        self._assign_bool_field(info, "enable", payload, "enable", False)
        self._assign_bool_field(info, "online", payload, "online", True)
        self._assign_bool_field(info, "calibrated", payload, "calibrated", False)
        return info

    @staticmethod
    def _assign_int_field(
        info: GripperInfo,
        attr_name: str,
        payload: Mapping[str, object],
        key: str,
        default: int,
    ) -> None:
        """将整数字段写入 `GripperInfo`。"""

        value = payload.get(key, default)
        if value is None:
            setattr(info, attr_name, int(default))
            return
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            raise TypeError(f"gripper field {key} is not int-like: {value!r}")
        setattr(info, attr_name, int(value))

    @staticmethod
    def _assign_bool_field(
        info: GripperInfo,
        attr_name: str,
        payload: Mapping[str, object],
        key: str,
        default: bool,
    ) -> None:
        """将布尔字段写入 `GripperInfo`。"""

        value = payload.get(key)
        if value is None:
            setattr(info, attr_name, bool(default))
            return
        setattr(info, attr_name, bool(value))


# endregion
