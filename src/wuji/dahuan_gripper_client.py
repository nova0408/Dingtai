from __future__ import annotations

from collections.abc import Mapping

from qmlinker import GripperInfo, QMGripper, create_channel

from src.wuji.client_base import WujiQmlinkerBaseClient, _SshTcpForwarder


# region 主入口


class DahuanGripperClient(QMGripper):
    """大寰夹爪客户端。

    职责边界：
    - 初始化时建立经 `orin` 的 SSH 端口转发。
    - 继承 `QMGripper` 复用已有使能、校准与位置控制接口。
    - 只重写 `get_status()`，不走原始 `QMGripper.get_status()`。

    设计思想：
    - 当前现场稳定链路是“本机 qmlinker -> Orin SSH 转发 -> 工控机夹爪端口”。
    - 本机环境中 `QMGripper.get_status()` 会因 `GripperInfo` 字段不匹配报错，因此这里
      直接调用 `_send_control(QMGripper.STATUS)` 读取原始字典，再手动封装为 `GripperInfo`。

    生命周期：
    - SSH 转发器挂到 `base_client` 的 `_forwarders` 中，由 `base_client.close()` 统一释放。

    继承关系：
    - 直接继承 `QMGripper`，不再额外包装控制接口。
    """

    def __init__(
        self,
        base_client: WujiQmlinkerBaseClient,
        ssh_alias: str | None = None,
        remote_host: str | None = None,
        remote_port: int | None = None,
    ) -> None:
        """初始化夹爪客户端并建立 SSH 转发。

        Parameters
        ----------
        base_client:
            共享机器人网络配置与生命周期的基础客户端。
        ssh_alias:
            可选 SSH 别名。为 `None` 时使用机器人网络配置中的 `orin_ssh_alias`。
        remote_host:
            可选远端夹爪地址。为 `None` 时使用机器人网络配置中的 `base_control_ip`。
        remote_port:
            可选远端夹爪端口。为 `None` 时使用机器人网络配置中的 `gripper_port`。
        """

        network = base_client.robot_network_config
        resolved_ssh_alias = network.orin_ssh_alias if ssh_alias is None else str(ssh_alias)
        resolved_remote_host = network.base_control_ip if remote_host is None else str(remote_host)
        resolved_remote_port = network.gripper_port if remote_port is None else int(remote_port)

        forwarder = _SshTcpForwarder(
            resolved_ssh_alias,
            resolved_remote_host,
            resolved_remote_port,
        )
        base_client._forwarders.append(forwarder)
        local_target = forwarder.start()
        super().__init__(create_channel(local_target))

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
            raise RuntimeError(f"gripper status payload is invalid: {payload!r}")

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
