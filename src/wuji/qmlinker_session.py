from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any

import grpc
from qmlinker import create_channel

# region 默认配置

DEFAULT_WUJI_QMLINKER_HOST = "192.168.100.60"
"默认 qmlinker 远端主机地址。"

DEFAULT_WUJI_AGV_HOST = "192.168.100.70"
"默认 AGV qmlinker 远端主机地址。"

DEFAULT_WUJI_QMLINKER_PORT = 50062
"默认 qmlinker 远端端口号。"

DEFAULT_WUJI_SSH_ALIAS = "orin"
"默认 SSH Host 别名。"

DEFAULT_WUJI_REQUEST_TIMEOUT_S = 3.0
"默认 unary RPC 超时时间，单位 s。"

DEFAULT_WUJI_TUNNEL_WAIT_S = 1.0
"启动 SSH 隧道后的等待时间，单位 s。"

# endregion


# region 数据结构


@dataclass(slots=True)
class _SshTunnelProcess:
    """单个 SSH 本地端口转发进程。"""

    remote_host: str
    remote_port: int
    local_port: int
    process: subprocess.Popen[str]

    @property
    def local_target(self) -> str:
        """返回本地转发目标。"""

        return f"127.0.0.1:{self.local_port}"


# endregion


# region 主入口


class WujiQmlinkerSession:
    """无际 qmlinker 会话。

    职责边界：
    - 只负责为本机调试创建 qmlinker 基础 channel。
    - 只负责按固定规则启动和关闭 SSH 端口转发。
    - 不负责 AGV、右手、机械臂、body、head 等设备域的业务方法封装。

    设计思想：
    - 按 `test/wuji/common.py` 的思路实现最简单的 SSH 转发：本地端口固定使用
      `remote_port - 1`，通过 `ssh -N -L ... orin` 建立隧道。
    - 不再保留自动直连探测、随机端口分配、paramiko、自定义 socket 转发等复杂逻辑。

    生命周期：
    - 通常随 GUI 后端创建。
    - `close()` 会终止本会话创建的全部 SSH 子进程。

    继承关系：
    - 不继承业务基类，作为各设备 client 共享的轻量连接会话使用。
    """

    REQUEST_TIMEOUT_S = DEFAULT_WUJI_REQUEST_TIMEOUT_S
    "会话统一使用的 unary RPC 超时时间，单位 s。"

    # region 初始化

    def __init__(
        self,
        host: str = DEFAULT_WUJI_QMLINKER_HOST,
        port: int = DEFAULT_WUJI_QMLINKER_PORT,
        *,
        ssh_alias: str = DEFAULT_WUJI_SSH_ALIAS,
    ) -> None:
        """初始化 qmlinker 会话。

        Parameters
        ----------
        host:
            qmlinker 远端主机地址。
        port:
            qmlinker 远端端口号，单位为 TCP 端口号。
        ssh_alias:
            本机 SSH 配置中的 Host 别名。
        """

        self._host = str(host)
        self._port = int(port)
        self._ssh_alias = str(ssh_alias)
        self._tunnels: dict[tuple[str, int], _SshTunnelProcess] = {}
        self._connect_target_value = self.open_ssh_tunnel(self._host, self._port)
        self._channel = create_channel(self._connect_target_value)
        self._default_channel = self._channel["DEFAULT"] if isinstance(self._channel, dict) else self._channel
        self._move_base_target = self.open_ssh_tunnel(DEFAULT_WUJI_AGV_HOST, self._port)

    # endregion

    # region 属性

    @property
    def host(self) -> str:
        """返回 qmlinker 远端主机地址。"""

        return self._host

    @property
    def port(self) -> int:
        """返回 qmlinker 远端端口号。"""

        return self._port

    @property
    def channel(self) -> Any:
        """返回 qmlinker 基础 channel。"""

        return self._channel

    @property
    def move_base_target(self) -> str:
        """返回 AGV 底盘连接目标。"""

        return self._move_base_target

    @property
    def request_timeout_s(self) -> float:
        """返回统一 unary RPC 超时时间。"""

        return self.REQUEST_TIMEOUT_S

    # endregion

    # region 生命周期

    def check_ready(self) -> None:
        """等待 qmlinker 默认通道进入 ready。"""

        grpc.channel_ready_future(self._default_channel).result(timeout=self.REQUEST_TIMEOUT_S)

    def close(self) -> None:
        """关闭本会话创建的全部 SSH 隧道进程。"""

        for tunnel in self._tunnels.values():
            process = tunnel.process
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        self._tunnels.clear()

    # endregion

    # region 转发能力

    def open_ssh_tunnel(self, remote_host: str, remote_port: int) -> str:
        """为指定远端地址创建固定本地端口 SSH 转发。"""

        key = (str(remote_host), int(remote_port))
        existing_tunnel = self._tunnels.get(key)
        if existing_tunnel is not None and existing_tunnel.process.poll() is None:
            return existing_tunnel.local_target

        local_port = self._allocate_local_port(int(remote_port))
        command = [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:{remote_host}:{int(remote_port)}",
            self._ssh_alias,
        ]
        process = subprocess.Popen(command, text=True)
        time.sleep(DEFAULT_WUJI_TUNNEL_WAIT_S)
        if process.poll() is not None:
            raise RuntimeError(
                f"ssh tunnel exited early: remote={remote_host}:{int(remote_port)} local=127.0.0.1:{local_port}"
            )
        tunnel = _SshTunnelProcess(
            remote_host=str(remote_host),
            remote_port=int(remote_port),
            local_port=local_port,
            process=process,
        )
        self._tunnels[key] = tunnel
        return tunnel.local_target

    def _allocate_local_port(self, remote_port: int) -> int:
        """为当前会话分配一个不冲突的本地端口。"""

        used_local_ports = {tunnel.local_port for tunnel in self._tunnels.values()}
        local_port = int(remote_port) - 1
        while local_port in used_local_ports:
            local_port -= 1
        return local_port

    def debug_connection_summary(self) -> str:
        """返回当前会话的连接调试摘要。"""

        tunnel_targets = [tunnel.local_target for tunnel in self._tunnels.values()]
        return (
            f"host={self._host} "
            f"port={self._port} "
            f"channel_target={self._connect_target_value} "
            f"move_base_target={self._move_base_target} "
            f"tunnels={tunnel_targets}"
        )

    # endregion


# endregion
