from __future__ import annotations

import os
import select
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import paramiko
from loguru import logger
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

DEFAULT_WUJI_SSH_USERNAME = "wuji-brain"
"Orin SSH 登录用户名。"

DEFAULT_WUJI_SSH_PASSWORD = "wuji-brain"
"Orin SSH 登录密码；仅供本机 GUI 共享转发使用。"

DEFAULT_WUJI_REQUEST_TIMEOUT_S = 3.0
"默认 unary RPC 超时时间，单位 s。"

DEFAULT_WUJI_TUNNEL_WAIT_S = 1.0
"启动 SSH 隧道后的等待时间，单位 s。"

DEFAULT_WUJI_TUNNEL_READY_TIMEOUT_S = 5.0
"等待共享 SSH 隧道全部本地端点就绪的时间，单位 s。"

# endregion


# region 通道工厂


def create_qmlinker_channel(target: str) -> Any:
    """创建 qmlinker channel，供项目业务层使用。

    Parameters
    ----------
    target:
        目标地址，格式为 ``host:port``。

    Returns
    -------
    Any
        qmlinker 返回的基础 channel 或 channel 字典。

    Notes
    -----
    业务模块应调用本函数，不直接导入无类型桩的第三方 ``qmlinker`` 包。
    """

    return create_channel(target)


# endregion


# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiSshForward:
    """一个共享 SSH 进程中的本地转发规则。

    该结构只描述 ``ssh -L`` 的四元组，不持有进程。调用方可在连接前一次性组合
    qmlinker、AR5、相机和夹爪规则，由会话统一启动和释放。
    """

    local_host: str
    "本地监听地址。"

    local_port: int
    "本地监听 TCP 端口。"

    remote_host: str
    "Orin 可访问的远端设备地址。"

    remote_port: int
    "远端设备 TCP 端口。"


@dataclass(slots=True)
class _SshTunnelProcess:
    """单个 SSH 本地端口转发进程。

    该结构只保存一个 ``ssh -N -L`` 子进程及其两端地址，不负责设备协议。会话关闭时
    按该结构中的进程句柄终止转发，不能跨进程或序列化复用。
    """

    remote_host: str
    "Orin 可访问的远端设备地址。"

    remote_port: int
    "远端设备 TCP 端口。"

    local_host: str
    "本地监听地址；双 AR5 使用不同 loopback 地址。"

    local_port: int
    "本地监听 TCP 端口。"

    process: subprocess.Popen[str]
    "持有 SSH 转发生命周期的子进程。"

    @property
    def local_target(self) -> str:
        """返回本地转发目标。"""

        return f"{self.local_host}:{self.local_port}"


class _ForwardingTcpServer(socketserver.ThreadingTCPServer):
    """一个绑定到 Paramiko Transport 的本地 TCP 转发服务。

    该类只保存单条本地监听规则与共享 SSH Transport，不创建机械臂或 qmlinker
    客户端。每个本地连接由 ``ThreadingTCPServer`` 分配独立 handler 线程，服务关闭
    时由所属 ``_PasswordSshTunnelGroup`` 统一停止。

    设计思想：
    - 左右 AR5 需要在不同 loopback 地址复用相同 SDK 端口，因此每条转发规则保留独立
      TCP server。
    - 所有 server 共享一个已完成密码认证的 SSH Transport，避免重复登录和密码窗口。

    继承关系：
    - 继承标准库 ``ThreadingTCPServer`` 以复用监听、并发和关闭生命周期。
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        local_address: tuple[str, int],
        remote_address: tuple[str, int],
        transport: paramiko.Transport,
    ) -> None:
        """创建一条本地监听规则。

        Parameters
        ----------
        local_address:
            本地监听地址与 TCP 端口。
        remote_address:
            Orin 可访问的目标地址与 TCP 端口。
        transport:
            已完成认证且保持活动的 Paramiko SSH Transport。
        """

        self.remote_address = remote_address
        self.transport = transport
        super().__init__(local_address, _ForwardingRequestHandler)


class _ForwardingRequestHandler(socketserver.BaseRequestHandler):
    """在本地 TCP socket 与 Paramiko direct-tcpip channel 之间双向转发。"""

    def handle(self) -> None:
        """持续转发单个本地连接，任一端关闭后释放 SSH channel。"""

        if not isinstance(self.server, _ForwardingTcpServer):
            raise RuntimeError("SSH 转发 handler 绑定了错误的 server 类型")
        local_socket = self.request
        peer_address = local_socket.getpeername()
        channel: paramiko.Channel | None = None
        try:
            channel = self.server.transport.open_channel(
                "direct-tcpip",
                self.server.remote_address,
                peer_address,
            )
            if channel is None:
                logger.error(
                    "SSH forward channel rejected: local={} remote={}",
                    self.server.server_address,
                    self.server.remote_address,
                )
                return
            logger.debug(
                "SSH forward stream opened: local={} remote={} peer={}",
                self.server.server_address,
                self.server.remote_address,
                peer_address,
            )
            while True:
                readable, _, _ = select.select((local_socket, channel), (), ())
                if local_socket in readable:
                    data = local_socket.recv(65536)
                    if not data:
                        return
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(65536)
                    if not data:
                        return
                    local_socket.sendall(data)
        except (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
        ) as exc:
            logger.debug(
                "SSH forward stream ended by local connection: "
                "local={} remote={} peer={} reason={}",
                self.server.server_address,
                self.server.remote_address,
                peer_address,
                exc,
            )
        except Exception:
            logger.exception(
                "SSH forward stream failed: local={} remote={} peer={}",
                self.server.server_address,
                self.server.remote_address,
                peer_address,
            )
        finally:
            if channel is not None:
                channel.close()
            logger.debug(
                "SSH forward stream closed: local={} remote={} peer={}",
                self.server.server_address,
                self.server.remote_address,
                peer_address,
            )


@dataclass(slots=True)
class _PasswordSshTunnelGroup:
    """一个密码认证 SSH 连接及其全部本地转发服务。

    该结构由 GUI 后台连接线程创建，运行期间由多个 TCP handler 线程读取共享 Transport；
    GUI 断开时先停止监听服务，再关闭 SSHClient 并等待服务线程退出。它不跨进程复用，
    也不在日志或异常中暴露认证密码。
    """

    client: paramiko.SSHClient
    "持有已认证 SSH Transport 的 Paramiko 客户端。"

    servers: tuple[_ForwardingTcpServer, ...]
    "每条本地转发规则对应的 TCP 服务。"

    threads: tuple[threading.Thread, ...]
    "运行各 TCP 服务循环的后台线程。"

    def close(self) -> None:
        """停止全部本地监听并释放 SSH 连接。"""

        logger.info("SSH password tunnel closing: forwards={}", len(self.servers))
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.client.close()
        for thread in self.threads:
            thread.join(timeout=1.0)
        logger.info("SSH password tunnel closed")


# endregion


# region 主入口


class WujiQmlinkerSession:
    """无际 qmlinker 会话。

    职责边界：
    - 负责为平板直连或本机 SSH 调试创建 qmlinker 基础 channel。
    - 负责按显式本地地址启动和关闭 SSH 端口转发。
    - 不负责 AGV、右手、机械臂、body、head 等设备域的业务方法封装。

    设计思想：
    - 普通服务默认使用 ``remote_port - 1``，双 AR5 允许显式指定 loopback 地址和端口。
    - 连接模式由调用方完成 ping 探测后明确传入，本类不隐式探测网络。
    - GUI 共享转发使用单个 Paramiko 密码认证连接，避免 OpenSSH 弹出交互密码框。
    - 未提供共享规则的按需转发仍保留系统 OpenSSH 路径，供现有独立调用方使用。

    生命周期：
    - 通常随 GUI 连接包创建。
    - `close()` 会停止 Paramiko TCP 服务并终止本会话创建的 OpenSSH 子进程。

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
        direct: bool = False,
        ssh_host: str | None = None,
        tunnel_forwards: tuple[WujiSshForward, ...] = (),
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
        direct:
            是否从平板直接访问现场固定网段。为 ``False`` 时通过 SSH 转发。
        ssh_host:
            本机调试时的 SSH 接入地址。允许使用 DHCP 地址覆盖 ``ssh_alias``。
        tunnel_forwards:
            本机模式启动时一次性建立的共享转发规则。为空时保持按需创建能力。
        """

        self._host = str(host)
        self._port = int(port)
        self._ssh_alias = str(ssh_alias)
        self._ssh_host = str(ssh_host).strip() if ssh_host is not None else None
        self._direct = bool(direct)
        self._tunnels: dict[tuple[str, int, str, int], _SshTunnelProcess] = {}
        self._shared_forward_targets: dict[tuple[str, int], str] = {}
        self._password_tunnel: _PasswordSshTunnelGroup | None = None
        if not self._direct and tunnel_forwards:
            self._open_shared_ssh_tunnel(tunnel_forwards)
        self._connect_target_value = self.resolve_target(self._host, self._port)
        self._channel = create_channel(self._connect_target_value)
        self._default_channel = self._channel["DEFAULT"] if isinstance(self._channel, dict) else self._channel
        self._move_base_target = self.resolve_target(DEFAULT_WUJI_AGV_HOST, self._port)

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

    @property
    def direct(self) -> bool:
        """返回当前会话是否采用现场网段直连。"""

        return self._direct

    # endregion

    # region 生命周期

    def check_ready(self) -> None:
        """等待 qmlinker 默认通道进入 ready。"""

        grpc.channel_ready_future(self._default_channel).result(timeout=self.REQUEST_TIMEOUT_S)

    def close(self) -> None:
        """关闭本会话创建的全部 SSH 隧道进程。"""

        if self._password_tunnel is not None:
            self._password_tunnel.close()
            self._password_tunnel = None
            self._shared_forward_targets.clear()
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

    def resolve_target(self, remote_host: str, remote_port: int) -> str:
        """按当前连接模式返回直连或本地转发目标。

        Parameters
        ----------
        remote_host:
            现场固定网段中的服务主机。
        remote_port:
            服务 TCP 端口。

        Returns
        -------
        target:
            可传给 qmlinker 或 ZMQ 客户端的 ``host:port``。
        """

        if self._direct:
            return f"{remote_host}:{int(remote_port)}"
        shared_target = self._shared_forward_targets.get(
            (str(remote_host), int(remote_port))
        )
        if shared_target is not None:
            return shared_target
        return self.open_ssh_tunnel(remote_host, remote_port)

    def open_ssh_tunnel(self, remote_host: str, remote_port: int) -> str:
        """为指定远端地址创建固定本地端口 SSH 转发。"""

        for tunnel in self._tunnels.values():
            if (
                tunnel.remote_host == str(remote_host)
                and tunnel.remote_port == int(remote_port)
                and tunnel.local_host == "127.0.0.1"
                and tunnel.process.poll() is None
            ):
                return tunnel.local_target
        return self.open_fixed_ssh_tunnel(
            local_host="127.0.0.1",
            local_port=self._allocate_local_port(int(remote_port)),
            remote_host=remote_host,
            remote_port=remote_port,
        )

    def open_fixed_ssh_tunnel(
        self,
        *,
        local_host: str,
        local_port: int,
        remote_host: str,
        remote_port: int,
    ) -> str:
        """创建显式本地地址和端口的 SSH 转发。

        Parameters
        ----------
        local_host:
            本地监听地址。双 AR5 使用不同 loopback 地址复用 SDK 固定端口。
        local_port:
            本地监听端口。
        remote_host:
            Orin 可访问的现场设备地址。
        remote_port:
            现场设备端口。

        Returns
        -------
        target:
            本地转发目标，格式为 ``host:port``。
        """

        if self._direct:
            return f"{remote_host}:{int(remote_port)}"
        key = (str(remote_host), int(remote_port), str(local_host), int(local_port))
        existing_tunnel = self._tunnels.get(key)
        if existing_tunnel is not None and existing_tunnel.process.poll() is None:
            return existing_tunnel.local_target

        ssh_target = self._ssh_host or self._ssh_alias
        command = [
            "ssh",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "TCPKeepAlive=yes",
            "-N",
            "-L",
            f"{local_host}:{int(local_port)}:{remote_host}:{int(remote_port)}",
            ssh_target,
        ]
        process = subprocess.Popen(command, text=True, stderr=subprocess.PIPE)
        time.sleep(DEFAULT_WUJI_TUNNEL_WAIT_S)
        if process.poll() is not None:
            error_text = ""
            if process.stderr is not None:
                error_text = process.stderr.read().strip()
            raise RuntimeError(
                "ssh tunnel exited early: "
                f"remote={remote_host}:{int(remote_port)} "
                f"local={local_host}:{int(local_port)} "
                f"ssh={ssh_target} error={error_text or 'unknown'}"
            )
        tunnel = _SshTunnelProcess(
            remote_host=str(remote_host),
            remote_port=int(remote_port),
            local_host=str(local_host),
            local_port=int(local_port),
            process=process,
        )
        self._tunnels[key] = tunnel
        return tunnel.local_target

    def _open_shared_ssh_tunnel(
        self,
        forwards: tuple[WujiSshForward, ...],
    ) -> None:
        """按已验证 AR5 路径和 Paramiko 服务路径建立共享转发。

        Parameters
        ----------
        forwards:
            需要同时建立的本地监听与远端目标规则。

        Notes
        -----
        ``127.0.0.2`` / ``127.0.0.3`` 的 AR5 规则严格复用
        ``test/wuji/xcoresdk_arm_cli_test.py`` 的系统 OpenSSH 命令。其余服务使用一个
        Paramiko 连接。平板直连模式不会进入该方法。
        """

        if not forwards:
            return
        ssh_target = self._ssh_host or self._ssh_alias
        service_forwards = tuple(
            forward for forward in forwards if forward.local_host == "127.0.0.1"
        )
        arm_forwards = tuple(
            forward for forward in forwards if forward.local_host != "127.0.0.1"
        )
        logger.info(
            "SSH password tunnel connecting: host={} username={} forwards={}",
            ssh_target,
            DEFAULT_WUJI_SSH_USERNAME,
            len(service_forwards),
        )
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ssh_target,
            username=DEFAULT_WUJI_SSH_USERNAME,
            password=DEFAULT_WUJI_SSH_PASSWORD,
            timeout=DEFAULT_WUJI_TUNNEL_READY_TIMEOUT_S,
            banner_timeout=DEFAULT_WUJI_TUNNEL_READY_TIMEOUT_S,
            auth_timeout=DEFAULT_WUJI_TUNNEL_READY_TIMEOUT_S,
            allow_agent=False,
            look_for_keys=False,
        )
        transport = client.get_transport()
        if transport is None or not transport.is_active():
            client.close()
            raise RuntimeError("SSH 密码认证完成后未获得活动 Transport")
        transport.set_keepalive(5)
        logger.info("SSH password authentication succeeded: host={}", ssh_target)

        servers: list[_ForwardingTcpServer] = []
        threads: list[threading.Thread] = []
        try:
            for forward in service_forwards:
                server = _ForwardingTcpServer(
                    (forward.local_host, forward.local_port),
                    (forward.remote_host, forward.remote_port),
                    transport,
                )
                thread = threading.Thread(
                    target=server.serve_forever,
                    kwargs={"poll_interval": 0.1},
                    name=(
                        f"ssh-forward-{forward.local_host}-{forward.local_port}"
                    ),
                    daemon=True,
                )
                thread.start()
                servers.append(server)
                threads.append(thread)
                self._shared_forward_targets[
                    (forward.remote_host, forward.remote_port)
                ] = f"{forward.local_host}:{forward.local_port}"
                logger.debug(
                    "SSH forward listening: local={}:{} remote={}:{}",
                    forward.local_host,
                    forward.local_port,
                    forward.remote_host,
                    forward.remote_port,
                )
        except Exception:
            for server in servers:
                server.shutdown()
                server.server_close()
            client.close()
            for thread in threads:
                thread.join(timeout=1.0)
            self._shared_forward_targets.clear()
            raise
        self._password_tunnel = _PasswordSshTunnelGroup(
            client=client,
            servers=tuple(servers),
            threads=tuple(threads),
        )
        logger.info(
            "SSH password tunnel ready: host={} forwards={}",
            ssh_target,
            len(service_forwards),
        )
        try:
            self._open_verified_arm_ssh_tunnel(ssh_target, arm_forwards)
        except Exception:
            self._password_tunnel.close()
            self._password_tunnel = None
            self._shared_forward_targets.clear()
            raise

    def _open_verified_arm_ssh_tunnel(
        self,
        ssh_host: str,
        forwards: tuple[WujiSshForward, ...],
    ) -> None:
        """按已验证 CLI 命令启动双 AR5 OpenSSH 固定端口转发。

        Parameters
        ----------
        ssh_host:
            用户输入的 Orin DHCP 地址或 SSH Host 别名。
        forwards:
            左右 AR5 的独立 loopback 地址、控制器地址与 SDK 固定端口。

        Raises
        ------
        RuntimeError
            OpenSSH 在启动等待期内退出，或 askpass 脚本不存在。

        Notes
        -----
        密码通过 ``SSH_ASKPASS`` 提供，不出现在进程命令行。转发参数顺序、心跳参数和
        1 秒启动等待与 ``test/wuji/xcoresdk_arm_cli_test.py`` 保持一致。
        """

        if not forwards:
            return
        askpass_path = Path(__file__).with_name("orin_ssh_askpass.cmd")
        if not askpass_path.is_file():
            raise RuntimeError(f"Orin SSH askpass 脚本不存在：{askpass_path}")
        command = [
            "ssh",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "TCPKeepAlive=yes",
            "-o",
            "BatchMode=no",
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-N",
        ]
        for forward in forwards:
            command.extend(
                (
                    "-L",
                    f"{forward.local_host}:{forward.local_port}:"
                    f"{forward.remote_host}:{forward.remote_port}",
                )
            )
        command.append(f"{DEFAULT_WUJI_SSH_USERNAME}@{ssh_host}")
        process_environment = os.environ.copy()
        process_environment.update(
            {
                "SSH_ASKPASS": str(askpass_path),
                "SSH_ASKPASS_REQUIRE": "force",
                "DISPLAY": "dingtai-ssh",
            }
        )
        logger.info(
            "AR5 OpenSSH tunnel starting: host={} forwards={}",
            ssh_host,
            len(forwards),
        )
        creation_flags = (
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=process_environment,
            creationflags=creation_flags,
        )
        time.sleep(DEFAULT_WUJI_TUNNEL_WAIT_S)
        if process.poll() is not None:
            error_text = ""
            if process.stderr is not None:
                error_text = process.stderr.read().strip()
            raise RuntimeError(
                f"AR5 OpenSSH 转发提前退出：{error_text or 'unknown'}"
            )
        for forward in forwards:
            key = (
                forward.remote_host,
                forward.remote_port,
                forward.local_host,
                forward.local_port,
            )
            self._tunnels[key] = _SshTunnelProcess(
                remote_host=forward.remote_host,
                remote_port=forward.remote_port,
                local_host=forward.local_host,
                local_port=forward.local_port,
                process=process,
            )
        logger.success(
            "AR5 OpenSSH tunnel ready: host={} pid={} forwards={}",
            ssh_host,
            process.pid,
            len(forwards),
        )

    def _allocate_local_port(self, remote_port: int) -> int:
        """为当前会话分配一个不冲突的本地端口。"""

        used_local_ports = {
            tunnel.local_port
            for tunnel in self._tunnels.values()
            if tunnel.local_host == "127.0.0.1"
        }
        local_port = int(remote_port) - 1
        while local_port in used_local_ports:
            local_port -= 1
        return local_port

    def debug_connection_summary(self) -> str:
        """返回当前会话的连接调试摘要。"""

        tunnel_targets = [
            *self._shared_forward_targets.values(),
            *(tunnel.local_target for tunnel in self._tunnels.values()),
        ]
        return (
            f"host={self._host} "
            f"port={self._port} "
            f"direct={self._direct} "
            f"ssh={self._ssh_host or self._ssh_alias} "
            f"channel_target={self._connect_target_value} "
            f"move_base_target={self._move_base_target} "
            f"tunnels={tunnel_targets}"
        )

    # endregion


# endregion
