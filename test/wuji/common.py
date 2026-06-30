from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass

from loguru import logger
from qmlinker import create_channel
from network_discovery import (
    DEFAULT_ORIN_FALLBACKS,
    DEFAULT_ORIN_SSH_ALIAS,
    DEFAULT_WUYOU_FALLBACKS,
    DEFAULT_WUYOU_SSH_ALIAS,
    get_cached_wuyou_host,
    iter_candidate_hosts,
    remember_host,
)

GRIPPER_PORT = 50066
DEFAULT_PORT = 50062
DATA_PORT = 50061
WUYOU_QMLINKER_HOST = "192.168.100.60"
WUYOU_HOST = get_cached_wuyou_host()
AGV_HOST = "192.168.100.70"
SSH_ALIAS = "orin"
WUYOU_SSH_ALIAS = "wuyou"
TUNNEL_WAIT_S = 1.0
TUNNEL_HEARTBEAT_INTERVAL_S = 5.0
TUNNEL_HEARTBEAT_MAX_MISSES = 3
TUNNEL_WATCHDOG_INTERVAL_S = 1.0
TUNNEL_READY_TIMEOUT_S = 5.0


@dataclass(slots=True)
class SshTunnelGroup:
    """一组需要统一关闭的 SSH 转发进程。"""

    remote_host: str
    remote_port: int
    local_port: int
    ssh_alias: str
    process: subprocess.Popen[bytes]
    stop_event: threading.Event
    watcher_thread: threading.Thread

    @property
    def processes(self) -> tuple[subprocess.Popen[bytes], ...]:
        """兼容旧调用方，暴露当前活动 SSH 进程。"""

        return (self.process,)


def create_wuyou_channel(remote_port: int, remote_host: str = WUYOU_QMLINKER_HOST) -> tuple[SshTunnelGroup, object]:
    """通过 `orin` 连接 `wuyou` 的 qmlinker 服务，并创建本机调试 channel。

    Notes
    -----
    这里的链路含义是：
    - 本机先通过 SSH 连接 `orin`
    - 再由 `orin` 去访问 `wuyou` 侧的 `qmlinker` 服务地址
    - 远端 `remote_port` 映射到本地 `remote_port - 1`
    - 最后直接调用 `create_channel("127.0.0.1:{local_port}")`
    """

    local_port = int(remote_port) - 1
    process = start_ssh_tunnel(int(remote_port), remote_host=str(remote_host))
    _wait_for_local_tunnel_ready(local_port)
    stop_event = threading.Event()
    tunnel_group = SshTunnelGroup(
        remote_host=str(remote_host),
        remote_port=int(remote_port),
        local_port=local_port,
        ssh_alias=SSH_ALIAS,
        process=process,
        stop_event=stop_event,
        watcher_thread=threading.Thread(target=lambda: None),
    )
    watcher_thread = threading.Thread(
        target=_watch_ssh_tunnel,
        args=(tunnel_group,),
        name=f"ssh-tunnel-watch-{local_port}",
        daemon=True,
    )
    tunnel_group.watcher_thread = watcher_thread
    watcher_thread.start()
    return tunnel_group, create_channel(f"127.0.0.1:{local_port}")


def close_wuyou_channel(channel: object) -> None:
    """关闭 qmlinker channel，避免退出后后台 gRPC 继续刷屏。"""

    if isinstance(channel, dict):
        for item in channel.values():
            _close_single_channel(item)
        return
    _close_single_channel(channel)


def stop_ssh_process(process: subprocess.Popen[bytes] | SshTunnelGroup) -> None:
    """停止 SSH 转发进程。"""

    if isinstance(process, SshTunnelGroup):
        process.stop_event.set()
        _stop_single_ssh_process(process.process)
        if process.watcher_thread.is_alive():
            process.watcher_thread.join(timeout=3.0)
        return
    _stop_single_ssh_process(process)


def start_ssh_tunnel(
    remote_port: int,
    remote_host: str = WUYOU_QMLINKER_HOST,
    ssh_alias: str = SSH_ALIAS,
) -> subprocess.Popen[bytes]:
    """启动固定端口 SSH 本地转发。

    Notes
    -----
    默认只尝试缓存主机，避免每次启动都做探测。
    仅当 SSH 进程提前退出时，才按候选列表逐个重试并回写成功主机。
    """

    return _start_ssh_tunnel_with_local_port(
        remote_port=int(remote_port),
        remote_host=str(remote_host),
        local_port=int(remote_port) - 1,
        ssh_alias=str(ssh_alias),
    )


def _start_ssh_tunnel_with_local_port(
    remote_port: int,
    remote_host: str,
    local_port: int,
    ssh_alias: str,
    allow_discovery: bool = True,
) -> subprocess.Popen[bytes]:
    """按指定本地端口启动 SSH 转发。"""

    candidate_hosts = _build_tunnel_candidates(
        remote_host=str(remote_host),
        ssh_alias=str(ssh_alias),
        allow_discovery=bool(allow_discovery),
    )
    last_process: subprocess.Popen[bytes] | None = None
    for index, candidate_host in enumerate(candidate_hosts):
        if index > 0:
            logger.warning(
                "SSH 隧道首选主机失败，尝试候选 {}:{} -> {}",
                ssh_alias,
                int(remote_port),
                candidate_host,
            )
        process = _spawn_ssh_tunnel(
            remote_port=int(remote_port),
            remote_host=candidate_host,
            local_port=int(local_port),
            ssh_alias=str(ssh_alias),
        )
        time.sleep(TUNNEL_WAIT_S)
        if process.poll() is None:
            _remember_alias_host(ssh_alias=str(ssh_alias), remote_host=candidate_host)
            return process
        _read_early_exit_error(process)
        last_process = process

    if last_process is None:
        raise RuntimeError(f"ssh tunnel start failed: remote={remote_host}:{int(remote_port)} alias={ssh_alias}")
    raise RuntimeError(
        f"ssh tunnel exited early after retries: alias={ssh_alias} remote_port={int(remote_port)} host={remote_host}"
    )


def _spawn_ssh_tunnel(remote_port: int, remote_host: str, local_port: int, ssh_alias: str) -> subprocess.Popen[bytes]:
    """创建单次 SSH 隧道子进程。"""

    command = [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ServerAliveInterval={int(TUNNEL_HEARTBEAT_INTERVAL_S)}",
        "-o",
        f"ServerAliveCountMax={int(TUNNEL_HEARTBEAT_MAX_MISSES)}",
        "-o",
        "TCPKeepAlive=yes",
        "-N",
        "-L",
        f"127.0.0.1:{int(local_port)}:{remote_host}:{int(remote_port)}",
        ssh_alias,
    ]
    logger.info(
        "启动 SSH 隧道-本地 127.0.0.1:{} | 远端 {}:{} | 跳板 {}",
        int(local_port),
        remote_host,
        int(remote_port),
        ssh_alias,
    )
    return subprocess.Popen(command, stderr=subprocess.PIPE)


def _watch_ssh_tunnel(tunnel_group: SshTunnelGroup) -> None:
    """后台监视 SSH 隧道，异常退出时自动重连。"""

    while not tunnel_group.stop_event.wait(TUNNEL_WATCHDOG_INTERVAL_S):
        if tunnel_group.process.poll() is None:
            continue
        if tunnel_group.stop_event.is_set():
            return
        _read_early_exit_error(tunnel_group.process)
        logger.warning(
            "检测到 SSH 隧道退出，准备重连: local=127.0.0.1:{} remote={}:{} alias={}",
            tunnel_group.local_port,
            tunnel_group.remote_host,
            tunnel_group.remote_port,
            tunnel_group.ssh_alias,
        )
        try:
            restarted_process = _start_ssh_tunnel_with_local_port(
                remote_port=tunnel_group.remote_port,
                remote_host=tunnel_group.remote_host,
                local_port=tunnel_group.local_port,
                ssh_alias=tunnel_group.ssh_alias,
            )
            if tunnel_group.stop_event.is_set():
                _stop_single_ssh_process(restarted_process)
                return
            _wait_for_local_tunnel_ready(tunnel_group.local_port)
            if tunnel_group.stop_event.is_set():
                _stop_single_ssh_process(restarted_process)
                return
            tunnel_group.process = restarted_process
            logger.success(
                "SSH 隧道已重连: local=127.0.0.1:{} remote={}:{}",
                tunnel_group.local_port,
                tunnel_group.remote_host,
                tunnel_group.remote_port,
            )
        except Exception as exc:
            logger.warning("SSH 隧道重连失败，稍后重试: {}", exc)


def _wait_for_local_tunnel_ready(local_port: int, timeout_s: float = TUNNEL_READY_TIMEOUT_S) -> None:
    """等待本地 SSH 转发端口真正进入可连接状态。"""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", int(local_port))) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"ssh tunnel local port not ready: 127.0.0.1:{int(local_port)}")


def _build_tunnel_candidates(remote_host: str, ssh_alias: str, allow_discovery: bool) -> tuple[str, ...]:
    """按当前目标类型生成 SSH 转发重试候选。"""

    if not allow_discovery:
        return (str(remote_host),)
    if ssh_alias == DEFAULT_ORIN_SSH_ALIAS and remote_host == WUYOU_QMLINKER_HOST:
        return iter_candidate_hosts(ssh_alias, DEFAULT_ORIN_FALLBACKS, preferred_host=remote_host)
    if ssh_alias == DEFAULT_WUYOU_SSH_ALIAS and remote_host == WUYOU_QMLINKER_HOST:
        return iter_candidate_hosts(ssh_alias, DEFAULT_WUYOU_FALLBACKS, preferred_host=remote_host)
    return (str(remote_host),)


def _remember_alias_host(ssh_alias: str, remote_host: str) -> None:
    """仅为自动探测场景回写缓存。"""

    if ssh_alias == DEFAULT_ORIN_SSH_ALIAS:
        remember_host(ssh_alias, remote_host)
        return
    if ssh_alias == DEFAULT_WUYOU_SSH_ALIAS:
        remember_host(ssh_alias, remote_host)


def _read_early_exit_error(process: subprocess.Popen[bytes]) -> None:
    """读取 SSH 进程的提前退出错误日志。"""

    if process.stderr is None:
        return
    try:
        stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return
    if stderr_text:
        logger.warning("SSH 隧道提前退出: {}", stderr_text)


def _stop_single_ssh_process(process: subprocess.Popen[bytes]) -> None:
    """停止单个 SSH 转发进程。"""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except Exception:  # noqa: BLE001
        process.kill()
        process.wait(timeout=3.0)


def _close_single_channel(channel: object) -> None:
    """关闭单个 gRPC channel。"""

    close_method = getattr(channel, "close", None)
    if not callable(close_method):
        return
    try:
        close_method()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 qmlinker channel 失败: {}", exc)
