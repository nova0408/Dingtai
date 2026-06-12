from __future__ import annotations

# region 依赖导入
import socket
import socketserver
import subprocess
import time
from dataclasses import dataclass

# endregion


# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiZmqCameraForwardSpec:
    """无际 ZMQ 相机端口转发规格。

    职责边界：
    - 只描述远端主机、SSH 别名与需要转发的端口集合。
    - 不负责 ZMQ 协议消费，也不负责图像解码。

    设计思想：
    - 把端口转发从相机客户端中拆出，避免客户端同时承担连接建立与业务消费两类职责。
    - 仅在确实需要本机 localhost 访问远端 ZMQ 服务时使用。

    生命周期：
    - 由调用方显式创建和关闭，不会被 `WujiZmqCameraClient` 自动持有。

    继承关系：
    - 不继承业务基类，作为转发配置数据使用。
    """

    ssh_alias: str
    "本机 SSH 配置中的跳板别名，例如 `orin`。"

    remote_host: str
    "远端 ZMQ 服务主机地址，例如 `192.168.100.60`。"

    remote_ports: tuple[int, ...]
    "需要转发的远端 TCP 端口集合。"


# endregion


# region 端口转发


class WujiZmqCameraPortForwarder:
    """无际 ZMQ 相机端口转发器。

    职责边界：
    - 只负责把本机 localhost 端口转发到远端 ZMQ 服务端口。
    - 不负责 ZMQ 消息解析、相机状态消费或 GUI 逻辑。

    设计思想：
    - 把连接建立、存活维护和业务消费拆开，避免客户端职责膨胀。
    - 采用显式 `start()` / `close()` 生命周期，便于脚本和 GUI 外围控制。

    生命周期：
    - 由调用方显式持有，`close()` 后释放 SSH 进程与本地监听端口。

    继承关系：
    - 不继承业务基类，作为独立网络适配器使用。
    """

    def __init__(self, spec: WujiZmqCameraForwardSpec) -> None:
        self._spec = spec
        self._process: subprocess.Popen[str] | None = None
        self.local_port_map: dict[int, int] = {}

    def start(self) -> dict[int, int]:
        """启动全部端口的本地转发并返回远端到本地端口映射。"""

        for remote_port in self._spec.remote_ports:
            self.local_port_map[int(remote_port)] = int(_find_free_local_port())

        ssh_args = ["ssh", "-N"]
        for remote_port, local_port in self.local_port_map.items():
            ssh_args.extend(["-L", f"{local_port}:{self._spec.remote_host}:{remote_port}"])
        ssh_args.append(self._spec.ssh_alias)
        self._process = subprocess.Popen(ssh_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("failed to start OpenSSH local forward for ZMQ camera")
            if all(_can_connect_local_port(port) for port in self.local_port_map.values()):
                return dict(self.local_port_map)
            time.sleep(0.1)

        raise RuntimeError("OpenSSH local forward ports were not ready in time")

    def close(self) -> None:
        """关闭本地 OpenSSH 端口转发进程。"""

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self._process.kill()
            self._process = None
        self.local_port_map.clear()


def _find_free_local_port() -> int:
    """查找一个本地可用 TCP 端口。"""

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as server:
        return int(server.server_address[1])


def _can_connect_local_port(port: int) -> bool:
    """检查本地端口是否已开始监听。"""

    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return True
    except OSError:
        return False


# endregion
