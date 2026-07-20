"""双臂自动执行服务本机入口，部署参数集中写在本文件。"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

from loguru import logger

if sys.platform == "win32":
    import msvcrt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business.record_replay.context import ReplayContext
from src.business.record_replay.cycle_service import RecordReplayCycleService
from src.business.record_replay.offset_detector_gateway import CameraPipelineThreeBallDetector, load_three_ball_priors
from src.business.record_replay.offset_updater import GlobalOffsetUpdater
from src.business.record_replay.settings import (
    OffsetConfig,
    ReplayCycleConfig,
    ReplayDeviceConnection,
    ReplayNetworkSettings,
    ReplayServiceSettings,
)
from src.wuji import WujiAgvClient
from src.wuji.qmlinker_session import create_qmlinker_channel

# region 本机测试固定参数

LEFT_ARM_IP = "192.168.1.161"
"左臂 xCoreSDK 直连地址。"

RIGHT_ARM_IP = "192.168.1.160"
"右臂 xCoreSDK 直连地址。"

SSH_ALIAS = "orin"
"本机 SSH 配置中的 Orin 跳板别名。"

LOCAL_TUNNEL_HOST = "127.0.0.1"
"SSH 转发仅绑定本机环回地址。"

LOCAL_HAND_BODY_PORT = 50061
"本机 hand/body qmlinker 转发端口。"

LOCAL_GRIPPER_PORT = 50065
"本机左侧夹爪 qmlinker 转发端口。"

LOCAL_AGV_PORT = 50063
"本机 AGV qmlinker 转发端口，避开 hand/body 的同远端端口映射。"

TUNNEL_READY_TIMEOUT_S = 5.0
"等待三条本地 SSH 转发全部可连接的超时时间，单位 s。"

LEFT_RECORD_DIR = PROJECT_ROOT / "record_left"
RIGHT_RECORD_DIR = PROJECT_ROOT / "record_right"
OFFSET_CAMERA_NAME = "left_hand_camera"
OFFSET_PRIOR_CAPTURE_PATH = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture" / "summary.json"
HAND_EYE_RESULT_PATH = PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "20260708_152829" / "hand_eye_result.txt"

DEFAULT_AGV_POINT = "3"
"按下 a 后首先导航到的 AGV 地图站点名称。"

# endregion


# region 本机 SSH 隧道


class LocalServiceTunnelGroup:
    """持有本机测试所需的唯一 SSH 隧道及其全部设备客户端。

    该对象只存在于本机测试入口，不进入 `src.business.record_replay`。它使用一个
    `ssh -N` 进程同时建立 hand/body、gripper 与 AGV 三条本地转发：前两条指向
    Orin 内固定的 qmlinker 服务地址，AGV 指向固定的 AGV 地址。AGV 与 hand/body
    的远端端口均为 50062，因此必须使用不同本地端口。

    生命周期由 `main()` 的 `finally` 统一管理。业务服务只看到构造完成的
    `ReplayDeviceConnection` 与本对象暴露的 AGV 方法，不感知 SSH、进程或本机端口。
    本类不继承业务基类，避免将本机调试环境耦合到部署服务。
    """

    def __init__(self, network: ReplayNetworkSettings) -> None:
        """启动单一 SSH 进程，并创建 hand、gripper、AGV 的本地客户端。

        Parameters
        ----------
        network:
            Orin 现场网络配置。qmlinker 与 AGV 远端地址由该 dataclass 提供，
            本机监听端口由本测试文件顶部常量固定。

        Raises
        ------
        RuntimeError
            SSH 进程提前退出，或任一本地转发端口未在超时内就绪时抛出。
        """

        self._process: subprocess.Popen[bytes] | None = None
        self.device_connection = ReplayDeviceConnection(
            LEFT_ARM_IP,
            RIGHT_ARM_IP,
            LOCAL_TUNNEL_HOST,
            LOCAL_HAND_BODY_PORT,
            LOCAL_GRIPPER_PORT,
        )
        try:
            self._process = self._start_tunnel(network)
            _wait_for_local_tunnels((LOCAL_HAND_BODY_PORT, LOCAL_GRIPPER_PORT, LOCAL_AGV_PORT))
            self._agv_client = WujiAgvClient(
                create_qmlinker_channel(f"{LOCAL_TUNNEL_HOST}:{LOCAL_AGV_PORT}")
            )
        except BaseException:
            self.close()
            raise

    def navigate_to(self, target_name: str) -> object:
        """通过持有的 AGV 本地转发下发导航命令。

        Parameters
        ----------
        target_name:
            AGV 地图导航点名称。

        Returns
        -------
        object
            qmlinker AGV 客户端返回的原始导航结果。
        """

        return self._agv_client.navigate_to(target_name)

    def get_runtime_info(self) -> dict[str, object]:
        """通过持有的 AGV 本地转发读取当前运行信息。

        Returns
        -------
        dict[str, object]
            AGV 返回的运行状态字典。
        """

        return self._agv_client.get_runtime_info()

    def close(self) -> None:
        """停止唯一 SSH 进程，释放本机测试创建的全部转发。

        Notes
        -----
        qmlinker channel 随进程与解释器退出释放；此处首先保证 SSH 子进程不会残留。
        """

        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    @staticmethod
    def _start_tunnel(network: ReplayNetworkSettings) -> subprocess.Popen[bytes]:
        """创建承载三条本地转发的单一 SSH 进程。

        Parameters
        ----------
        network:
            Orin 现场网络配置。使用其中的 qmlinker、gripper、AGV 远端地址和端口。

        Returns
        -------
        subprocess.Popen[bytes]
            正在运行的 SSH 隧道进程。标准错误保留给提前退出诊断。

        Raises
        ------
        RuntimeError
            SSH 进程启动后立刻退出时抛出。
        """

        forwards = (
            (LOCAL_HAND_BODY_PORT, network.qmlinker_host, network.qmlinker_port),
            (LOCAL_GRIPPER_PORT, network.qmlinker_host, network.gripper_port),
            (LOCAL_AGV_PORT, network.agv_host, network.qmlinker_port),
        )
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
        ]
        for local_port, remote_host, remote_port in forwards:
            command.extend(("-L", f"{LOCAL_TUNNEL_HOST}:{local_port}:{remote_host}:{remote_port}"))
        command.append(SSH_ALIAS)
        logger.info(
            "启动统一 SSH 隧道：hand/body={}、gripper={}、agv={}，跳板={}",
            LOCAL_HAND_BODY_PORT,
            LOCAL_GRIPPER_PORT,
            LOCAL_AGV_PORT,
            SSH_ALIAS,
        )
        process = subprocess.Popen(command, stderr=subprocess.PIPE)
        time.sleep(0.2)
        if process.poll() is None:
            return process
        stderr = b"" if process.stderr is None else process.stderr.read()
        error_text = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"统一 SSH 隧道启动失败: {error_text or 'ssh 进程提前退出'}")


def _wait_for_local_tunnels(local_ports: tuple[int, ...]) -> None:
    """等待统一 SSH 隧道中的每个本地端口均可连接。

    Parameters
    ----------
    local_ports:
        需要验证的本机 TCP 端口序列。

    Raises
    ------
    RuntimeError
        任一端口在 `TUNNEL_READY_TIMEOUT_S` 内未就绪时抛出。
    """

    deadline = time.monotonic() + TUNNEL_READY_TIMEOUT_S
    pending_ports = set(local_ports)
    while pending_ports and time.monotonic() < deadline:
        for local_port in tuple(pending_ports):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex((LOCAL_TUNNEL_HOST, local_port)) == 0:
                    pending_ports.remove(local_port)
        if pending_ports:
            time.sleep(0.1)
    if pending_ports:
        ports_text = ", ".join(str(port) for port in sorted(pending_ports))
        raise RuntimeError(f"统一 SSH 隧道端口未就绪: {LOCAL_TUNNEL_HOST}:{ports_text}")


# endregion


# region 服务入口


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    """将 Linux service 停止信号转换为可执行 finally 的退出路径。"""

    del signum, frame
    raise KeyboardInterrupt


def _read_trigger_key() -> str:
    """读取启动按键，Windows 使用单键输入，其他平台使用标准输入。"""

    if sys.platform == "win32":
        return msvcrt.getwch().lower()
    return input().strip().lower()[:1]


def _run_on_keypress(service: RecordReplayCycleService) -> None:
    """按下 a 时执行一轮 AGV 导航与双臂动作，按 q 退出。"""

    logger.info("双臂服务已启动：按 a 导航到 {} 并执行双臂动作，按 q 退出", DEFAULT_AGV_POINT)
    while True:
        key = _read_trigger_key()
        if key == "a":
            logger.info("收到 a 键，开始导航到 AGV 站点 {}", DEFAULT_AGV_POINT)
            service.run_once()
            logger.success("本轮 AGV 导航与双臂动作执行完成，可再次按 a 启动下一轮")
        elif key == "q":
            logger.info("收到 q 键，退出本机双臂测试")
            return


def main() -> None:
    """启动本机双臂常态化测试服务。"""

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    service_settings = ReplayServiceSettings()
    tunnel_group = LocalServiceTunnelGroup(service_settings.network)
    try:
        context = ReplayContext(
            ReplayCycleConfig(
                LEFT_RECORD_DIR,
                RIGHT_RECORD_DIR,
                tunnel_group.device_connection,
                service_settings,
                start_station=DEFAULT_AGV_POINT,
            )
        )
        offset_config = OffsetConfig(
            OFFSET_PRIOR_CAPTURE_PATH,
            HAND_EYE_RESULT_PATH,
            service_addr=service_settings.network.zmq_service_addr,
            camera_name=OFFSET_CAMERA_NAME,
        )
        detector = CameraPipelineThreeBallDetector(
            offset_config.service_addr,
            OFFSET_CAMERA_NAME,
            load_three_ball_priors(OFFSET_PRIOR_CAPTURE_PATH, service_settings.offset),
            service_settings.offset,
        )
        service = RecordReplayCycleService(
            context,
            tunnel_group,
            offset_updater=GlobalOffsetUpdater(offset_config, detector),
        )
        _run_on_keypress(service)
    finally:
        tunnel_group.close()


if __name__ == "__main__":
    main()
