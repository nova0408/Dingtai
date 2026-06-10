from __future__ import annotations
# pyright: reportMissingImports=false

# region 依赖导入
import socket
import socketserver
import struct
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Lock
import re

import cv2
import lz4.block
import numpy as np
import zmq

from src.wuji.camera_protocol import (
    WujiCameraEnableState,
    WujiCameraFrame,
    WujiCameraIntrinsicsInfo,
    WujiCameraName,
    WujiCameraRuntimeInfo,
)
from src.wuji.protocol import load_wuji_robot_network_config

# endregion


# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiZmqCameraEndpoint:
    """无际 ZMQ 相机端口配置。

    职责边界：
    - 只描述某一路逻辑相机在 ZMQ 方案中的控制名、数据端口与默认分辨率。
    - 不负责网络连接、图像解码、SSH 隧道或 GUI 渲染。

    设计思想：
    - 保留 `camera_id`，用于向控制端口发送 `HEAD/CHEST/LEFT/RIGHT` 命令。
    - 保留 `stream_port` 与默认宽高，避免 GUI 与测试脚本散落硬编码端口。

    生命周期：
    - 模块加载时构造，可长期跨线程只读复用。

    继承关系：
    - 不继承业务基类，作为 ZMQ 相机配置数据使用。
    """

    camera_name: WujiCameraName
    "项目内逻辑相机名。"

    camera_id: str
    "ZMQ 控制通道使用的相机标识，例如 `HEAD`。"

    config_key: str
    "远端 `ob_camera.yaml` 中的相机配置键，例如 `camera_head`。"

    stream_port: int
    "对应相机数据流 ZMQ PUB 端口，单位 TCP 端口号。"

    width: int = 1280
    "默认图像宽度，单位 像素。"

    height: int = 720
    "默认图像高度，单位 像素。"


@dataclass(frozen=True, slots=True)
class WujiZmqCameraStatus:
    """无际 ZMQ 相机状态结果。

    职责边界：
    - 只保存控制端口 `get_status` 返回的在线、彩色和深度状态。
    - 不负责控制命令发送、帧拉流或 UI 显示逻辑。

    设计思想：
    - 把在线状态与彩色/深度开关一起返回，便于 GUI 判断“离线”和“被关闭”的区别。
    - 保持字段扁平，避免界面层继续解析无结构 JSON。

    生命周期：
    - 每次控制请求完成后构造，不持有外部资源。

    继承关系：
    - 不继承业务基类，作为相机控制结果数据使用。
    """

    camera_name: WujiCameraName
    "项目内逻辑相机名。"

    online: bool
    "相机服务端报告的在线状态。"

    color_enabled: bool
    "彩色流开关状态。"

    depth_enabled: bool
    "深度流开关状态。"


@dataclass(frozen=True, slots=True)
class WujiZmqCameraConfig:
    """无际 ZMQ 相机客户端配置。

    职责边界：
    - 只描述 ZMQ 相机控制口、SSH 跳板与主机地址。
    - 不负责套接字生命周期或后台线程调度。

    设计思想：
    - 默认沿用 `config/robot_network.toml` 中的基础控制工控机地址。
    - 当本机无法直连 `wuyou` 时，通过 `orin` SSH 跳板建立本地端口转发。

    生命周期：
    - 由调用方创建并传给 `WujiZmqCameraClient`；不持有运行态资源。

    继承关系：
    - 不继承业务基类，作为相机网络配置使用。
    """

    host: str
    "相机控制与数据服务目标主机。"

    control_port: int = 5570
    "ZMQ 控制端口，单位 TCP 端口号。"

    request_timeout_ms: int = 3000
    "控制命令超时，单位 ms。"

    stream_timeout_ms: int = 5000
    "数据流首帧等待超时，单位 ms。"

    ssh_alias: str = "orin"
    "SSH 跳板别名，对应用户本机 `~/.ssh/config` 中的 Orin 配置。"

    remote_service_alias: str = "wuyou-x1-via-orin"
    "用于读取 `wuyou` 相机配置文件的 SSH 别名。"

    remote_camera_config_path: str = "/home/wuyou/.zkgj_libs/casia_config/ob_camera/ob_camera.yaml"
    "远端相机配置文件路径。"


@dataclass(frozen=True, slots=True)
class _ZmqFrameHeader:
    """ZMQ 相机单帧头解析结果。"""

    magic: bytes
    version: int
    camera_index: int
    color_format: int
    depth_format: int
    color_width: int
    color_height: int
    color_data_size: int
    depth_width: int
    depth_height: int
    depth_data_size: int
    depth_original_size: int
    timestamp_us: int
    sequence: int


# endregion


# region 配置

_FRAME_HEADER_STRUCT = struct.Struct("<4sBBBBIIIIIIIQI")
_FRAME_HEADER_SIZE = _FRAME_HEADER_STRUCT.size

SUPPORTED_WUJI_ZMQ_CAMERAS: tuple[WujiZmqCameraEndpoint, ...] = (
    WujiZmqCameraEndpoint("head_camera", "HEAD", "camera_head", 5560),
    WujiZmqCameraEndpoint("chest_camera", "CHEST", "camera_chest", 5561),
    WujiZmqCameraEndpoint("left_hand_camera", "LEFT", "camera_left", 5562),
    WujiZmqCameraEndpoint("right_hand_camera", "RIGHT", "camera_right", 5563),
)
"当前 `sensors_depthcamera_ob_zmq_v2` 方案定义的逻辑相机与端口映射。"


# endregion


# region SSH 端口转发


class _SshTcpPortForwarder:
    """经 Orin 跳板访问 `wuyou` ZMQ 端口的本地 TCP 转发器。

    职责边界：
    - 只负责将本机 localhost 端口转发到 `wuyou` 上的控制口或数据口。
    - 不负责解析 ZMQ 协议，不解码图像，也不管理 GUI 线程。

    设计思想：
    - 复用用户已有 SSH key 与 `~/.ssh/config` 中的 `orin` 配置，避免把账号口令写死在项目里。
    - 一个远端端口对应一个本地 `ThreadingTCPServer`，保持实现简单且便于调试。

    生命周期：
    - 由 `WujiZmqCameraClient` 持有；client 关闭时统一关闭全部本地转发端口与 SSH 连接。

    继承关系：
    - 不继承业务基类，作为相机网络适配工具使用。

    线程/异步语义：
    - 每个本地 TCP 连接由 server 分配工作线程，只做 socket 与 SSH channel 之间的透明转发。
    """

    def __init__(self, ssh_alias: str, remote_host: str, remote_ports: tuple[int, ...]) -> None:
        self._ssh_alias = ssh_alias
        self._remote_host = remote_host
        self._remote_ports = remote_ports
        self._process: subprocess.Popen[str] | None = None
        self.local_port_map: dict[int, int] = {}

    def start(self) -> dict[int, int]:
        """启动全部端口的本地转发并返回远端到本地端口映射。"""
        for remote_port in self._remote_ports:
            self.local_port_map[int(remote_port)] = int(_find_free_local_port())

        ssh_args = ["ssh", "-N"]
        for remote_port, local_port in self.local_port_map.items():
            ssh_args.extend(["-L", f"{local_port}:{self._remote_host}:{remote_port}"])
        ssh_args.append(self._ssh_alias)
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
    return False


# endregion


# region 主入口


class WujiZmqCameraClient:
    """无际 ZMQ 相机客户端。

    职责边界：
    - 负责访问 `sensors_depthcamera_ob_zmq_v2` 的控制口与数据口。
    - 不负责机械臂、手部、AGV 等 qmlinker 业务。
    - 不创建 GUI 控件，只返回结构化状态、内参和帧数据。

    设计思想：
    - 当前 `wuyou` 现场运行的是 ZMQ 相机服务，不是 qmlinker CameraService。
    - 控制命令采用短连接 REQ/REP；图像流采用单路 SUB，便于后台线程按需独立拉流。
    - 当本机无法直连 `wuyou` 端口时，自动通过 `orin` 建立本地端口转发。

    生命周期：
    - 可被 GUI backend 长期持有。
    - `close()` 会终止 ZMQ Context，并关闭 SSH 转发。

    继承关系：
    - 不继承业务基类，作为相机协议适配器使用。

    线程/异步语义：
    - 控制请求受 `_control_lock` 保护，避免多个 REQ socket 并发切换时造成诊断混乱。
    - 数据流读取通常由调用方单独线程消费。
    """

    def __init__(self, config: WujiZmqCameraConfig | None = None) -> None:
        network_config = load_wuji_robot_network_config()
        self._config = (
            WujiZmqCameraConfig(host=network_config.base_control_ip) if config is None else config
        )
        self._context = zmq.Context()
        self._control_lock = Lock()
        self._forwarder: _SshTcpPortForwarder | None = None
        self._local_port_map: dict[int, int] = {}
        self._connect_host = self._config.host
        self._prepare_endpoints()

    def close(self) -> None:
        """关闭 ZMQ 上下文与 SSH 端口转发。"""

        if self._forwarder is not None:
            self._forwarder.close()
            self._forwarder = None
        self._context.term()

    def get_camera_status(self, camera_name: WujiCameraName) -> WujiZmqCameraStatus:
        """读取指定相机的在线与流开关状态。"""

        payload = self._send_control_command("get_status", camera_name)
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid get_status payload: {payload!r}")
        return WujiZmqCameraStatus(
            camera_name=camera_name,
            online=bool(data.get("online", False)),
            color_enabled=bool(data.get("color_enabled", False)),
            depth_enabled=bool(data.get("depth_enabled", False)),
        )

    def get_camera_enable_state(self, camera_name: WujiCameraName) -> WujiCameraEnableState:
        """读取当前彩色流开关，并转换为 GUI 使能状态结构。"""

        status = self.get_camera_status(camera_name)
        message = (
            f"online={status.online}, color_enabled={status.color_enabled}, depth_enabled={status.depth_enabled}"
        )
        return WujiCameraEnableState(
            camera_name=camera_name,
            enabled=status.color_enabled,
            api_available=True,
            message=message,
        )

    def set_camera_color_enabled(self, camera_name: WujiCameraName, enabled: bool) -> WujiCameraEnableState:
        """设置彩色流开关并返回最新状态。"""

        self._send_control_command("set_color_enabled", camera_name, {"enable": bool(enabled)})
        return self.get_camera_enable_state(camera_name)

    def set_camera_depth_enabled(self, camera_name: WujiCameraName, enabled: bool) -> WujiZmqCameraStatus:
        """设置深度流开关并返回最新状态。"""

        self._send_control_command("set_depth_enabled", camera_name, {"enable": bool(enabled)})
        return self.get_camera_status(camera_name)

    def get_camera_intrinsics(self, camera_name: WujiCameraName) -> WujiCameraIntrinsicsInfo:
        """读取指定相机内参与默认分辨率。"""

        payload = self._send_control_command("get_intrinsics", camera_name)
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid get_intrinsics payload: {payload!r}")
        endpoint = self._endpoint_for(camera_name)
        return WujiCameraIntrinsicsInfo(
            camera_name=camera_name,
            fx=float(data.get("fx", 0.0)),
            fy=float(data.get("fy", 0.0)),
            cx=float(data.get("cx", 0.0)),
            cy=float(data.get("cy", 0.0)),
            distortion=tuple(float(value) for value in data.get("dist", [])),
            width=endpoint.width,
            height=endpoint.height,
        )

    def list_camera_runtime_infos(self, online_only: bool = True) -> tuple[WujiCameraRuntimeInfo, ...]:
        """列出远端相机服务当前可见的运行时清单。

        Parameters
        ----------
        online_only:
            是否只返回远端服务报告为在线的相机。`True` 时会过滤离线槽位。

        Returns
        -------
        runtime_infos:
            相机运行时清单元组。每一项同时包含逻辑相机名、远端槽位名、
            可复制序列号以及在线/彩色/深度状态。

        Notes
        -----
        序列号来自 `wuyou` 上的 `ob_camera.yaml`，在线状态来自 ZMQ 控制端口
        `get_status` 请求。这样 GUI 显示的是“远端当前配置 + 当前在线状态”的组合，
        而不是本地硬编码的物理安装位推断。
        """

        serial_map = self._read_remote_camera_serial_map()
        runtime_infos: list[WujiCameraRuntimeInfo] = []
        for endpoint in SUPPORTED_WUJI_ZMQ_CAMERAS:
            status = self.get_camera_status(endpoint.camera_name)
            if online_only and not status.online:
                continue
            serial_number = serial_map.get(endpoint.config_key, "")
            runtime_infos.append(
                WujiCameraRuntimeInfo(
                    camera_name=endpoint.camera_name,
                    camera_id=endpoint.camera_id,
                    serial_number=serial_number,
                    display_name=self._build_runtime_display_name(endpoint.camera_id, serial_number),
                    online=status.online,
                    color_enabled=status.color_enabled,
                    depth_enabled=status.depth_enabled,
                )
            )
        return tuple(runtime_infos)

    def stream_camera_rgb_frames(self, camera_name: WujiCameraName) -> Iterator[WujiCameraFrame]:
        """流式读取指定相机的 RGB 图像。"""

        yield from self._stream_frames(camera_name=camera_name, expect_depth=False)

    def stream_camera_rgbd_frames(self, camera_name: WujiCameraName) -> Iterator[WujiCameraFrame]:
        """流式读取指定相机的 RGBD 图像。"""

        self.set_camera_depth_enabled(camera_name, True)
        yield from self._stream_frames(camera_name=camera_name, expect_depth=True)

    def stop_camera_depth_stream(self, camera_name: WujiCameraName) -> None:
        """关闭指定相机的深度流。"""

        self.set_camera_depth_enabled(camera_name, False)

    def _stream_frames(self, camera_name: WujiCameraName, expect_depth: bool) -> Iterator[WujiCameraFrame]:
        endpoint = self._endpoint_for(camera_name)
        socket_obj = self._context.socket(zmq.SUB)
        socket_obj.setsockopt(zmq.CONFLATE, 1)
        socket_obj.setsockopt(zmq.RCVHWM, 1)
        socket_obj.setsockopt_string(zmq.SUBSCRIBE, "")
        socket_obj.setsockopt(zmq.RCVTIMEO, int(self._config.stream_timeout_ms))
        socket_obj.connect(self._tcp_address(self._stream_port(endpoint.stream_port)))
        try:
            while True:
                raw_message = socket_obj.recv()
                yield self._decode_frame(
                    camera_name=camera_name,
                    raw_message=raw_message,
                    expect_depth=expect_depth,
                )
        finally:
            socket_obj.close(linger=0)

    def _decode_frame(self, camera_name: WujiCameraName, raw_message: bytes, expect_depth: bool) -> WujiCameraFrame:
        """解析单条 ZMQ 帧消息。"""

        if len(raw_message) < _FRAME_HEADER_SIZE:
            raise RuntimeError(f"ZMQ camera frame too short: {len(raw_message)}")
        header = _ZmqFrameHeader(*_FRAME_HEADER_STRUCT.unpack(raw_message[:_FRAME_HEADER_SIZE]))
        if header.magic != b"ZCAM":
            raise RuntimeError(f"invalid ZMQ camera frame magic: {header.magic!r}")

        color_start = _FRAME_HEADER_SIZE
        color_end = color_start + int(header.color_data_size)
        color_jpeg = raw_message[color_start:color_end]
        color_bgr = cv2.imdecode(np.frombuffer(color_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError("camera jpeg decode failed")

        depth_array: np.ndarray | None = None
        if int(header.depth_format) != 0 and int(header.depth_data_size) > 0:
            depth_start = color_end
            depth_end = depth_start + int(header.depth_data_size)
            depth_bytes = raw_message[depth_start:depth_end]
            depth_raw = lz4.block.decompress(depth_bytes, uncompressed_size=int(header.depth_original_size))
            depth_array = np.frombuffer(depth_raw, dtype=np.uint16).reshape(
                (int(header.depth_height), int(header.depth_width))
            ).copy()
        elif expect_depth:
            raise RuntimeError("camera rgbd frame missing depth payload")

        return WujiCameraFrame(
            camera_name=camera_name,
            color_bgr=np.asarray(color_bgr, dtype=np.uint8).copy(),
            timestamp=int(header.timestamp_us),
            sequence_id=int(header.sequence),
            depth=depth_array,
        )

    def _send_control_command(
        self,
        command_name: str,
        camera_name: WujiCameraName,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """向控制端口发送单次命令。"""

        endpoint = self._endpoint_for(camera_name)
        payload = {
            "cmd": command_name,
            "camera": endpoint.camera_id,
            "params": {} if params is None else params,
        }
        with self._control_lock:
            response: object | None = None
            last_error: Exception | None = None
            for attempt in range(1, 4):
                socket_obj = self._context.socket(zmq.REQ)
                socket_obj.setsockopt(zmq.RCVTIMEO, int(self._config.request_timeout_ms))
                socket_obj.setsockopt(zmq.SNDTIMEO, int(self._config.request_timeout_ms))
                socket_obj.connect(self._tcp_address(self._control_port()))
                try:
                    socket_obj.send_json(payload)
                    response = socket_obj.recv_json()
                    last_error = None
                    break
                except zmq.error.Again as exc:
                    last_error = exc
                    time.sleep(0.2 * attempt)
                finally:
                    socket_obj.close(linger=0)
            if last_error is not None:
                raise last_error
        if not isinstance(response, dict):
            raise RuntimeError(f"invalid camera control response: {response!r}")
        if not bool(response.get("success", False)):
            raise RuntimeError(str(response.get("error", "unknown camera control error")))
        return {str(key): value for key, value in response.items()}

    def _read_remote_camera_serial_map(self) -> dict[str, str]:
        """读取远端相机配置文件中的槽位到序列号映射。"""

        ssh_command = [
            "ssh",
            self._config.remote_service_alias,
            f"sed -n '1,220p' {self._config.remote_camera_config_path}",
        ]
        completed = subprocess.run(
            ssh_command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        serial_map: dict[str, str] = {}
        current_config_key = ""
        for raw_line in completed.stdout.splitlines():
            line = raw_line.rstrip()
            section_match = re.match(r"^\s*(camera_[a-z]+):\s*$", line)
            if section_match is not None:
                current_config_key = str(section_match.group(1))
                continue
            serial_match = re.match(r'^\s*sn:\s*"([^"]+)"\s*$', line)
            if serial_match is not None and current_config_key:
                serial_map[current_config_key] = str(serial_match.group(1))
        return serial_map

    @staticmethod
    def _build_runtime_display_name(camera_id: str, serial_number: str) -> str:
        """构造 GUI 展示用的相机名称。"""

        if serial_number:
            return f"{camera_id} | SN {serial_number}"
        return camera_id

    def _prepare_endpoints(self) -> None:
        """准备直连或 SSH 转发后的目标地址。"""

        required_ports = (self._config.control_port,) + tuple(spec.stream_port for spec in SUPPORTED_WUJI_ZMQ_CAMERAS)
        if all(self._can_connect_tcp(self._config.host, port) for port in required_ports):
            self._connect_host = self._config.host
            self._local_port_map = {}
            return

        self._forwarder = _SshTcpPortForwarder(
            ssh_alias=self._config.ssh_alias,
            remote_host=self._config.host,
            remote_ports=required_ports,
        )
        self._local_port_map = self._forwarder.start()
        self._connect_host = "127.0.0.1"

    def _control_port(self) -> int:
        return int(self._local_port_map.get(self._config.control_port, self._config.control_port))

    def _stream_port(self, remote_port: int) -> int:
        return int(self._local_port_map.get(remote_port, remote_port))

    def _tcp_address(self, port: int) -> str:
        return f"tcp://{self._connect_host}:{int(port)}"

    def _endpoint_for(self, camera_name: WujiCameraName) -> WujiZmqCameraEndpoint:
        for spec in SUPPORTED_WUJI_ZMQ_CAMERAS:
            if spec.camera_name == camera_name:
                return spec
        raise ValueError(f"unsupported ZMQ camera: {camera_name}")

    def _can_connect_tcp(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return True
        except OSError:
            return False


# endregion
