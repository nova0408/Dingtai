from __future__ import annotations

import json
import os
import select
import socket
import socketserver
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Any

import grpc
import cv2
from google.protobuf import empty_pb2
import numpy as np
from qmlinker import QMArm, QMCamera, QMHand, QMMoveBase, create_channel
from qmlinker.grpc_py import arm_pb2, common_pb2, head_pb2_grpc
from qmlinker.grpc_py import lift_pb2_grpc
from qmlinker.grpc_py import hand_pb2
from qmlinker.grpc_py import camera_pb2, camera_pb2_grpc
from qmlinker.grpc_py import waist_pb2_grpc

from src.arm.wuji_arm_protocol import (
    ArmDeviceName,
    WUJI_BODY_AXIS_LIMITS,
    WUJI_HEAD_AXIS_LIMITS,
    WujiArmJointLimit,
)
from src.hand import HandDeviceName
from src.wuji.camera_protocol import WujiCameraName
from src.wuji.right_hand_specs import RIGHT_HAND_ACTUATOR_SPECS
from src.wuji.protocol import (
    WujiQmlinkerConfig,
    WujiQmlinkerEnableModuleName,
    WujiRobotNetworkConfig,
    WujiRobotRuntimeStructure,
    WujiRuntimeAxisSpec,
    WujiRuntimeModuleSpec,
    load_wuji_robot_network_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_QMLINKER_WHEEL_PATH = PROJECT_ROOT / "env" / "qmlinker-1.0.15-py3-none-any.whl"
REMOTE_QMLINKER_WHEEL_PATH = "/tmp/qmlinker-1.0.15-py3-none-any.whl"

class _SshTcpForwarder:
    """经 Orin 跳板访问工控机 qmlinker 端口的本地 TCP 转发器。

    职责边界：
    - 只负责在本机打开一个临时 localhost 端口，并通过 SSH direct-tcpip 转发到工控机。
    - 不负责解释 qmlinker 协议，不创建 qmlinker SDK 对象，不修改远端服务。

    设计思想：
    - 当前调试网络中 Windows 主机只能访问 Orin 的 `192.168.1.x` 地址，不能直接访问
      工控机 `192.168.100.x` 网段。
    - Orin 同时连接两个网段，因此可作为只读/控制链路的 SSH 跳板。

    生命周期：
    - 随 `WujiQmlinkerBaseClient` 创建和关闭。
    - 后台 server 线程仅处理本客户端的 gRPC TCP 流量，`close()` 时停止。

    继承关系：
    - 不继承业务基类，作为 qmlinker client 内部网络适配工具使用。

    线程/异步语义：
    - 每个本地 TCP 连接由 `ThreadingTCPServer` 分配工作线程。
    - 数据只在 socket 与 SSH channel 之间转发，不在 Python 层缓存业务数据。
    """

    def __init__(self, ssh_alias: str, remote_host: str, remote_port: int) -> None:
        """创建 SSH TCP 转发器。

        Parameters
        ----------
        ssh_alias:
            本机 `~/.ssh/config` 中的 Orin Host 别名。
        remote_host:
            Orin 可访问的工控机地址。
        remote_port:
            工控机 qmlinker gRPC 端口，单位 TCP 端口号。
        """

        self._ssh_alias = ssh_alias
        self._remote_host = remote_host
        self._remote_port = int(remote_port)
        self._ssh_client: Any | None = None
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: Thread | None = None
        self.local_port = 0

    def start(self) -> str:
        """启动本地转发并返回 qmlinker 可连接 target。

        Returns
        -------
        str
            本地 gRPC target，格式为 `127.0.0.1:port`。
        """

        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError("paramiko is required for qmlinker SSH tunnel") from exc

        ssh_config = paramiko.SSHConfig()
        config_path = Path.home() / ".ssh" / "config"
        with config_path.open("r", encoding="utf-8") as file:
            ssh_config.parse(file)
        info = ssh_config.lookup(self._ssh_alias)
        hostname = str(info.get("hostname", self._ssh_alias))
        username = str(info.get("user", self._ssh_alias))
        identity_files = info.get("identityfile", [])
        key_filename = str(identity_files[0]) if identity_files else None

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname,
            username=username,
            key_filename=key_filename,
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=key_filename is None,
            allow_agent=key_filename is None,
        )
        self._ssh_client = ssh_client
        transport = ssh_client.get_transport()
        if transport is None:
            raise RuntimeError("Orin SSH transport is not ready")

        remote_host = self._remote_host
        remote_port = self._remote_port

        class _ForwardHandler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                channel = transport.open_channel(
                    "direct-tcpip",
                    (remote_host, remote_port),
                    self.request.getpeername(),
                )
                try:
                    while True:
                        readable, _, _ = select.select([self.request, channel], [], [], 0.5)
                        if self.request in readable:
                            data = self.request.recv(65535)
                            if not data:
                                break
                            channel.sendall(data)
                        if channel in readable:
                            data = channel.recv(65535)
                            if not data:
                                break
                            self.request.sendall(data)
                finally:
                    channel.close()

        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _ForwardHandler)
        server.daemon_threads = True
        self.local_port = int(server.server_address[1])
        thread = Thread(target=server.serve_forever, name="qmlinker-ssh-tunnel", daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        return f"127.0.0.1:{self.local_port}"

    def is_active(self) -> bool:
        """判断本地转发器是否仍处于可用状态。"""

        return self._server is not None and self._ssh_client is not None

    def probe_local_target(self, timeout_s: float = 0.2) -> bool:
        """探测本地转发端口是否仍可建立连接。"""

        if self.local_port <= 0:
            return False
        try:
            with socket.create_connection(("127.0.0.1", self.local_port), timeout=timeout_s):
                return True
        except OSError:
            return False

    def debug_summary(self) -> str:
        """返回本地转发器的调试摘要。"""

        return (
            f"local_target={self.local_target()} "
            f"active={self.is_active()} "
            f"reachable={self.probe_local_target()}"
        )

    def local_target(self) -> str:
        """返回本地转发端口目标。"""

        return f"127.0.0.1:{self.local_port}" if self.local_port > 0 else "127.0.0.1:0"

    def close(self) -> None:
        """停止本地转发并关闭 SSH 连接。"""

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._ssh_client is not None:
            self._ssh_client.close()
            self._ssh_client = None


# endregion


# region 相机 gRPC 适配


class _WujiCameraServiceAdapter:
    """无际相机 gRPC 适配器。

    职责边界：
    - 负责直接调用相机 proto 定义的 unary 与 stream RPC。
    - 不负责 GUI 渲染、线程调度或相机名称解析。

    设计思想：
    - qmlinker wheel 中的 `QMCamera` 在 channel 为 dict 时固定使用 `DATA` 通道，这会让
      `GetCameraIntrinsics` 等 unary RPC 落到错误端口并收到 `UNIMPLEMENTED`。
    - 这里显式拆分 `DEFAULT` 与 `DATA` 通道：状态/内参/开关走 `DEFAULT`，图像流走 `DATA`。

    生命周期：
    - 随 `WujiQmlinkerBaseClient` 构造。
    - 不持有线程，仅持有 gRPC stub，可跨线程复用。

    继承关系：
    - 不继承业务基类，作为 qmlinker client 内部协议修正层使用。
    """

    def __init__(self, channel: Any, request_timeout_s: float) -> None:
        """创建相机 gRPC 适配器。

        Parameters
        ----------
        channel:
            qmlinker `create_channel` 返回的 channel 或 channel dict。
        request_timeout_s:
            unary RPC 超时时间，单位 s。
        """

        if isinstance(channel, dict):
            self._default_channel = channel["DEFAULT"]
            self._data_channel = channel["DATA"]
        else:
            self._default_channel = channel
            self._data_channel = channel
        self._request_timeout_s = float(request_timeout_s)
        self._default_stub = camera_pb2_grpc.CameraServiceStub(self._default_channel)
        self._data_stub = camera_pb2_grpc.CameraServiceStub(self._data_channel)

    def set_enable(self, camera_type: camera_pb2.CameraType, enable: bool) -> bool:
        """设置相机使能状态。"""

        request = camera_pb2.CameraEnableRequest(camera_type=camera_type, enable=bool(enable))
        response = self._default_stub.SetEnabled(request, timeout=self._request_timeout_s)
        return bool(response.status.success)

    def get_enable(self, camera_type: camera_pb2.CameraType) -> bool:
        """读取相机使能状态。"""

        request = camera_pb2.CameraRequest(camera_type=camera_type)
        response = self._default_stub.GetEnabled(request, timeout=self._request_timeout_s)
        return bool(response.status.success and response.current_state == common_pb2.MODULE_ENABLED)

    def get_camera_intrinsics(self, camera_type: camera_pb2.CameraType) -> dict[str, object]:
        """读取相机内参与基准分辨率。"""

        request = camera_pb2.CameraRequest(camera_type=camera_type)
        response = self._default_stub.GetCameraIntrinsics(request, timeout=self._request_timeout_s)
        if not response.status.success:
            raise RuntimeError(f"camera intrinsics rpc failed: {response.status.message}")
        return {
            "fx": float(response.intrinsics.fx),
            "fy": float(response.intrinsics.fy),
            "cx": float(response.intrinsics.cx),
            "cy": float(response.intrinsics.cy),
            "distortion": tuple(float(value) for value in response.intrinsics.distortion),
            "base_width": int(response.intrinsics.base_width),
            "base_height": int(response.intrinsics.base_height),
        }

    def control_depth_stream(self, camera_type: camera_pb2.CameraType, control_enable: bool) -> bool:
        """控制深度流开关。"""

        request = camera_pb2.DepthStreamControlRequest(camera_type=camera_type, enable=bool(control_enable))
        response = self._default_stub.ControlDepthStream(request, timeout=self._request_timeout_s)
        return bool(response.status.success and response.current_state == bool(control_enable))

    def stream_get_image_2d(self, camera_type: camera_pb2.CameraType) -> Iterator[tuple[np.ndarray, object]]:
        """流式读取 2D 图像。"""

        request = camera_pb2.CameraRequest(camera_type=camera_type)
        for response in self._data_stub.StreamGetImage2D(request, timeout=None):
            image = cv2.imdecode(np.frombuffer(response.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("camera jpeg decode failed")
            yield image, response.timestamp

    def stream_get_rgbd_image(
        self,
        camera_type: camera_pb2.CameraType,
    ) -> Iterator[tuple[np.ndarray, object, np.ndarray]]:
        """流式读取 RGBD 图像。"""

        intrinsics = self.get_camera_intrinsics(camera_type)
        width = self._require_int(intrinsics["base_width"], "base_width")
        height = self._require_int(intrinsics["base_height"], "base_height")
        request = camera_pb2.CameraRequest(camera_type=camera_type)
        for response in self._data_stub.StreamGetRGBDImage(request, timeout=None):
            color_bgr = cv2.imdecode(np.frombuffer(response.image_2d.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if color_bgr is None:
                raise RuntimeError("camera rgbd jpeg decode failed")
            # depth_data: (H * W * 2,) bytes，按 uint16 解码后重排为 (H, W) 深度矩阵。
            depth_array = np.frombuffer(response.depth_data, dtype=np.uint16)
            depth = depth_array.reshape((height, width))
            yield color_bgr, response.image_2d.timestamp, depth

    @staticmethod
    def _require_int(value: object, field_name: str) -> int:
        """将相机内参字段收窄为整数。"""

        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise TypeError(f"qmlinker camera intrinsics field {field_name} is not integer-like: {value!r}")


# endregion


# region 主入口


class WujiQmlinkerBaseClient:
    """无际 qmlinker 本机基础客户端。

    职责边界：
    - 负责在 DingTai 环境中通过 qmlinker SDK 访问基础控制工控机 qmlinker 服务。
    - 不执行远端 Python，不修改远端环境。
    - 不持有 GUI 控件，不发 Qt 信号。

    设计思想：
    - 直接使用接口文档提供的 `qmlinker-1.0.8-py3-none-any.whl`，避免重复实现 SDK 内部逻辑。
    - 客户端持有一个 channel，并集中管理臂、手、底盘、腰、升降和头部 SDK 对象。

    生命周期：
    - 随 GUI 后端创建和关闭。
    - `close` 只停止 SDK 内部线程，不负责关闭 GUI 会话。

    继承关系：
    - 不继承业务基类，作为硬件协议适配器使用。
    """

    def __init__(self, config: WujiQmlinkerConfig | None = None) -> None:
        """初始化无际 qmlinker 基础客户端。

        Parameters
        ----------
        config:
            qmlinker 连接配置，为 `None` 时使用默认配置。
        """

        self._config = WujiQmlinkerConfig() if config is None else config
        self._robot_network_config = load_wuji_robot_network_config()
        self._forwarders: list[_SshTcpForwarder] = []
        self._connect_target_value = self._connect_target()
        self._channel = create_channel(self._connect_target_value)
        self._move_base_target = self._connect_orin_local_target(self._config.port)
        self._arms: dict[ArmDeviceName, Any] = {}
        self._hands: dict[HandDeviceName, Any] = {}
        self._default_channel = self._channel["DEFAULT"] if isinstance(self._channel, dict) else self._channel
        self._camera = _WujiCameraServiceAdapter(self._channel, request_timeout_s=self._config.request_timeout_s)
        self._move_base = QMMoveBase(create_channel(self._move_base_target))
        self._waist_stub = waist_pb2_grpc.WaistServiceStub(self._default_channel)
        self._lift_stub = lift_pb2_grpc.LiftServiceStub(self._default_channel)
        self._head_stub = head_pb2_grpc.HeadServiceStub(self._default_channel)

    @property
    def channel(self) -> Any:
        """返回 gRPC channel 对象。"""
        return self._channel

    @property
    def move_base(self) -> QMMoveBase:
        """返回原始 AGV 底盘对象。"""

        return self._move_base

    def hand(self, device_name: HandDeviceName) -> Any:
        """返回指定手部对象。"""

        return self._hand(device_name)

    def get_base_status(self) -> Any:
        """读取 AGV 底盘原始状态。"""

        return self._move_base.get_base_status()

    def real_time_translate(self, speed_mps: float, direction_deg: int) -> Any:
        """实时平移 AGV。"""

        return self._move_base.real_time_translate(float(speed_mps), int(direction_deg))

    def real_time_rotate(self, speed_degps: float, direction: Any) -> Any:
        """实时旋转 AGV。"""

        return self._move_base.real_time_rotate(float(speed_degps), direction)

    def translate_with_distance(
        self,
        speed_mps: float = 0.1,
        direction_deg: int = 0,
        distance_m: float = 0.1,
    ) -> Any:
        """按距离平移 AGV。"""

        return self._move_base.translate_with_distance(float(speed_mps), int(direction_deg), float(distance_m))

    def translate_with_distance_sync(
        self,
        speed_mps: float = 0.1,
        direction_deg: int = 0,
        distance_m: float = 0.1,
        acc_threshold: Any | None = None,
    ) -> Any:
        """同步按距离平移 AGV。"""

        return self._move_base.translate_with_distance_sync(
            float(speed_mps),
            int(direction_deg),
            float(distance_m),
            acc_threshold,
        )

    def rotate_with_angle(self, speed_degps: float, direction: Any, angle_deg: float) -> Any:
        """按角度旋转 AGV。"""

        return self._move_base.rotate_with_angle(float(speed_degps), direction, float(angle_deg))

    def rotate_with_angle_sync(self, speed_degps: float, direction: Any, angle_deg: float) -> Any:
        """同步按角度旋转 AGV。"""

        return self._move_base.rotate_with_angle_sync(float(speed_degps), direction, float(angle_deg))

    def navigate_to(self, navi_target_name: str) -> Any:
        """发送 AGV 导航目标。"""

        return self._move_base.navigate_to(navi_target_name)

    def navigate_to_sync(self, target_name: str) -> Any:
        """同步发送 AGV 导航目标。"""

        return self._move_base.navigate_to_sync(target_name)

    def stop(self) -> Any:
        """停止 AGV。"""

        return self._move_base.stop()

    @property
    def config(self) -> WujiQmlinkerConfig:
        """返回配置对象。"""
        return self._config

    @property
    def robot_network_config(self) -> WujiRobotNetworkConfig:
        """返回整机网络配置。"""

        return self._robot_network_config

    def close(self) -> None:
        """停止 qmlinker 内部状态线程。"""

        arms = list(self._arms.values())
        try:
            self._move_base.stop()
        except Exception:  # noqa: BLE001
            pass
        for forwarder in self._forwarders:
            forwarder.close()
        self._forwarders.clear()
        self._arms.clear()
        self._hands.clear()
        for arm in arms:
            if hasattr(arm, "running"):
                arm.running = False
            thread = getattr(arm, "thread_joint_states", None)
            if thread is not None:
                thread.join(timeout=0.5)
            if not hasattr(arm, "thread_arm_pose"):
                arm.thread_arm_pose = thread

    def debug_connection_summary(self) -> str:
        """返回当前 qmlinker 连接链路的调试摘要。"""

        forwarder_summaries = [forwarder.debug_summary() for forwarder in self._forwarders]
        forwarder_text = "; ".join(forwarder_summaries) if forwarder_summaries else "none"
        return (
            f"config_target={self._config.target()} "
            f"channel_target={self._connect_target_value} "
            f"move_base_target={self._move_base_target} "
            f"forwarders=[{forwarder_text}]"
        )

    def check_ready(self) -> None:
        """检查 qmlinker gRPC 通道是否可创建并进入 ready。

        Raises
        ------
        TimeoutError
            在配置时间内 gRPC 通道无法 ready。
        """

        import grpc

        grpc.channel_ready_future(self._default_channel).result(timeout=self._config.request_timeout_s)

    def run_orin_python_json(self, remote_script: str, ssh_alias: str | None = None) -> dict[str, object]:
        """通过 Orin SSH 跳板执行 Python 脚本并解析 JSON 输出。

        Parameters
        ----------
        remote_script:
            要在 Orin 侧执行的 Python 代码。代码末尾应打印单个 JSON 对象。
        ssh_alias:
            可选 SSH 别名。为 `None` 时使用 `robot_network_config.orin_ssh_alias`。

        Returns
        -------
        dict[str, object]
            远端脚本打印的 JSON 字典。
        """

        alias = self._robot_network_config.orin_ssh_alias if ssh_alias is None else str(ssh_alias)
        self._ensure_orin_qmlinker_wheel(alias)
        bootstrap = f"""
import os
import subprocess
import sys

try:
    import qmlinker  # noqa: F401
except ImportError:
    wheel_path = {REMOTE_QMLINKER_WHEEL_PATH!r}
    if not os.path.exists(wheel_path):
        raise RuntimeError(f"missing qmlinker wheel: {{wheel_path}}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel_path], check=True)
"""
        python_command = f"{self._robot_network_config.orin_python} - <<'PY'\n{bootstrap}\n{remote_script}\nPY"
        completed = subprocess.run(
            ["ssh", alias, python_command],
            capture_output=True,
            text=True,
            timeout=self._config.request_timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "remote python command failed: "
                f"returncode={completed.returncode} stderr={completed.stderr.strip()} stdout={completed.stdout.strip()}"
            )
        output = completed.stdout.strip()
        if not output:
            raise RuntimeError("remote python command returned empty output")
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"remote python command returned invalid json: {output}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"remote python command returned invalid payload: {payload!r}")
        return payload

    def _ensure_orin_qmlinker_wheel(self, ssh_alias: str) -> None:
        """必要时把本地 qmlinker wheel 同步到 Orin。"""

        if not LOCAL_QMLINKER_WHEEL_PATH.exists():
            return
        target = f"{ssh_alias}:{REMOTE_QMLINKER_WHEEL_PATH}"
        subprocess.run(
            ["scp", str(LOCAL_QMLINKER_WHEEL_PATH), target],
            capture_output=True,
            text=True,
            timeout=self._config.request_timeout_s,
            check=True,
        )

    def get_module_enable(self, module_name: WujiQmlinkerEnableModuleName) -> bool:
        """读取整机模块使能状态。"""

        if module_name in {"left_arm", "right_arm"}:
            return bool(self._arm(self._arm_device_name(module_name)).get_enable())
        if module_name == "body":
            return self._get_stub_enable(self._lift_stub) and self._get_stub_enable(self._waist_stub)
        response = self._module_stub(module_name).GetEnabled(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success and response.current_state == common_pb2.MODULE_ENABLED)

    def set_module_enable(self, module_name: WujiQmlinkerEnableModuleName, enabled: bool) -> bool:
        """设置整机模块使能状态。"""

        if module_name in {"left_arm", "right_arm"}:
            return bool(self._arm(self._arm_device_name(module_name)).set_enable(enabled))
        if module_name == "body":
            return self._set_stub_enable(self._lift_stub, enabled) and self._set_stub_enable(
                self._waist_stub,
                enabled,
            )
        request = common_pb2.ModuleEnableRequest(enable=bool(enabled))
        response = self._module_stub(module_name).SetEnabled(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_agv_status_values(self) -> dict[str, float]:
        """读取 AGV 底盘基础状态值。"""

        last_error: Exception | None = None
        for _ in range(10):
            try:
                response = self._move_base.stub.GetBaseStatus(
                    empty_pb2.Empty(),
                    timeout=self._config.request_timeout_s,
                )
                status = {
                    "x": response.x,
                    "y": response.y,
                    "yaw": response.yaw,
                    "battery": response.battery,
                }
                if not status:
                    raise RuntimeError("qmlinker get base status failed")
                return {
                    "agv_x": float(status.get("x", 0.0)),
                    "agv_y": float(status.get("y", 0.0)),
                    "agv_yaw": float(status.get("yaw", 0.0)),
                    "agv_battery": float(status.get("battery", 0.0)),
                }
            except grpc.RpcError as exc:
                last_error = exc
                if exc.code() not in {grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}:
                    raise
                time.sleep(0.5)
        raise RuntimeError("qmlinker get base status failed") from last_error

    def get_agv_base_status(self) -> Any:
        """读取 AGV 底盘原始状态。"""

        return self._move_base.get_base_status()

    def get_agv_enable(self) -> bool:
        """读取 AGV 底盘使能状态。"""

        return bool(self._move_base.get_enable())

    def set_agv_enable(self, enabled: bool) -> bool:
        """设置 AGV 底盘使能状态。"""

        return bool(self._move_base.set_enable(bool(enabled)))

    def move_agv_translate(self, speed_mps: float, direction_deg: int, distance_m: float) -> bool:
        """按距离平移 AGV。"""

        return bool(
            self._move_base.translate_with_distance_sync(
                float(speed_mps),
                int(direction_deg),
                float(distance_m),
            )
        )

    def move_agv_rotate(self, angle_deg: float, direction: Any, speed_ratio: float) -> bool:
        """按角度旋转 AGV。"""

        return bool(self._move_base.rotate_with_angle_sync(float(angle_deg), direction, float(speed_ratio)))

    def stop_agv(self) -> bool:
        """停止 AGV。"""

        return bool(self._move_base.stop())

    def agv_real_time_translate(self, speed_mps: float, direction_deg: int) -> bool:
        """实时平移 AGV。"""

        return bool(self._move_base.real_time_translate(float(speed_mps), int(direction_deg)))

    def agv_real_time_rotate(self, speed_degps: float, direction: Any) -> bool:
        """实时旋转 AGV。"""

        return bool(self._move_base.real_time_rotate(float(speed_degps), direction))

    def agv_translate_with_distance(
        self,
        speed_mps: float = 0.1,
        direction_deg: int = 0,
        distance_m: float = 0.1,
    ) -> bool:
        """按距离平移 AGV。"""

        return bool(
            self._move_base.translate_with_distance_sync(
                float(speed_mps),
                int(direction_deg),
                float(distance_m),
            )
        )

    def agv_translate_with_distance_sync(
        self,
        speed_mps: float = 0.1,
        direction_deg: int = 0,
        distance_m: float = 0.1,
        acc_threshold: Any | None = None,
    ) -> bool:
        """同步按距离平移 AGV。"""

        return bool(
            self._move_base.translate_with_distance_sync(
                float(speed_mps),
                int(direction_deg),
                float(distance_m),
                acc_threshold,
            )
        )

    def agv_rotate_with_angle(
        self,
        speed_degps: float,
        direction: Any,
        angle_deg: float,
    ) -> bool:
        """按角度旋转 AGV。"""

        return bool(self._move_base.rotate_with_angle(float(speed_degps), direction, float(angle_deg)))

    def agv_rotate_with_angle_sync(
        self,
        speed_degps: float,
        direction: Any,
        angle_deg: float,
    ) -> bool:
        """同步按角度旋转 AGV。"""

        return bool(self._move_base.rotate_with_angle_sync(float(speed_degps), direction, float(angle_deg)))

    def agv_navigate_to(self, navi_target_name: str) -> Any:
        """发送 AGV 导航目标。"""

        return self._move_base.navigate_to(navi_target_name)

    def agv_navigate_to_sync(self, target_name: str) -> Any:
        """同步发送 AGV 导航目标。"""

        return self._move_base.navigate_to_sync(target_name)

    def describe_robot_runtime_structure(self) -> WujiRobotRuntimeStructure:
        """读取当前 qmlinker 连接对应的机器人结构快照。"""

        left_arm_limits = self._arm_joint_limits("left_arm")
        right_arm_limits = self._arm_joint_limits("right_arm")
        modules = [
            WujiRuntimeModuleSpec(
                tab_name="arm",
                title="left arm",
                device_name="left_arm",
                axes=tuple(
                    WujiRuntimeAxisSpec(
                        axis_name=f"left_j{index}",
                        minimum=limit.minimum_deg,
                        maximum=limit.maximum_deg,
                        unit=limit.unit,
                    )
                    for index, limit in enumerate(left_arm_limits, start=1)
                ),
            ),
            WujiRuntimeModuleSpec(
                tab_name="arm",
                title="right arm",
                device_name="right_arm",
                axes=tuple(
                    WujiRuntimeAxisSpec(
                        axis_name=f"right_j{index}",
                        minimum=limit.minimum_deg,
                        maximum=limit.maximum_deg,
                        unit=limit.unit,
                    )
                    for index, limit in enumerate(right_arm_limits, start=1)
                ),
            ),
            WujiRuntimeModuleSpec(
                tab_name="hand",
                title="right hand",
                device_name="right_hand",
                axes=tuple(
                    WujiRuntimeAxisSpec(
                        axis_name=spec.axis_name,
                        minimum=spec.minimum,
                        maximum=spec.maximum,
                        unit="",
                        control_supported=False,
                    )
                    for spec in RIGHT_HAND_ACTUATOR_SPECS
                ),
                enable_supported=False,
            ),
            WujiRuntimeModuleSpec(
                tab_name="body",
                title="body",
                device_name="body",
                axes=(
                    WujiRuntimeAxisSpec(
                        "body_z",
                        WUJI_BODY_AXIS_LIMITS["body_z"].minimum,
                        WUJI_BODY_AXIS_LIMITS["body_z"].maximum,
                        WUJI_BODY_AXIS_LIMITS["body_z"].unit,
                    ),
                    WujiRuntimeAxisSpec(
                        "body_ry",
                        WUJI_BODY_AXIS_LIMITS["body_ry"].minimum,
                        WUJI_BODY_AXIS_LIMITS["body_ry"].maximum,
                        WUJI_BODY_AXIS_LIMITS["body_ry"].unit,
                    ),
                ),
            ),
            WujiRuntimeModuleSpec(
                tab_name="body",
                title="head",
                device_name="head",
                axes=(
                    WujiRuntimeAxisSpec(
                        "head_yaw",
                        WUJI_HEAD_AXIS_LIMITS["head_yaw"].minimum,
                        WUJI_HEAD_AXIS_LIMITS["head_yaw"].maximum,
                        WUJI_HEAD_AXIS_LIMITS["head_yaw"].unit,
                    ),
                ),
            ),
            WujiRuntimeModuleSpec(
                tab_name="agv",
                title="AGV",
                device_name="agv",
                axes=(
                    WujiRuntimeAxisSpec("agv_x", -100.0, 100.0, "m", control_supported=False),
                    WujiRuntimeAxisSpec("agv_y", -100.0, 100.0, "m", control_supported=False),
                    WujiRuntimeAxisSpec("agv_yaw", -180.0, 180.0, "deg", control_supported=False),
                    WujiRuntimeAxisSpec("agv_battery", 0.0, 100.0, "%", control_supported=False),
                ),
                enable_supported=False,
            ),
        ]
        return WujiRobotRuntimeStructure(modules=tuple(modules))

    def _channel_target(self) -> str:
        """返回传给 qmlinker SDK 的 channel 目标。

        Returns
        -------
        str
            当配置使用 qmlinker 默认端口时返回纯 host，使 SDK 同时创建 `DEFAULT` 和 `DATA` 通道。
            自定义端口或 host 已显式包含端口时返回完整 target。
        """

        if ":" not in self._config.host and self._config.port == 50062:
            return self._config.host
        return self._config.target()

    def _connect_target(self) -> str:
        """返回最终 qmlinker 连接目标。

        Returns
        -------
        str
            直连可达时返回配置目标。工控机网段直连不可达时返回经 Orin SSH 隧道的本地目标。
        """

        return self._connect_target_for_port(
            self._config.host,
            self._config.port,
            passthrough_target=self._channel_target(),
        )

    def _connect_target_for_port(
        self,
        host: str,
        port: int,
        passthrough_target: str | None = None,
    ) -> str:
        """返回指定 host:port 的最终连接目标。

        Parameters
        ----------
        host:
            目标主机名或 IPv4 地址。
        port:
            目标 TCP 端口号。
        passthrough_target:
            当端口可直连时应直接返回的 target。为 `None` 时默认返回 `host:port`。

        Returns
        -------
        str
            直连可达时返回直接目标；工控机网段不可直连时返回经 Orin SSH 隧道映射的
            `127.0.0.1:local_port`。
        """

        direct_target = f"{host}:{port}" if passthrough_target is None else passthrough_target
        if self._can_connect_tcp(host, port):
            return direct_target
        if not host.startswith("192.168.100."):
            return direct_target
        forwarder = _SshTcpForwarder("orin", host, port)
        self._forwarders.append(forwarder)
        return forwarder.start()

    def _connect_orin_local_target(self, port: int) -> str:
        """返回 Orin 本地回环端口的转发目标。"""

        forwarder = _SshTcpForwarder(self._robot_network_config.orin_ssh_alias, "127.0.0.1", port)
        self._forwarders.append(forwarder)
        return forwarder.start()

    def _can_connect_tcp(self, host: str, port: int) -> bool:
        """检查 qmlinker TCP 端口是否可直连。

        Parameters
        ----------
        host:
            qmlinker 目标主机名或 IPv4 地址。
        port:
            qmlinker gRPC 端口，单位 TCP 端口号。

        Returns
        -------
        bool
            `True` 表示本机可在短超时内建立 TCP 连接。
        """

        if ":" in host:
            return True
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return True
        except OSError:
            return False

    def _module_stub(self, module_name: WujiQmlinkerEnableModuleName) -> Any:
        """返回非机械臂模块的 qmlinker gRPC stub。"""

        if module_name == "body":
            return self._lift_stub
        if module_name == "head":
            return self._head_stub
        raise ValueError(f"unsupported non-arm module: {module_name}")

    def _arm_device_name(self, module_name: WujiQmlinkerEnableModuleName) -> ArmDeviceName:
        """将整机模块名收窄为机械臂设备名。"""

        if module_name == "left_arm":
            return "left_arm"
        if module_name == "right_arm":
            return "right_arm"
        raise ValueError(f"module is not an arm device: {module_name}")

    @staticmethod
    def _require_float(value: object, field_name: str) -> float:
        """将 qmlinker 返回值收窄为浮点数。"""

        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"qmlinker camera intrinsics field {field_name} is not numeric: {value!r}")

    @staticmethod
    def _require_int(value: object, field_name: str) -> int:
        """将 qmlinker 返回值收窄为整数。"""

        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise TypeError(f"qmlinker camera intrinsics field {field_name} is not integer-like: {value!r}")

    @staticmethod
    def _require_float_sequence(value: object, field_name: str) -> tuple[float, ...]:
        """将 qmlinker 返回值收窄为浮点序列。"""

        if isinstance(value, (list, tuple)):
            return tuple(float(item) for item in value if isinstance(item, (int, float)))
        raise TypeError(f"qmlinker camera intrinsics field {field_name} is not a sequence: {value!r}")

    def _get_stub_enable(self, stub: Any) -> bool:
        """读取通用模块 stub 的使能状态。"""

        response = stub.GetEnabled(empty_pb2.Empty(), timeout=self._config.request_timeout_s)
        return bool(response.status.success and response.current_state == common_pb2.MODULE_ENABLED)

    def _set_stub_enable(self, stub: Any, enabled: bool) -> bool:
        """设置通用模块 stub 的使能状态。"""

        request = common_pb2.ModuleEnableRequest(enable=bool(enabled))
        response = stub.SetEnabled(request, timeout=self._config.request_timeout_s)
        return bool(response.status.success)

    def _arm(self, device_name: ArmDeviceName) -> Any:
        """返回指定机械臂的 qmlinker QMArm 实例。"""

        from src.wuji.arm_client import WujiArmClient

        arm = self._arms.get(device_name)
        if arm is not None:
            return arm
        arm = WujiArmClient(self, device_name)
        self._arms[device_name] = arm
        return arm

    def _hand(self, device_name: HandDeviceName) -> Any:
        """返回指定手部的 qmlinker QMHand 实例。"""

        hand = self._hands.get(device_name)
        if hand is not None:
            return hand
        hand = QMHand(self._channel, self._hand_pb_id(device_name))
        self._hands[device_name] = hand
        return hand

    def _hand_pb_id(self, device_name: HandDeviceName) -> Any:
        """返回 qmlinker proto 中的手部枚举值。"""

        return QMHand.HAND_RIGHT

    def _camera_type(self, camera_name: WujiCameraName) -> Any:
        """将项目相机名称转换为 qmlinker 相机枚举。"""

        if camera_name == "head_camera":
            return QMCamera.CAM_HEAD
        if camera_name == "chest_camera":
            return QMCamera.CAM_CHEST
        if camera_name == "left_hand_camera":
            return QMCamera.CAM_LEFT_HAND
        if camera_name == "right_hand_camera":
            return QMCamera.CAM_RIGHT_HAND
        raise ValueError(f"unsupported camera: {camera_name}")

    def _arm_joint_limits(self, device_name: ArmDeviceName) -> tuple[WujiArmJointLimit, ...]:
        """读取指定机械臂的运行时关节限位。"""

        arm = self._arm(device_name)
        joint_mins = list(getattr(arm.fkik, "joint_min", ()))
        joint_maxs = list(getattr(arm.fkik, "joint_max", ()))
        joint_count = min(len(joint_mins), len(joint_maxs))
        return tuple(
            WujiArmJointLimit(
                f"j{index}",
                float(np.rad2deg(joint_mins[index - 1])),
                float(np.rad2deg(joint_maxs[index - 1])),
            )
            for index in range(1, joint_count + 1)
        )

    def _hand_actuator_count(self, device_name: HandDeviceName) -> int:
        """读取指定手部执行器数量。"""

        request = hand_pb2.GetHandStateRequest()
        request.hand_id = self._hand_pb_id(device_name)
        request.include_tactile = False
        response = self._hand(device_name).stub.GetHandState(
            request,
            timeout=self._config.request_timeout_s,
        )
        return len(response.actuators)

    def get_hand_enable(self, device_name: HandDeviceName) -> bool:
        """读取指定手部使能状态。"""

        return bool(self._hand(device_name).get_enable())

    def get_hand_info(self, device_name: HandDeviceName) -> Any:
        """读取指定手部基础信息。"""

        return self._hand(device_name).get_hand_info()

    def set_tactile_data_read_enabled(self, device_name: HandDeviceName, enabled: bool) -> Any:
        """设置指定手部触觉数据读取开关。"""

        return self._hand(device_name).set_tactile_data_read_enabled(bool(enabled))

    def get_hand_state(self, device_name: HandDeviceName, include_tactile: bool = False) -> Any:
        """读取指定手部状态。"""

        return self._hand(device_name).get_hand_state(bool(include_tactile))

    def stream_get_hand_state(self, device_name: HandDeviceName) -> Any:
        """流式读取指定手部状态。"""

        return self._hand(device_name).stream_get_hand_state()

    def set_hand_enable(self, device_name: HandDeviceName, enabled: bool) -> bool:
        """设置指定手部使能状态。"""

        return bool(self._hand(device_name).set_enable(bool(enabled)))

    def set_hand_state(self, device_name: HandDeviceName, actuator_commands: list[dict[str, object]]) -> bool:
        """设置指定手部状态。"""

        return bool(self._hand(device_name).set_hand_state(actuator_commands))

    def stream_set_hand_state(self, device_name: HandDeviceName, command_frames: Any) -> Any:
        """流式设置指定手部状态。"""

        return self._hand(device_name).stream_set_hand_state(command_frames)

    def _arm_pb_type(self, device_name: ArmDeviceName) -> Any:
        """返回 qmlinker proto 中的机械臂枚举值。"""

        if device_name == "left_arm":
            return arm_pb2.ArmType.ARM_LEFT
        return arm_pb2.ArmType.ARM_RIGHT


# endregion
