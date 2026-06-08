from __future__ import annotations

import select
import socket
import socketserver
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
from google.protobuf import empty_pb2
import grpc
import numpy as np
from qmlinker import QMArm, QMCamera, QMHand, QMMoveBase, create_channel
from qmlinker.grpc_py import arm_pb2, common_pb2, hand_pb2, head_pb2, head_pb2_grpc
from qmlinker.grpc_py import lift_pb2, lift_pb2_grpc
from qmlinker.grpc_py import camera_pb2, camera_pb2_grpc
from qmlinker.grpc_py import waist_pb2, waist_pb2_grpc

from src.arm.wuji_arm_protocol import (
    ArmDeviceName,
    WUJI_BODY_AXIS_LIMITS,
    WUJI_HEAD_AXIS_LIMITS,
    WujiArmJointLimit,
)
from src.hand import HandDeviceName
from src.hand.wuji_hand_protocol import WujiHandInstanceSpec
from src.wuji.camera_protocol import (
    WujiCameraEnableState,
    WujiCameraFrame,
    WujiCameraIntrinsicsInfo,
    WujiCameraName,
)
from src.wuji.qmlinker_protocol import WujiQmlinkerConfig, WujiQmlinkerEnableModuleName
from src.wuji.qmlinker_protocol import WujiRobotRuntimeStructure, WujiRuntimeAxisSpec, WujiRuntimeModuleSpec

# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiArmJointState:
    """无际机械臂单关节状态。

    职责边界：
    - 只保存 qmlinker 返回的单关节遥测数据。
    - 不负责单位换算、GUI 展示或运动控制。

    设计思想：
    - 使用不可变 dataclass，便于跨线程从工作线程传回 GUI 线程。
    - 字段与 qmlinker 的 `arm_info.joint_states` 保持一致。

    生命周期：
    - 由一次状态读取构造，不持有网络连接或 qmlinker 对象。

    继承关系：
    - 不继承业务基类，作为协议结果数据使用。
    """

    joint_id: int
    "关节 ID，范围 1 到 6。"

    angle_deg: float
    "当前关节角度，单位 deg。"

    current_a: float
    "当前关节电流，单位 A。"

    power_w: float
    "当前关节功率，单位 W。"


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
    - 随 `WujiQmlinkerClient` 创建和关闭。
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
    - 随 `WujiQmlinkerClient` 构造。
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


class WujiQmlinkerClient:
    """无际 qmlinker 本机客户端。

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
        """初始化无际 qmlinker 客户端。

        Parameters
        ----------
        config:
            qmlinker 连接配置，为 `None` 时使用默认配置。
        """

        self._config = WujiQmlinkerConfig() if config is None else config
        self._forwarder: _SshTcpForwarder | None = None
        self._channel = create_channel(self._connect_target())
        self._arms: dict[ArmDeviceName, Any] = {}
        self._hands: dict[HandDeviceName, Any] = {}
        self._default_channel = self._channel["DEFAULT"] if isinstance(self._channel, dict) else self._channel
        self._camera = _WujiCameraServiceAdapter(self._channel, request_timeout_s=self._config.request_timeout_s)
        self._move_base = QMMoveBase(self._channel)
        self._waist_stub = waist_pb2_grpc.WaistServiceStub(self._default_channel)
        self._lift_stub = lift_pb2_grpc.LiftServiceStub(self._default_channel)
        self._head_stub = head_pb2_grpc.HeadServiceStub(self._default_channel)

    def close(self) -> None:
        """停止 qmlinker 内部状态线程。"""

        for arm in self._arms.values():
            if hasattr(arm, "running"):
                arm.running = False
            thread = getattr(arm, "thread_joint_states", None)
            if thread is not None:
                thread.join(timeout=0.5)
            if not hasattr(arm, "thread_arm_pose"):
                arm.thread_arm_pose = thread
        self._arms.clear()
        self._hands.clear()
        if self._forwarder is not None:
            self._forwarder.close()
            self._forwarder = None

    def check_ready(self) -> None:
        """检查 qmlinker gRPC 通道是否可创建并进入 ready。

        Raises
        ------
        TimeoutError
            在配置时间内 gRPC 通道无法 ready。
        """

        import grpc

        grpc.channel_ready_future(self._default_channel).result(timeout=self._config.request_timeout_s)

    def get_module_enable(self, module_name: WujiQmlinkerEnableModuleName) -> bool:
        """读取整机模块使能状态。

        Parameters
        ----------
        module_name:
            模块名，支持 `base`、`body`、`head`、`left_arm` 与 `right_arm`。

        Returns
        -------
        bool
            `True` 表示模块已使能。
        """

        if module_name in {"left_arm", "right_arm"}:
            return self.get_enable(self._arm_device_name(module_name))
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
            return self.set_enable(self._arm_device_name(module_name), enabled)
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

    def get_enable(self, device_name: ArmDeviceName) -> bool:
        """读取机械臂使能状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。

        Returns
        -------
        bool
            `True` 表示当前状态为已使能。
        """

        return bool(self._arm(device_name).get_enable())

    def set_enable(self, device_name: ArmDeviceName, enabled: bool) -> bool:
        """设置机械臂使能状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        enabled:
            目标使能状态。

        Returns
        -------
        bool
            `True` 表示设置请求成功。
        """

        return bool(self._arm(device_name).set_enable(enabled))

    def get_joint_states(self, device_name: ArmDeviceName) -> tuple[WujiArmJointState, ...]:
        """读取指定机械臂的一帧关节状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。

        Returns
        -------
        tuple[WujiArmJointState, ...]
            关节状态序列，通常长度为 6，角度单位为 deg。
        """

        arm = self._arm(device_name)
        arm_info = arm.get_arm_info(timeout=self._config.request_timeout_s)
        if arm_info is None or not arm_info.initialized:
            raise TimeoutError(f"qmlinker joint states not ready: {device_name}")
        return tuple(
            WujiArmJointState(
                joint_id=index,
                angle_deg=float(joint.angle_deg),
                current_a=float(joint.current_a),
                power_w=float(joint.power_w),
            )
            for index, joint in enumerate(arm_info.joint_states, start=1)
        )

    def set_joint(
        self,
        device_name: ArmDeviceName,
        joint_index: int,
        target_angle_deg: float,
    ) -> bool:
        """设置单个机械臂关节目标角度。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        joint_index:
            关节索引，范围 1 到 6。
        target_angle_deg:
            目标关节角度，单位 deg。

        Returns
        -------
        bool
            `True` 表示 qmlinker 接受该关节命令。
        """

        joints = self.get_joint_states(device_name)
        commands = [
            {
                "joint_id": joint.joint_id,
                "target_angle_deg": target_angle_deg if joint.joint_id == joint_index else joint.angle_deg,
                "speed_ratio": self._config.default_speed_ratio,
            }
            for joint in joints
        ]
        return bool(self._arm(device_name).set_joints(commands))

    def set_joints(
        self,
        device_name: ArmDeviceName,
        joint_commands: Iterable[dict[str, float | int]],
        sync_threshold: float = 0.0,
    ) -> bool:
        """批量设置机械臂关节目标角度。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        joint_commands:
            关节命令序列，每个元素包含 `joint_id`、`target_angle_deg` 和 `speed_ratio`。
        sync_threshold:
            qmlinker 同步阈值，单位 deg。为 0 时不等待同步。

        Returns
        -------
        bool
            `True` 表示 qmlinker 接受该批关节命令。
        """

        commands = [dict(command) for command in joint_commands]
        return bool(self._arm(device_name).set_joints(commands, sync_threshold=sync_threshold))

    def stream_get_joint_states(
        self,
        device_name: ArmDeviceName,
        duration_s: float,
    ) -> Iterator[tuple[WujiArmJointState, ...]]:
        """在指定时长内流式读取机械臂关节状态。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        duration_s:
            读取持续时间，单位 s。

        Yields
        ------
        tuple[WujiArmJointState, ...]
            每帧关节状态，角度单位为 deg。
        """

        deadline = None if duration_s <= 0.0 else time.time() + float(duration_s)
        request = arm_pb2.GetJointStatesRequest()
        request.arm_type = self._arm_pb_type(device_name)
        for response in self._arm(device_name).stub.StreamGetJointStates(request):
            if deadline is not None and time.time() > deadline:
                break
            yield tuple(
                WujiArmJointState(
                    joint_id=int(joint.joint_id),
                    angle_deg=float(joint.angle_deg),
                    current_a=float(joint.current_a),
                    power_w=float(joint.power_w),
                )
                for joint in response.joints
            )

    def stream_set_joint_states(
        self,
        device_name: ArmDeviceName,
        command_frames: Iterable[Iterable[dict[str, float | int]]],
    ) -> bool:
        """流式设置机械臂关节角度。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        command_frames:
            多帧关节命令。每帧为若干 `joint_id`、`target_angle_deg`、`speed_ratio` 字典。

        Returns
        -------
        bool
            `True` 表示流式发送结束。
        """

        frames = [[dict(command) for command in frame] for frame in command_frames]
        result = self._arm(device_name).stream_set_joint_states(frames)
        return result is None or bool(result)

    def fk(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> list[Any]:
        """执行 qmlinker 正向运动学。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        joint_angles_rad:
            6 轴关节角，单位 rad。

        Returns
        -------
        list[Any]
            qmlinker FK 返回的各关节变换矩阵列表。
        """

        return list(self._arm(device_name).fkik.fk(list(joint_angles_rad)))

    def fk_fast(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> Any:
        """执行 qmlinker 快速正向运动学，返回末端位姿矩阵。"""

        return self._arm(device_name).fkik.fk_fast(list(joint_angles_rad))

    def ik(
        self,
        device_name: ArmDeviceName,
        target_pose: Any,
        reference_joint_angles_rad: Iterable[float],
    ) -> tuple[float, ...]:
        """执行 qmlinker 逆向运动学。

        Parameters
        ----------
        device_name:
            机械臂设备名，取值为 `left_arm` 或 `right_arm`。
        target_pose:
            目标 4x4 齐次变换矩阵。
        reference_joint_angles_rad:
            参考 6 轴关节角，单位 rad。

        Returns
        -------
        tuple[float, ...]
            逆解关节角，单位 rad。无解时返回空元组。
        """

        result = self._arm(device_name).fkik.ik(target_pose, list(reference_joint_angles_rad))
        return tuple(float(value) for value in result)

    def current_fk_fast(self, device_name: ArmDeviceName) -> Any:
        """读取当前关节角并计算末端位姿矩阵。"""

        joints = self.get_joint_states(device_name)
        joint_angles_rad = [float(np.deg2rad(joint.angle_deg)) for joint in joints]
        return self.fk_fast(device_name, joint_angles_rad)

    def get_arm_joint_count(self, device_name: ArmDeviceName) -> int:
        """读取指定机械臂当前关节数量。"""

        return len(self.get_joint_states(device_name))

    def get_arm_joint_limits(self, device_name: ArmDeviceName) -> tuple[WujiArmJointLimit, ...]:
        """读取指定机械臂的运行时关节限位。

        Notes
        -----
        qmlinker `FKIKCalc.joint_min` 与 `joint_max` 在 wheel 中保存为弧度值；
        GUI 与 arm gRPC 接口统一使用 deg，因此这里显式执行 `rad -> deg` 转换。
        """

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

    def get_body_z(self) -> float:
        """读取身体 z 轴升降高度。

        Returns
        -------
        float
            当前升降高度，单位 mm。
        """

        response = self._lift_stub.GetCurrentLiftPhysicalHeight(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return float(response.current_height_mm)

    def set_body_z(self, height_mm: float) -> bool:
        """设置身体 z 轴升降高度，单位 mm。"""

        request = lift_pb2.SetLiftPhysicalHeightRequest(height_mm=int(round(height_mm)))
        response = self._lift_stub.SetLiftPhysicalHeight(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_body_ry(self) -> float:
        """读取身体 Ry 俯仰角，单位 deg。"""

        response = self._waist_stub.GetCurrentPitch(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return float(response.current_pitch_deg)

    def set_body_ry(self, pitch_deg: float) -> bool:
        """设置身体 Ry 俯仰角，单位 deg。"""

        request = waist_pb2.SetWaistPitchRequest(pitch_angle_deg=float(pitch_deg))
        response = self._waist_stub.SetPitchAngle(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_head_yaw(self) -> float:
        """读取头部旋转轴角度，单位 deg。"""

        response = self._head_stub.GetHeadYaw(
            empty_pb2.Empty(),
            timeout=self._config.request_timeout_s,
        )
        return float(response.current_yaw_deg)

    def set_head_yaw(self, yaw_deg: float) -> bool:
        """设置头部旋转轴角度，单位 deg。"""

        request = head_pb2.SetHeadYawRequest(yaw_angle_deg=float(yaw_deg))
        response = self._head_stub.SetHeadYaw(
            request,
            timeout=self._config.request_timeout_s,
        )
        return bool(response.status.success)

    def get_agv_status_values(self) -> dict[str, float]:
        """读取 AGV 底盘基础状态值。"""

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

    def get_hand_values(self, device_name: HandDeviceName) -> dict[str, float]:
        """读取指定手部执行器位置。"""

        request = hand_pb2.GetHandStateRequest()
        request.hand_id = self._hand_pb_id(device_name)
        request.include_tactile = False
        response = self._hand(device_name).stub.GetHandState(
            request,
            timeout=self._config.request_timeout_s,
        )
        return self._coerce_hand_state_response(device_name, response)

    def get_hand_actuator_count(self, device_name: HandDeviceName) -> int:
        """读取指定手部当前执行器数量。"""

        return len(self.get_hand_values(device_name))

    def get_hand_instance_specs(self) -> tuple[WujiHandInstanceSpec, ...]:
        """读取左右手当前执行器规格。"""

        return (
            WujiHandInstanceSpec("left_hand", "left hand", self.get_hand_actuator_count("left_hand")),
            WujiHandInstanceSpec("right_hand", "right hand", self.get_hand_actuator_count("right_hand")),
        )

    def describe_robot_runtime_structure(self) -> WujiRobotRuntimeStructure:
        """读取当前 qmlinker 连接对应的机器人结构快照。"""

        left_arm_limits = self.get_arm_joint_limits("left_arm")
        right_arm_limits = self.get_arm_joint_limits("right_arm")
        left_hand_count = self.get_hand_actuator_count("left_hand")
        right_hand_count = self.get_hand_actuator_count("right_hand")
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
                title="left hand",
                device_name="left_hand",
                axes=tuple(
                    WujiRuntimeAxisSpec(axis_name=f"left_hand_a{idx}", minimum=0.0, maximum=1.0, unit="", control_supported=False)
                    for idx in range(left_hand_count)
                ),
                enable_supported=False,
            ),
            WujiRuntimeModuleSpec(
                tab_name="hand",
                title="right hand",
                device_name="right_hand",
                axes=tuple(
                    WujiRuntimeAxisSpec(axis_name=f"right_hand_a{idx}", minimum=0.0, maximum=1.0, unit="", control_supported=False)
                    for idx in range(right_hand_count)
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

    def stream_get_hand_values(self, device_name: HandDeviceName) -> Iterator[dict[str, float]]:
        """流式读取指定手部执行器位置。"""

        for state in self._hand(device_name).stream_get_hand_state(include_tactile=False):
            yield self._coerce_hand_state_values(device_name, state)

    def get_camera_enable_state(self, camera_name: WujiCameraName) -> WujiCameraEnableState:
        """读取指定相机使能状态与接口可用性。

        Parameters
        ----------
        camera_name:
            无际相机逻辑名称。

        Returns
        -------
        state:
            相机使能状态结果；若服务端未实现该接口，则 `api_available=False`。
        """

        try:
            enabled = bool(self._camera.get_enable(self._camera_type(camera_name)))
            return WujiCameraEnableState(
                camera_name=camera_name,
                enabled=enabled,
                api_available=True,
                message="qmlinker camera enable api available",
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                return WujiCameraEnableState(
                    camera_name=camera_name,
                    enabled=False,
                    api_available=False,
                    message=f"qmlinker camera enable api unavailable: {camera_name}",
                )
            raise

    def set_camera_enable(self, camera_name: WujiCameraName, enabled: bool) -> bool:
        """设置指定相机使能状态。

        Parameters
        ----------
        camera_name:
            无际相机逻辑名称。
        enabled:
            目标使能状态。

        Returns
        -------
        success:
            `True` 表示 qmlinker 接口返回成功。
        """

        try:
            return bool(self._camera.set_enable(self._camera_type(camera_name), bool(enabled)))
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise NotImplementedError(f"qmlinker camera enable api unavailable: {camera_name}") from exc
            raise

    def get_camera_intrinsics(self, camera_name: WujiCameraName) -> WujiCameraIntrinsicsInfo:
        """读取指定相机内参与基准分辨率。

        Parameters
        ----------
        camera_name:
            无际相机逻辑名称。

        Returns
        -------
        intrinsics:
            qmlinker 相机内参与基准分辨率。

        Raises
        ------
        RuntimeError
            qmlinker 未返回有效内参时抛出。
        """

        raw = self._camera.get_camera_intrinsics(self._camera_type(camera_name))
        if not isinstance(raw, dict):
            raise RuntimeError(f"qmlinker get camera intrinsics failed: {camera_name}")
        fx_value = self._require_float(raw.get("fx", 0.0), "fx")
        fy_value = self._require_float(raw.get("fy", 0.0), "fy")
        cx_value = self._require_float(raw.get("cx", 0.0), "cx")
        cy_value = self._require_float(raw.get("cy", 0.0), "cy")
        distortion_value = self._require_float_sequence(raw.get("distortion", ()), "distortion")
        base_width_value = self._require_int(raw.get("base_width", 0), "base_width")
        base_height_value = self._require_int(raw.get("base_height", 0), "base_height")
        return WujiCameraIntrinsicsInfo(
            camera_name=camera_name,
            fx=fx_value,
            fy=fy_value,
            cx=cx_value,
            cy=cy_value,
            distortion=distortion_value,
            width=base_width_value,
            height=base_height_value,
        )

    def stream_camera_rgb_frames(self, camera_name: WujiCameraName) -> Iterator[WujiCameraFrame]:
        """流式读取指定相机 2D 彩色图像。

        Parameters
        ----------
        camera_name:
            无际相机逻辑名称。

        Yields
        ------
        frame:
            qmlinker 2D 图像帧，depth 固定为 `None`。
        """

        for item in self._camera.stream_get_image_2d(self._camera_type(camera_name)):
            if not isinstance(item, tuple) or len(item) != 2:
                raise RuntimeError(f"qmlinker rgb stream failed: {camera_name}")
            color_bgr, timestamp = item
            yield WujiCameraFrame(
                camera_name=camera_name,
                color_bgr=np.asarray(color_bgr, dtype=np.uint8).copy(),
                timestamp=timestamp,
                sequence_id=None,
            )

    def stream_camera_rgbd_frames(self, camera_name: WujiCameraName) -> Iterator[WujiCameraFrame]:
        """流式读取指定相机 RGBD 图像。

        Parameters
        ----------
        camera_name:
            无际相机逻辑名称。

        Yields
        ------
        frame:
            qmlinker RGBD 图像帧，包含 BGR 彩色图和深度矩阵。
        """

        if not self._camera.control_depth_stream(self._camera_type(camera_name), True):
            raise RuntimeError(f"qmlinker enable depth stream failed: {camera_name}")
        for item in self._camera.stream_get_rgbd_image(self._camera_type(camera_name)):
            if not isinstance(item, tuple) or len(item) != 3:
                raise RuntimeError(f"qmlinker rgbd stream failed: {camera_name}")
            color_bgr, timestamp, depth = item
            yield WujiCameraFrame(
                camera_name=camera_name,
                color_bgr=np.asarray(color_bgr, dtype=np.uint8).copy(),
                timestamp=timestamp,
                sequence_id=None,
                depth=np.asarray(depth).copy(),
            )

    def stop_camera_depth_stream(self, camera_name: WujiCameraName) -> None:
        """请求 qmlinker 停止指定相机深度流生成。"""

        self._camera.control_depth_stream(self._camera_type(camera_name), False)

    def _coerce_hand_state_values(self, device_name: HandDeviceName, state: object) -> dict[str, float]:
        """将 qmlinker 手部状态转换为 GUI 轴值字典。"""

        if not isinstance(state, dict) or not isinstance(state.get("actuators"), list):
            raise RuntimeError(f"qmlinker get hand state failed: {device_name}")
        values: dict[str, float] = {}
        for actuator in state["actuators"]:
            if not isinstance(actuator, dict):
                continue
            actuator_id = int(actuator.get("actuator_id", -1))
            if actuator_id < 0:
                continue
            values[f"{device_name}_a{actuator_id}"] = float(actuator.get("position", 0.0))
        return values

    def _coerce_hand_state_response(self, device_name: HandDeviceName, state: Any) -> dict[str, float]:
        """将 qmlinker 手部 proto 状态转换为 GUI 轴值字典。

        Parameters
        ----------
        device_name:
            手部设备名，取值为 `left_hand` 或 `right_hand`。
        state:
            qmlinker `HandState` proto 响应，包含执行器状态序列。

        Returns
        -------
        dict[str, float]
            GUI 轴名到执行器位置的映射，位置单位沿用 qmlinker 归一化比例。
        """

        values: dict[str, float] = {}
        for actuator in state.actuators:
            actuator_id = int(actuator.actuator_id)
            if actuator_id < 0:
                continue
            values[f"{device_name}_a{actuator_id}"] = float(actuator.position)
        return values

    def _arm(self, device_name: ArmDeviceName) -> Any:
        """返回指定机械臂的 qmlinker QMArm 实例。"""

        arm = self._arms.get(device_name)
        if arm is not None:
            return arm
        arm_type: Any = QMArm.ARM_LEFT if device_name == "left_arm" else QMArm.ARM_RIGHT
        arm = QMArm(self._channel, arm_type)
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

        if device_name == "left_hand":
            return QMHand.HAND_LEFT
        return QMHand.HAND_RIGHT

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

        if self._can_connect_tcp(self._config.host, self._config.port):
            return self._channel_target()
        if not self._config.host.startswith("192.168.100."):
            return self._channel_target()
        self._forwarder = _SshTcpForwarder("orin", self._config.host, self._config.port)
        return self._forwarder.start()

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

    def _arm_pb_type(self, device_name: ArmDeviceName) -> Any:
        """返回 qmlinker proto 中的机械臂枚举值。"""

        if device_name == "left_arm":
            return arm_pb2.ArmType.ARM_LEFT
        return arm_pb2.ArmType.ARM_RIGHT

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


# endregion
