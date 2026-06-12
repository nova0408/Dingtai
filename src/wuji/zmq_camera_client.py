from __future__ import annotations
# pyright: reportMissingImports=false

from collections.abc import Iterator

from src.wuji.camera_protocol import WujiCameraFrame, WujiCameraIntrinsicsInfo, WujiCameraName
from src.wuji.zmq_camera_catalog import (
    SUPPORTED_WUJI_ZMQ_CAMERAS,
    WujiZmqCameraEndpoint,
    WujiZmqCameraStatus,
    get_wuji_zmq_camera_endpoint,
)
from src.wuji.zmq_camera_transport import WujiZmqCameraTransport

# region 配置

DEFAULT_WUJI_ZMQ_CAMERA_HOST = "192.168.100.60"
"默认 ZMQ 相机服务主机地址。"

DEFAULT_WUJI_ZMQ_CAMERA_CONTROL_PORT = 5570
"默认 ZMQ 相机控制口端口号。"

DEFAULT_WUJI_ZMQ_CAMERA_REQUEST_TIMEOUT_MS = 3000
"控制命令超时，单位 ms。"

DEFAULT_WUJI_ZMQ_CAMERA_STREAM_TIMEOUT_MS = 5000
"数据流首帧等待超时，单位 ms。"

# endregion


# region 主入口


class WujiZmqCameraClient:
    """无际 ZMQ 相机客户端。

    职责边界：
    - 负责访问 `sensors_depthcamera_ob_zmq_v2` 的控制口与数据口。
    - 只提供单相机状态查询、参数查询和数据流读取能力。
    - 不负责远端 SSH 探测、端口转发、GUI 展示模型或多相机资产发现。

    设计思想：
    - 将纯 ZMQ 传输实现下沉到独立代码页，本类只保留相机语义适配。
    - 多相机清单与端点映射放到独立 catalog 代码页，避免继续和 client 糅杂。

    生命周期：
    - 可被 GUI backend 或测试脚本长期持有。
    - `close()` 只负责关闭内部传输器。

    继承关系：
    - 不继承业务基类，作为相机协议适配器使用。
    """

    def __init__(
        self,
        host: str = DEFAULT_WUJI_ZMQ_CAMERA_HOST,
        *,
        control_port: int = DEFAULT_WUJI_ZMQ_CAMERA_CONTROL_PORT,
        request_timeout_ms: int = DEFAULT_WUJI_ZMQ_CAMERA_REQUEST_TIMEOUT_MS,
        stream_timeout_ms: int = DEFAULT_WUJI_ZMQ_CAMERA_STREAM_TIMEOUT_MS,
        camera_endpoints: tuple[WujiZmqCameraEndpoint, ...] = SUPPORTED_WUJI_ZMQ_CAMERAS,
    ) -> None:
        """创建 ZMQ 相机客户端。

        Parameters
        ----------
        host:
            ZMQ 相机服务主机地址。本机 SSH 调试时通常为 `127.0.0.1`。
        control_port:
            控制口端口号。本机 SSH 调试时可显式传入转发后的本地控制口。
        request_timeout_ms:
            控制命令超时，单位 ms。
        stream_timeout_ms:
            数据流首帧等待超时，单位 ms。
        camera_endpoints:
            当前调用方采用的静态相机端点表。默认使用现场部署端点；本机 SSH 调试时可显式
            传入 `SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL`。
        """

        self._transport = WujiZmqCameraTransport(
            str(host),
            control_port=int(control_port),
            request_timeout_ms=int(request_timeout_ms),
            stream_timeout_ms=int(stream_timeout_ms),
        )
        self._camera_endpoints = tuple(camera_endpoints)

    def close(self) -> None:
        """关闭底层传输器。"""

        self._transport.close()

    def get_camera_status(self, camera_name: WujiCameraName) -> WujiZmqCameraStatus:
        """读取指定相机的在线与流开关状态。"""

        endpoint = get_wuji_zmq_camera_endpoint(camera_name, supported_endpoints=self._camera_endpoints)
        payload = self._transport.send_control_command(endpoint.camera_id, "get_status")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid get_status payload: {payload!r}")
        return WujiZmqCameraStatus(
            camera_name=camera_name,
            online=bool(data.get("online", False)),
            color_enabled=bool(data.get("color_enabled", False)),
            depth_enabled=bool(data.get("depth_enabled", False)),
        )

    def set_camera_color_enabled(self, camera_name: WujiCameraName, enabled: bool) -> WujiZmqCameraStatus:
        """设置彩色流开关并返回最新状态。"""

        endpoint = get_wuji_zmq_camera_endpoint(camera_name, supported_endpoints=self._camera_endpoints)
        self._transport.send_control_command(
            endpoint.camera_id,
            "set_color_enabled",
            {"enable": bool(enabled)},
        )
        return self.get_camera_status(camera_name)

    def set_camera_depth_enabled(self, camera_name: WujiCameraName, enabled: bool) -> WujiZmqCameraStatus:
        """设置深度流开关并返回最新状态。"""

        endpoint = get_wuji_zmq_camera_endpoint(camera_name, supported_endpoints=self._camera_endpoints)
        self._transport.send_control_command(
            endpoint.camera_id,
            "set_depth_enabled",
            {"enable": bool(enabled)},
        )
        return self.get_camera_status(camera_name)

    def get_camera_intrinsics(self, camera_name: WujiCameraName) -> WujiCameraIntrinsicsInfo:
        """读取指定相机内参与默认分辨率。"""

        endpoint = get_wuji_zmq_camera_endpoint(camera_name, supported_endpoints=self._camera_endpoints)
        payload = self._transport.send_control_command(endpoint.camera_id, "get_intrinsics")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid get_intrinsics payload: {payload!r}")
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

    def stream_camera_rgb_frames(self, camera_name: WujiCameraName) -> Iterator[WujiCameraFrame]:
        """流式读取指定相机的 RGB 图像。"""

        endpoint = get_wuji_zmq_camera_endpoint(camera_name, supported_endpoints=self._camera_endpoints)
        yield from self._transport.stream_frames(
            camera_name=camera_name,
            stream_port=endpoint.stream_port,
            expect_depth=False,
        )

    def stream_camera_rgbd_frames(self, camera_name: WujiCameraName) -> Iterator[WujiCameraFrame]:
        """流式读取指定相机的 RGBD 图像。"""

        endpoint = get_wuji_zmq_camera_endpoint(camera_name, supported_endpoints=self._camera_endpoints)
        self.set_camera_depth_enabled(camera_name, True)
        yield from self._transport.stream_frames(
            camera_name=camera_name,
            stream_port=endpoint.stream_port,
            expect_depth=True,
        )

    def stop_camera_depth_stream(self, camera_name: WujiCameraName) -> None:
        """关闭指定相机的深度流。"""

        self.set_camera_depth_enabled(camera_name, False)


# endregion
