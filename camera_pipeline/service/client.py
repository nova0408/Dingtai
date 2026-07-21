from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import zmq

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from ..protocol import CameraColorFramePacket, CameraDepthFramePacket, CameraFramePacket, CameraName
from .protocol import (
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraStatusResponse,
    CameraSummaryResponse,
    StableFrameResponse,
    CharucoDetectionRequest,
    CharucoDetectionResponse,
    build_camera_stream_topic,
)
from .transport import CameraPipelineRpcClient, ZmqSocketOptions
from .wire_codec import decode_wire

PacketT = TypeVar(
    "PacketT",
    bound=CameraFramePacket | CameraColorFramePacket | CameraDepthFramePacket,
)
_TCP_PREFIX = "tcp://"


class CameraPipelineClient:
    """外接开发机和 Orin 本地业务服务共用的公共客户端。"""

    def __init__(
        self,
        service_addr: str = "tcp://127.0.0.1:6200",
        timeout_ms: int = 30_000,
    ) -> None:
        self._service_addr = service_addr
        self._rpc_client = CameraPipelineRpcClient(
            connect_addr=service_addr,
            options=ZmqSocketOptions(
                receive_timeout_ms=timeout_ms,
                send_timeout_ms=timeout_ms,
            ),
        )

    def close(self) -> None:
        self._rpc_client.close()

    # region 相机查询

    def get_camera_summary(
        self,
        camera_name: CameraName,
        timeout_s: float = 10.0,
    ) -> CameraSummaryResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_summary",
                camera_name=camera_name,
                timeout_s=timeout_s,
            )
        )
        if response.error is not None or response.camera_summary is None:
            raise RuntimeError(response.error or "camera summary response missing")
        return response.camera_summary

    def get_camera_intrinsics(
        self,
        camera_name: CameraName,
        timeout_s: float = 10.0,
    ) -> CameraIntrinsicsResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_intrinsics",
                camera_name=camera_name,
                timeout_s=timeout_s,
            )
        )
        if response.error is not None or response.camera_intrinsics is None:
            raise RuntimeError(response.error or "camera intrinsics response missing")
        return response.camera_intrinsics

    def get_camera_status(
        self,
        camera_name: CameraName,
        timeout_s: float = 10.0,
    ) -> CameraStatusResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_status",
                camera_name=camera_name,
                timeout_s=timeout_s,
            )
        )
        if response.error is not None or response.camera_status is None:
            raise RuntimeError(response.error or "camera status response missing")
        return response.camera_status

    def get_stable_frame(
        self,
        camera_name: CameraName,
        timeout_s: float = 10.0,
    ) -> StableFrameResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="stable_frame",
                camera_name=camera_name,
                timeout_s=timeout_s,
            )
        )
        if response.error is not None or response.stable_frame is None:
            raise RuntimeError(response.error or "stable frame response missing")
        return response.stable_frame

    # endregion

    # region 帧订阅

    def subscribe_camera_frames(
        self, camera_name: CameraName
    ) -> Iterator[CameraFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_frame_subscribe",
                camera_name=camera_name,
            )
        )
        if response.error is not None or response.camera_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_frame_subscribe.stream_addr,
            camera_name,
            CameraFramePacket,
        )

    def subscribe_camera_color_frames(
        self,
        camera_name: CameraName,
    ) -> Iterator[CameraColorFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_color_frame_subscribe",
                camera_name=camera_name,
            )
        )
        if response.error is not None or response.camera_color_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera color frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_color_frame_subscribe.stream_addr,
            camera_name,
            CameraColorFramePacket,
        )

    def subscribe_camera_depth_frames(
        self,
        camera_name: CameraName,
    ) -> Iterator[CameraDepthFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_depth_frame_subscribe",
                camera_name=camera_name,
            )
        )
        if response.error is not None or response.camera_depth_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera depth frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_depth_frame_subscribe.stream_addr,
            camera_name,
            CameraDepthFramePacket,
        )

    def _subscribe_stream(
        self,
        stream_addr: str,
        camera_name: CameraName,
        packet_type: type[PacketT],
    ) -> Iterator[PacketT]:
        connect_addr = self._resolve_stream_connect_addr(stream_addr)
        topic = build_camera_stream_topic(camera_name)
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.setsockopt(zmq.RCVTIMEO, 10_000)
        socket.connect(connect_addr)
        try:
            while True:
                message = socket.recv()
                if not message.startswith(topic):
                    raise RuntimeError(f"camera stream topic mismatch: {camera_name}")
                yield decode_wire(message[len(topic) :], packet_type)
        finally:
            socket.close(linger=0)

    def _resolve_stream_connect_addr(self, stream_addr: str) -> str:
        service_addr = self._service_addr
        if service_addr.startswith(_TCP_PREFIX):
            service_addr = service_addr[len(_TCP_PREFIX) :]
        service_host = service_addr.split(":")[0]
        if stream_addr.startswith("tcp://0.0.0.0:") or stream_addr.startswith(
            "tcp://127.0.0.1:"
        ):
            return stream_addr.replace("0.0.0.0", service_host, 1).replace(
                "127.0.0.1", service_host, 1
            )
        return stream_addr or "tcp://127.0.0.1:6201"

    # endregion

    # region 算法请求

    def detect_ball(self, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="detect_ball",
                camera_name=request.camera_name,
                detect_ball=request,
            )
        )
        if response.error is not None or response.detect_ball is None:
            raise RuntimeError(response.error or "ball pose detection response missing")
        return response.detect_ball

    def detect_charuco(
        self,
        request: CharucoDetectionRequest,
    ) -> CharucoDetectionResponse:
        """使用完整 Board 协议请求检测。"""

        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="detect_charuco",
                camera_name=request.camera_name,
                detect_charuco=request,
            )
        )
        if response.error is not None or response.detect_charuco is None:
            raise RuntimeError(response.error or "charuco detection response missing")
        return response.detect_charuco

    # endregion
