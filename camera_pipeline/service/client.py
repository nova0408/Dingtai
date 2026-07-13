from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar, cast

import zmq

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from ..opening_detection.protocol import (
    OpeningDetectionPipelineRequest,
    OpeningDetectionPipelineResponse,
)
from ..protocol import CameraColorFramePacket, CameraDepthFramePacket, CameraFramePacket
from ..tray_detection.protocol import (
    OrinTrayDetectionRequest,
    OrinTrayDetectionResponse,
)
from .protocol import (
    CameraColorFrameSubscribeRequest,
    CameraDepthFrameSubscribeRequest,
    CameraFrameSubscribeRequest,
    CameraIntrinsicsRequest,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraStatusRequest,
    CameraStatusResponse,
    CameraSummaryRequest,
    CameraSummaryResponse,
    StableFrameRequest,
    StableFrameResponse,
)
from .transport import CameraPipelineRpcClient, ZmqSocketOptions
from .wire_codec import decode_wire

PacketT = TypeVar("PacketT")
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

    def get_camera_summary(self, timeout_s: float = 10.0) -> CameraSummaryResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_summary",
                camera_summary=CameraSummaryRequest(timeout_s=timeout_s),
            )
        )
        if response.error is not None or response.camera_summary is None:
            raise RuntimeError(response.error or "camera summary response missing")
        return response.camera_summary

    def get_camera_intrinsics(
        self, timeout_s: float = 10.0
    ) -> CameraIntrinsicsResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_intrinsics",
                camera_intrinsics=CameraIntrinsicsRequest(timeout_s=timeout_s),
            )
        )
        if response.error is not None or response.camera_intrinsics is None:
            raise RuntimeError(response.error or "camera intrinsics response missing")
        return response.camera_intrinsics

    def get_camera_status(self, timeout_s: float = 10.0) -> CameraStatusResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_status",
                camera_status=CameraStatusRequest(timeout_s=timeout_s),
            )
        )
        if response.error is not None or response.camera_status is None:
            raise RuntimeError(response.error or "camera status response missing")
        return response.camera_status

    def get_stable_frame(self, timeout_s: float = 10.0) -> StableFrameResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="stable_frame",
                stable_frame=StableFrameRequest(timeout_s=timeout_s),
            )
        )
        if response.error is not None or response.stable_frame is None:
            raise RuntimeError(response.error or "stable frame response missing")
        return response.stable_frame

    # endregion

    # region 帧订阅

    def subscribe_camera_frames(
        self, camera_name: str = "left_hand_camera"
    ) -> Iterator[CameraFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_frame_subscribe",
                camera_frame_subscribe=CameraFrameSubscribeRequest(
                    camera_name=camera_name
                ),
            )
        )
        if response.error is not None or response.camera_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_frame_subscribe.stream_addr, CameraFramePacket
        )

    def subscribe_camera_color_frames(
        self,
        camera_name: str = "left_hand_camera",
    ) -> Iterator[CameraColorFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_color_frame_subscribe",
                camera_color_frame_subscribe=CameraColorFrameSubscribeRequest(
                    camera_name=camera_name
                ),
            )
        )
        if response.error is not None or response.camera_color_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera color frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_color_frame_subscribe.stream_addr, CameraColorFramePacket
        )

    def subscribe_camera_depth_frames(
        self,
        camera_name: str = "left_hand_camera",
    ) -> Iterator[CameraDepthFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_depth_frame_subscribe",
                camera_depth_frame_subscribe=CameraDepthFrameSubscribeRequest(
                    camera_name=camera_name
                ),
            )
        )
        if response.error is not None or response.camera_depth_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera depth frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_depth_frame_subscribe.stream_addr, CameraDepthFramePacket
        )

    def _subscribe_stream(
        self, stream_addr: str, packet_type: type[PacketT]
    ) -> Iterator[PacketT]:
        connect_addr = self._resolve_stream_connect_addr(stream_addr)
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVTIMEO, 10_000)
        socket.connect(connect_addr)
        try:
            while True:
                packet = decode_wire(socket.recv(), packet_type)
                yield cast(PacketT, packet)
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

    def request_tray_detection(
        self, request: OrinTrayDetectionRequest
    ) -> OrinTrayDetectionResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="tray_detection", tray_detection=request
            )
        )
        if response.error is not None or response.tray_detection is None:
            raise RuntimeError(response.error or "tray detection response missing")
        return response.tray_detection

    def request_opening_detection(
        self,
        request: OpeningDetectionPipelineRequest,
    ) -> OpeningDetectionPipelineResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="opening_detection", opening_detection=request
            )
        )
        if response.error is not None or response.opening_detection is None:
            raise RuntimeError(response.error or "opening detection response missing")
        return response.opening_detection

    def request_ball_pose_detection(
        self, request: BallPoseDetectionRequest
    ) -> BallPoseDetectionResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="ball_pose_detection", ball_pose_detection=request
            )
        )
        if response.error is not None or response.ball_pose_detection is None:
            raise RuntimeError(response.error or "ball pose detection response missing")
        return response.ball_pose_detection

    # endregion
