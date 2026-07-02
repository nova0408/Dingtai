from __future__ import annotations

from collections.abc import Iterator

import pickle

import zmq

from .ball_pose_detection.protocol import BallPoseDetectionRequest, BallPoseDetectionResponse
from .camera_stream import CameraFramePacket
from .opening_detection.protocol import OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse
from .ports import CAMERA_PIPELINE_FRAME_STREAM_CONNECT_ADDR, CAMERA_PIPELINE_SERVICE_CONNECT_ADDR
from .tray_detection.protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse
from .unified_protocol import (
    CameraFrameSubscribeRequest,
    CameraFrameSubscribeResponse,
    CameraIntrinsicsRequest,
    CameraIntrinsicsResponse,
    CameraStatusRequest,
    CameraStatusResponse,
    CameraPipelineServiceRequest,
    CameraSummaryRequest,
    CameraSummaryResponse,
)
from .unified_transport import CameraPipelineRpcClient, ZmqSocketOptions


class CameraPipelineClient:
    """本机访问远端统一 camera pipeline 服务的唯一客户端。"""

    def __init__(
        self,
        service_addr: str = CAMERA_PIPELINE_SERVICE_CONNECT_ADDR,
        timeout_ms: int = 30_000,
    ) -> None:
        self._service_addr = str(service_addr)
        self._rpc_client = CameraPipelineRpcClient(
            connect_addr=self._service_addr,
            options=ZmqSocketOptions(
                recv_timeout_ms=int(timeout_ms),
                send_timeout_ms=int(timeout_ms),
            ),
        )

    def close(self) -> None:
        self._rpc_client.close()

    def get_camera_summary(self, timeout_s: float = 10.0) -> CameraSummaryResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_summary",
                camera_summary=CameraSummaryRequest(timeout_s=float(timeout_s)),
            )
        )
        if response.error is not None or response.camera_summary is None:
            raise RuntimeError(response.error or "camera summary response missing")
        return response.camera_summary

    def get_camera_intrinsics(self, timeout_s: float = 10.0) -> CameraIntrinsicsResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_intrinsics",
                camera_intrinsics=CameraIntrinsicsRequest(timeout_s=float(timeout_s)),
            )
        )
        if response.error is not None or response.camera_intrinsics is None:
            raise RuntimeError(response.error or "camera intrinsics response missing")
        return response.camera_intrinsics

    def get_camera_status(self, timeout_s: float = 10.0) -> CameraStatusResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_status",
                camera_status=CameraStatusRequest(timeout_s=float(timeout_s)),
            )
        )
        if response.error is not None or response.camera_status is None:
            raise RuntimeError(response.error or "camera status response missing")
        return response.camera_status

    def subscribe_camera_frames(self, camera_name: str = "left_hand_camera") -> Iterator[CameraFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_frame_subscribe",
                camera_frame_subscribe=CameraFrameSubscribeRequest(camera_name=str(camera_name)),
            )
        )
        if response.error is not None or response.camera_frame_subscribe is None:
            raise RuntimeError(response.error or "camera frame subscribe response missing")
        if response.camera_frame_subscribe.error is not None:
            raise RuntimeError(response.camera_frame_subscribe.error)
        stream_addr = response.camera_frame_subscribe.stream_addr
        service_host = self._service_addr.removeprefix("tcp://").split(":")[0]
        if stream_addr.startswith("tcp://0.0.0.0:") or stream_addr.startswith("tcp://127.0.0.1:"):
            stream_addr = stream_addr.replace("0.0.0.0", service_host, 1).replace("127.0.0.1", service_host, 1)
        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVTIMEO, 10_000)
        socket.connect(stream_addr or CAMERA_PIPELINE_FRAME_STREAM_CONNECT_ADDR)
        try:
            while True:
                yield pickle.loads(socket.recv())
        finally:
            socket.close(linger=0)

    def request_tray_detection(self, request: OrinTrayDetectionRequest) -> OrinTrayDetectionResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="tray_detection",
                tray_detection=request,
            )
        )
        if response.error is not None or response.tray_detection is None:
            raise RuntimeError(response.error or "tray detection response missing")
        return response.tray_detection

    def request_opening_detection(self, request: OpeningDetectionPipelineRequest) -> OpeningDetectionPipelineResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="opening_detection",
                opening_detection=request,
            )
        )
        if response.error is not None or response.opening_detection is None:
            raise RuntimeError(response.error or "opening detection response missing")
        return response.opening_detection

    def request_ball_pose_detection(self, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="ball_pose_detection",
                ball_pose_detection=request,
            )
        )
        if response.error is not None or response.ball_pose_detection is None:
            raise RuntimeError(response.error or "ball pose detection response missing")
        return response.ball_pose_detection
