from __future__ import annotations

from .ball_pose_detection.protocol import BallPoseDetectionRequest, BallPoseDetectionResponse
from .opening_detection.protocol import OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse
from .ports import CAMERA_PIPELINE_SERVICE_CONNECT_ADDR
from .tray_detection.protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse
from .unified_protocol import (
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
        self._rpc_client = CameraPipelineRpcClient(
            connect_addr=str(service_addr),
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
