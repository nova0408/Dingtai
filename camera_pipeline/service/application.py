from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..ball_pose_detection.protocol import BallPoseDetectionResponse
from ..protocol import RgbdFrameProtocol
from ..pipeline_context import PipelineContext
from .frame_publisher import CameraFramePublisher
from .protocol import (
    PROTOCOL_VERSION,
    CameraColorFrameSubscribeResponse,
    CameraDepthFrameSubscribeResponse,
    CameraFrameSubscribeResponse,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CameraStatusResponse,
    CameraSummaryResponse,
    StableFrameResponse,
)

if TYPE_CHECKING:
    from ..ball_pose_detection.service import BallPoseDetectionService


class CameraPipelineApplication:
    """编排 CameraPipeline 请求、帧选择和纯计算算法调用。

    本类不创建网络请求 socket、不运行服务主循环，也不直接访问相机
    上游协议。所有帧访问经由 `PipelineContext`，所有网络发布经由
    `CameraFramePublisher`。
    """

    def __init__(
        self,
        pipeline_context: PipelineContext,
        frame_publisher: CameraFramePublisher,
    ) -> None:
        self._pipeline_context = pipeline_context
        self._frame_publisher = frame_publisher
        self._ball_service: BallPoseDetectionService | None = None

    # region 请求分发

    def handle(
        self, request: CameraPipelineServiceRequest
    ) -> CameraPipelineServiceResponse:
        """显式分发一个经过传输层解码的请求。"""

        if request.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported protocol version {request.protocol_version}; expected {PROTOCOL_VERSION}"
            )
        if request.operation == "camera_summary":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_summary=self._handle_camera_summary(request),
            )
        if request.operation == "camera_intrinsics":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_intrinsics=self._handle_camera_intrinsics(request),
            )
        if request.operation == "camera_status":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_status=self._handle_camera_status(request),
            )
        if request.operation == "stable_frame":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                stable_frame=self._handle_stable_frame(request),
            )
        if request.operation == "camera_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_frame_subscribe=self._handle_frame_subscribe(request),
            )
        if request.operation == "camera_color_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_color_frame_subscribe=self._handle_color_frame_subscribe(
                    request
                ),
            )
        if request.operation == "camera_depth_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_depth_frame_subscribe=self._handle_depth_frame_subscribe(
                    request
                ),
            )
        if request.operation == "ball_pose_detection":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                ball_pose_detection=self._handle_ball_pose_detection(request),
            )
        raise RuntimeError(f"unsupported operation: {request.operation}")

    # endregion

    # region 相机查询与稳定帧

    def _handle_camera_summary(
        self, request: CameraPipelineServiceRequest
    ) -> CameraSummaryResponse:
        payload = request.camera_summary
        if payload is None:
            raise RuntimeError("camera_summary payload missing")
        frame = self._wait_for_latest_frame(payload.timeout_s, "camera first frame")
        logger.info(
            "api camera_summary resolved camera_name={} frame_id={} color_shape={} depth_shape={}",
            frame.camera_name,
            frame.frame_id,
            frame.color_bgr.shape,
            frame.depth_mm.shape,
        )
        return CameraSummaryResponse(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
            color_shape=(
                frame.color_bgr.shape[0],
                frame.color_bgr.shape[1],
                frame.color_bgr.shape[2],
            ),
            depth_shape=(frame.depth_mm.shape[0], frame.depth_mm.shape[1]),
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
        )

    def _handle_camera_intrinsics(
        self, request: CameraPipelineServiceRequest
    ) -> CameraIntrinsicsResponse:
        payload = request.camera_intrinsics
        if payload is None:
            raise RuntimeError("camera_intrinsics payload missing")
        frame = self._wait_for_latest_frame(
            payload.timeout_s,
            "camera intrinsics",
            camera_name=payload.camera_name,
        )
        logger.info(
            "api camera_intrinsics resolved camera_name={} frame_id={} size={}x{} distortion_count={}",
            frame.camera_name,
            frame.frame_id,
            frame.color_bgr.shape[1],
            frame.color_bgr.shape[0],
            len(frame.distortion),
        )
        return CameraIntrinsicsResponse(
            camera_name=frame.camera_name,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
            distortion=frame.distortion,
            width=frame.color_bgr.shape[1],
            height=frame.color_bgr.shape[0],
        )

    def _handle_camera_status(
        self, request: CameraPipelineServiceRequest
    ) -> CameraStatusResponse:
        payload = request.camera_status
        if payload is None:
            raise RuntimeError("camera_status payload missing")
        frame = self._wait_for_latest_frame(payload.timeout_s, "camera status")
        logger.info(
            "api camera_status resolved camera_name={} frame_id={} online=True",
            frame.camera_name,
            frame.frame_id,
        )
        return CameraStatusResponse(
            camera_name=frame.camera_name,
            camera_id=self._pipeline_context.get_camera_id(frame.camera_name),
            camera_model="unknown",
            width=frame.color_bgr.shape[1],
            height=frame.color_bgr.shape[0],
            color_enabled=True,
            depth_enabled=True,
            online=True,
        )

    def _handle_stable_frame(
        self, request: CameraPipelineServiceRequest
    ) -> StableFrameResponse:
        payload = request.stable_frame
        if payload is None:
            raise RuntimeError("stable_frame payload missing")
        frame = self._pipeline_context.wait_for_stable_frame(
            timeout_s=payload.timeout_s,
            camera_name=payload.camera_name,
        )
        logger.info(
            "api stable_frame resolved camera_name={} frame_id={} timeout_s={:.3f}",
            frame.camera_name,
            frame.frame_id,
            payload.timeout_s,
        )
        return StableFrameResponse(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
        )

    def _wait_for_latest_frame(
        self,
        timeout_s: float,
        description: str,
        camera_name: str | None = None,
    ) -> RgbdFrameProtocol:
        if not self._pipeline_context.wait_until_ready(
            timeout_s=timeout_s,
            camera_name=camera_name,
        ):
            raise RuntimeError(f"{description} not ready within {timeout_s:.1f}s")
        frame = self._pipeline_context.get_latest_frame(camera_name)
        if frame is None:
            raise RuntimeError(f"{description} unavailable")
        return frame

    # endregion

    # region 帧订阅

    def _handle_frame_subscribe(
        self, request: CameraPipelineServiceRequest
    ) -> CameraFrameSubscribeResponse:
        payload = request.camera_frame_subscribe
        if payload is None:
            raise RuntimeError("camera_frame_subscribe payload missing")
        self._pipeline_context.get_camera_runtime(payload.camera_name)
        self._frame_publisher.start()
        logger.info(
            "api camera_frame_subscribe ready camera_name={} stream_addr={}",
            payload.camera_name,
            self._frame_publisher.frame_bind_addr,
        )
        return CameraFrameSubscribeResponse(
            stream_addr=self._frame_publisher.frame_bind_addr,
            camera_name=payload.camera_name,
        )

    def _handle_color_frame_subscribe(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CameraColorFrameSubscribeResponse:
        payload = request.camera_color_frame_subscribe
        if payload is None:
            raise RuntimeError("camera_color_frame_subscribe payload missing")
        self._pipeline_context.get_camera_runtime(payload.camera_name)
        self._frame_publisher.start()
        logger.info(
            "api camera_color_frame_subscribe ready camera_name={} stream_addr={}",
            payload.camera_name,
            self._frame_publisher.color_bind_addr,
        )
        return CameraColorFrameSubscribeResponse(
            stream_addr=self._frame_publisher.color_bind_addr,
            camera_name=payload.camera_name,
        )

    def _handle_depth_frame_subscribe(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CameraDepthFrameSubscribeResponse:
        payload = request.camera_depth_frame_subscribe
        if payload is None:
            raise RuntimeError("camera_depth_frame_subscribe payload missing")
        self._pipeline_context.get_camera_runtime(payload.camera_name)
        self._frame_publisher.start()
        logger.info(
            "api camera_depth_frame_subscribe ready camera_name={} stream_addr={}",
            payload.camera_name,
            self._frame_publisher.depth_bind_addr,
        )
        return CameraDepthFrameSubscribeResponse(
            stream_addr=self._frame_publisher.depth_bind_addr,
            camera_name=payload.camera_name,
        )

    # endregion

    # region 算法业务编排

    def _handle_ball_pose_detection(
        self, request: CameraPipelineServiceRequest
    ) -> BallPoseDetectionResponse:
        payload = request.ball_pose_detection
        if payload is None:
            raise RuntimeError("ball_pose_detection payload missing")
        logger.info(
            "api ball_pose_detection requested request_id={} camera_name={} requested_frame_id={} prior_count={} debug_enabled={}",
            payload.request_id,
            payload.camera_name,
            payload.frame_id,
            len(payload.priors),
            payload.enable_debug,
        )
        frame = self._pipeline_context.resolve_frame(payload.frame_id)
        response = self._get_ball_service().compute(frame, payload)
        logger.info(
            "api ball_pose_detection completed request_id={} camera_name={} frame_id={} matched_count={} detection_count={} elapsed_ms={:.3f}",
            response.request_id,
            response.camera_name,
            response.frame_id,
            response.matched_count,
            len(response.detections),
            response.elapsed_ms,
        )
        return response

    def _get_ball_service(self) -> BallPoseDetectionService:
        if self._ball_service is None:
            from ..ball_pose_detection.service import BallPoseDetectionService

            self._ball_service = BallPoseDetectionService()
        return self._ball_service

    # endregion
