from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
from loguru import logger

from ..ball_pose_detection.protocol import BallPoseDetectionResponse
from ..protocol import RgbdFrameProtocol
from ..pipeline_context import PipelineContext
from .frame_publisher import CameraFramePublisher
from .protocol import (
    PROTOCOL_VERSION,
    SERVICE_VERSION,
    CameraColorFrameSubscribeResponse,
    CameraDepthFrameSubscribeResponse,
    CameraFrameSubscribeResponse,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CameraStatusResponse,
    CameraSummaryResponse,
    CharucoDetectionResponse,
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
        if request.operation == "detect_ball":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                detect_ball=self._handle_ball_pose_detection(request),
            )
        if request.operation == "detect_charuco":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                detect_charuco=self._handle_charuco_detection(request),
            )
        raise RuntimeError(f"unsupported operation: {request.operation}")

    # endregion

    # region 相机查询与稳定帧

    def _handle_camera_summary(
        self, request: CameraPipelineServiceRequest
    ) -> CameraSummaryResponse:
        frame = self._wait_for_latest_frame(
            request.timeout_s,
            "camera first frame",
            camera_name=request.camera_name,
        )
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
        frame = self._wait_for_latest_frame(
            request.timeout_s,
            "camera intrinsics",
            camera_name=request.camera_name,
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
        frame = self._wait_for_latest_frame(
            request.timeout_s,
            "camera status",
            camera_name=request.camera_name,
        )
        logger.info(
            "api camera_status resolved camera_name={} frame_id={} online=True",
            frame.camera_name,
            frame.frame_id,
        )
        return CameraStatusResponse(
            service_version=SERVICE_VERSION,
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
        frame = self._pipeline_context.wait_for_stable_frame(
            timeout_s=request.timeout_s,
            camera_name=request.camera_name,
        )
        logger.info(
            "api stable_frame resolved camera_name={} frame_id={} timeout_s={:.3f}",
            frame.camera_name,
            frame.frame_id,
            request.timeout_s,
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
        self._pipeline_context.get_camera_runtime(request.camera_name)
        self._frame_publisher.start()
        logger.info(
            "api camera_frame_subscribe ready camera_name={} stream_addr={}",
            request.camera_name,
            self._frame_publisher.frame_bind_addr,
        )
        return CameraFrameSubscribeResponse(
            stream_addr=self._frame_publisher.frame_bind_addr,
            camera_name=request.camera_name,
        )

    def _handle_color_frame_subscribe(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CameraColorFrameSubscribeResponse:
        self._pipeline_context.get_camera_runtime(request.camera_name)
        self._frame_publisher.start()
        logger.info(
            "api camera_color_frame_subscribe ready camera_name={} stream_addr={}",
            request.camera_name,
            self._frame_publisher.color_bind_addr,
        )
        return CameraColorFrameSubscribeResponse(
            stream_addr=self._frame_publisher.color_bind_addr,
            camera_name=request.camera_name,
        )

    def _handle_depth_frame_subscribe(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CameraDepthFrameSubscribeResponse:
        self._pipeline_context.get_camera_runtime(request.camera_name)
        self._frame_publisher.start()
        logger.info(
            "api camera_depth_frame_subscribe ready camera_name={} stream_addr={}",
            request.camera_name,
            self._frame_publisher.depth_bind_addr,
        )
        return CameraDepthFrameSubscribeResponse(
            stream_addr=self._frame_publisher.depth_bind_addr,
            camera_name=request.camera_name,
        )

    # endregion

    # region 算法业务编排

    def _handle_ball_pose_detection(
        self, request: CameraPipelineServiceRequest
    ) -> BallPoseDetectionResponse:
        payload = request.detect_ball
        if payload is None:
            raise RuntimeError("detect_ball payload missing")
        if payload.camera_name != request.camera_name:
            raise ValueError("detect_ball camera_name mismatch")
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

    def _handle_charuco_detection(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CharucoDetectionResponse:
        """在 CameraPipeline 内构造 Board、获取稳定帧并完成检测。"""

        payload = request.detect_charuco
        if payload is None:
            raise RuntimeError("detect_charuco payload missing")
        if payload.camera_name != request.camera_name:
            raise ValueError("detect_charuco camera_name mismatch")
        if payload.dictionary_name != "DICT_APRILTAG_16H5":
            raise ValueError(f"unsupported dictionary: {payload.dictionary_name}")
        if payload.squares_x < 2 or payload.squares_y < 2:
            raise ValueError("charuco board squares_x and squares_y must be at least 2")
        if payload.square_length_mm <= 0.0:
            raise ValueError("charuco square_length_mm must be greater than zero")
        if not 0.0 < payload.marker_length_mm < payload.square_length_mm:
            raise ValueError("charuco marker_length_mm must be between zero and square_length_mm")
        if payload.min_charuco_corners < 4:
            raise ValueError("charuco min_charuco_corners must be at least 4")
        if payload.max_frames <= 0 or payload.stable_timeout_s <= 0.0:
            raise ValueError("charuco max_frames and stable_timeout_s must be greater than zero")
        dictionary = cv2.aruco.getPredefinedDictionary(int(cv2.aruco.DICT_APRILTAG_16h5))
        board = cv2.aruco.CharucoBoard(
            (payload.squares_x, payload.squares_y),
            payload.square_length_mm,
            payload.marker_length_mm,
            dictionary,
        )
        from ..charuco_detection import CharucoDetectionConfig

        result = self._pipeline_context.detect_charuco(
            board,
            camera_name=request.camera_name,
            config=CharucoDetectionConfig(min_charuco_corners=payload.min_charuco_corners),
            enable_debug=False,
            max_frames=payload.max_frames,
            stable_timeout_s=payload.stable_timeout_s,
        )
        matrix_array = result.t_cam_board_mm
        if matrix_array.size == 0:
            matrix: tuple[tuple[float, float, float, float], ...] = ()
        elif matrix_array.shape == (4, 4):
            matrix = tuple(
                (
                    float(matrix_array[row_index, 0]),
                    float(matrix_array[row_index, 1]),
                    float(matrix_array[row_index, 2]),
                    float(matrix_array[row_index, 3]),
                )
                for row_index in range(4)
            )
        else:
            raise RuntimeError(f"unexpected charuco matrix shape: {matrix_array.shape}")
        return CharucoDetectionResponse(
            status=result.status,
            camera_name=request.camera_name,
            t_cam_board_mm=matrix,
            error_px=result.error_px,
            marker_num=result.marker_num,
            charuco_num=result.charuco_num,
        )

    # endregion
