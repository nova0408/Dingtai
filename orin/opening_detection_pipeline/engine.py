from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..camera_stream import CameraStreamRuntime
from ..opening_detection.engine import GraspPoseExecutor, GraspPoseExecutorConfig
from ..opening_detection.protocol import DebugArtifacts, GraspPoseRequest, GraspPoseResponse
from ..tray_detection.engine import OrinTrayDetectionExecutor, OrinTrayDetectionExecutorConfig
from ..tray_detection.protocol import OrinTrayDetectionRequest

from .protocol import OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse


@dataclass(frozen=True)
class OpeningDetectionPipelineExecutorConfig:
    """抓取位姿主服务执行器配置。"""

    tray_config: OrinTrayDetectionExecutorConfig = OrinTrayDetectionExecutorConfig()
    grasp_config: GraspPoseExecutorConfig = GraspPoseExecutorConfig()


class OpeningDetectionPipelineExecutor:
    """统一顺序执行托盘检测与抓取位姿的主执行器。"""

    def __init__(self, frame_runtime: CameraStreamRuntime, config: Optional[OpeningDetectionPipelineExecutorConfig] = None) -> None:
        self._frame_runtime = frame_runtime
        self._config = OpeningDetectionPipelineExecutorConfig() if config is None else config
        self._tray_executor = OrinTrayDetectionExecutor(frame_runtime=frame_runtime, config=self._config.tray_config)
        self._grasp_executor = GraspPoseExecutor(frame_runtime=frame_runtime, config=self._config.grasp_config)

    def process_request(self, request: OpeningDetectionPipelineRequest) -> OpeningDetectionPipelineResponse:
        t0 = time.perf_counter()
        tray_response = self._tray_executor.process_request(
            OrinTrayDetectionRequest(
                request_id=int(request.request_id),
                camera_name=str(request.camera_name),
                frame_id=int(request.frame_id),
                enable_debug=True,
            )
        )
        if tray_response.error is not None:
            return self._build_error_response(request, tray_response, float((time.perf_counter() - t0) * 1000.0))
        if tray_response.debug is None or len(tray_response.debug.tray_masks) != len(tray_response.tray_results):
            return self._build_error_response(request, tray_response, float((time.perf_counter() - t0) * 1000.0), "tray debug masks unavailable")

        tray_pose_responses: list[GraspPoseResponse] = []
        for tray_info, tray_mask in zip(tray_response.tray_results, tray_response.debug.tray_masks):
            tray_pose_responses.append(
                self._grasp_executor.process_request(
                    GraspPoseRequest(
                        request_id=int(request.request_id),
                        frame_id=int(tray_response.frame_id),
                        camera_name=str(tray_response.camera_name),
                        timestamp_ms=float(tray_response.timestamp_ms),
                        source_meta=dict(tray_response.source_meta),
                        target_tray_index=int(tray_info.tray_id),
                        enable_debug=bool(request.enable_debug),
                        tray_mask=np.asarray(tray_mask, dtype=np.uint8),
                    )
                )
            )

        selected_index = int(request.target_tray_index)
        selected_pose_response = None
        for item in tray_pose_responses:
            if item.selected_result is not None and int(item.selected_result.tray_index) == selected_index:
                selected_pose_response = item
                break
        selected_result = None if selected_pose_response is None else selected_pose_response.selected_result
        merged_debug = self._merge_debug_artifacts(bool(request.enable_debug), tray_response, selected_pose_response)
        error = tray_response.error
        if error is None:
            for item in tray_pose_responses:
                if item.error is not None:
                    error = item.error
                    break
        return OpeningDetectionPipelineResponse(
            request_id=int(request.request_id),
            frame_id=int(tray_response.frame_id),
            camera_name=str(tray_response.camera_name),
            timestamp_ms=float(tray_response.timestamp_ms),
            source_meta=dict(tray_response.source_meta),
            elapsed_ms=float((time.perf_counter() - t0) * 1000.0),
            tray_count=int(tray_response.tray_count),
            tray_results=tuple(tray_response.tray_results),
            selected_tray_index=selected_index,
            selected_result=selected_result,
            all_tray_results=tuple(item.selected_result for item in tray_pose_responses if item.selected_result is not None),
            debug=merged_debug,
            error=error,
        )

    def _merge_debug_artifacts(self, enable_debug: bool, tray_response, selected_pose_response: Optional[GraspPoseResponse]) -> Optional[DebugArtifacts]:
        if not enable_debug or selected_pose_response is None or selected_pose_response.debug is None:
            return None
        pose_debug = selected_pose_response.debug
        return DebugArtifacts(
            color_bgr=None if pose_debug.color_bgr is None else np.asarray(pose_debug.color_bgr, dtype=np.uint8).copy(),
            depth_mm=None if pose_debug.depth_mm is None else np.asarray(pose_debug.depth_mm, dtype=np.uint16).copy(),
            camera_intrinsics=pose_debug.camera_intrinsics,
            overlay_bgr=None if pose_debug.overlay_bgr is None else np.asarray(pose_debug.overlay_bgr, dtype=np.uint8).copy(),
            contrast_bgr=None if pose_debug.contrast_bgr is None else np.asarray(pose_debug.contrast_bgr, dtype=np.uint8).copy(),
            tray_instance_masks=tuple(np.asarray(item, dtype=np.uint8).copy() for item in tray_response.debug.tray_masks),
            selected_tray_mask=None if pose_debug.selected_tray_mask is None else np.asarray(pose_debug.selected_tray_mask, dtype=np.uint8).copy(),
            near_plane_mask=None if pose_debug.near_plane_mask is None else np.asarray(pose_debug.near_plane_mask, dtype=np.uint8).copy(),
            no_hole_mask=None if pose_debug.no_hole_mask is None else np.asarray(pose_debug.no_hole_mask, dtype=np.uint8).copy(),
        )

    def _build_error_response(self, request: OpeningDetectionPipelineRequest, tray_response, elapsed_ms: float, error: Optional[str] = None) -> OpeningDetectionPipelineResponse:
        return OpeningDetectionPipelineResponse(
            request_id=int(request.request_id),
            frame_id=int(tray_response.frame_id),
            camera_name=str(tray_response.camera_name),
            timestamp_ms=float(tray_response.timestamp_ms),
            source_meta=dict(tray_response.source_meta),
            elapsed_ms=float(elapsed_ms),
            tray_count=int(tray_response.tray_count),
            tray_results=tuple(tray_response.tray_results),
            selected_tray_index=int(request.target_tray_index),
            selected_result=None,
            all_tray_results=tuple(),
            debug=None,
            error=tray_response.error if error is None else str(error),
        )
