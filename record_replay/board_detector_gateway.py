"""通过 Orin 本机 CameraPipeline 服务获取 Board 检测结果。"""

from __future__ import annotations

from dataclasses import dataclass

from camera_pipeline.client import CameraName, CameraPipelineClient
from camera_pipeline.service.protocol import CharucoDetectionRequest


@dataclass(frozen=True, slots=True)
class BoardDetectionConfig:
    """RecordReplay 使用的固定 Board 几何与检测参数。"""

    camera_name: CameraName = CameraName.HEAD
    dictionary_name: str = "DICT_APRILTAG_16H5"
    squares_x: int = 4
    squares_y: int = 4
    square_length_mm: float = 20.0
    marker_length_mm: float = 14.0
    min_charuco_corners: int = 6
    max_frames: int = 300
    stable_timeout_s: float = 10.0


class CameraPipelineBoardDetector:
    """只调用 CameraPipeline Board API，不获取帧也不实现视觉算法。"""

    def __init__(self, config: BoardDetectionConfig | None = None) -> None:
        self._config = BoardDetectionConfig() if config is None else config

    def detect_t_camera_board_mm(self) -> tuple[tuple[float, float, float, float], ...]:
        """返回有效的 T_camera_board；未检测到 Board 时抛出明确异常。"""

        config = self._config
        client = CameraPipelineClient()
        try:
            response = client.detect_charuco(
                CharucoDetectionRequest(
                    camera_name=config.camera_name,
                    dictionary_name=config.dictionary_name,
                    squares_x=config.squares_x,
                    squares_y=config.squares_y,
                    square_length_mm=config.square_length_mm,
                    marker_length_mm=config.marker_length_mm,
                    min_charuco_corners=config.min_charuco_corners,
                    max_frames=config.max_frames,
                    stable_timeout_s=config.stable_timeout_s,
                )
            )
        finally:
            client.close()
        if response.status != "detected" or len(response.t_cam_board_mm) != 4:
            raise RuntimeError(
                "CameraPipeline 未检测到有效 Board "
                f"markers={response.marker_num} charuco={response.charuco_num}"
            )
        return response.t_cam_board_mm
