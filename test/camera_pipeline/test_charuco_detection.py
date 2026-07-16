from __future__ import annotations

from pathlib import Path
import sys

import cv2
from loguru import logger
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.charuco_detection import CharucoDetector
from camera_pipeline.pipeline_context import PipelineContext, PipelineContextConfig
from camera_pipeline.protocol import CameraFramePacket

DEFAULT_MAX_FRAMES = 5
"稳定帧识别重试上限，单位 帧。"


# region 测试上下文
class _MemoryCharucoContext(PipelineContext):
    """用内存帧替代真实相机稳定帧等待的测试上下文。

    本类只覆盖稳定帧获取入口，用于验证 `PipelineContext` 的重试控制流；不创建
    socket、线程或硬件句柄。帧由测试线程顺序写入和读取，不支持并发调用。

    继承 `PipelineContext` 是为了直接验证正式编排方法，不复制生产控制流。
    """

    def __init__(self, frames: tuple[CameraFramePacket, ...]) -> None:
        self._config = PipelineContextConfig()
        self._frames = frames
        self._index = 0
        self.wait_count = 0

    def wait_for_stable_frame(
        self,
        timeout_s: float = 10.0,
        camera_name: str | None = None,
    ) -> CameraFramePacket:
        """按顺序返回内存帧并记录稳定帧等待次数。

        Parameters
        ----------
        timeout_s:
            生产接口兼容参数，单位 s；内存测试不实际等待。
        camera_name:
            生产接口兼容参数；内存测试不区分相机。

        Returns
        -------
        CameraFramePacket
            当前索引对应的合成稳定帧。
        """

        del timeout_s, camera_name
        frame = self._frames[self._index]
        self._index = min(self._index + 1, len(self._frames) - 1)
        self.wait_count += 1
        return frame


# endregion


# region 测试用例
def test_detects_generated_board_without_debug_copy() -> None:
    board = _build_board()
    detector = CharucoDetector(board)

    result = detector.detect(_build_board_frame(board), enable_debug=False)

    assert result.status == "detected"
    assert result.t_cam_board_mm.shape == (4, 4)
    assert np.isfinite(result.error_px)
    assert result.marker_num > 0
    assert result.charuco_num >= 6
    assert result.debug_artifacts == ()


def test_debug_contains_marker_charuco_and_pose_overlay() -> None:
    board = _build_board()
    frame = _build_board_frame(board)

    result = CharucoDetector(board).detect(frame, enable_debug=True)

    assert result.status == "detected"
    assert len(result.debug_artifacts) == 1
    debug = result.debug_artifacts[0]
    assert debug.overlay_bgr.shape == frame.color_bgr.shape
    assert debug.marker_ids.shape[0] == result.marker_num
    assert debug.charuco_ids.shape[0] == result.charuco_num
    assert not np.array_equal(debug.overlay_bgr, frame.color_bgr)


def test_pipeline_context_retries_until_fifth_stable_frame() -> None:
    board = _build_board()
    blank = _build_blank_frame()
    detected = _build_board_frame(board)
    context = _MemoryCharucoContext((blank, blank, blank, blank, detected))

    result = context.detect_charuco(board, max_frames=DEFAULT_MAX_FRAMES)

    assert result.status == "detected"
    assert context.wait_count == DEFAULT_MAX_FRAMES


def test_pipeline_context_returns_last_missing_result_at_limit() -> None:
    board = _build_board()
    blank = _build_blank_frame()
    context = _MemoryCharucoContext((blank, blank, blank, blank, blank))

    result = context.detect_charuco(board, max_frames=DEFAULT_MAX_FRAMES)

    assert result.status == "missing"
    assert result.t_cam_board_mm.shape == (0, 0)
    assert context.wait_count == DEFAULT_MAX_FRAMES


# endregion


# region 数据构造
def _build_board() -> cv2.aruco.CharucoBoard:
    """构造使用 mm 尺寸的合成 ChArUco 板。"""

    return cv2.aruco.CharucoBoard(
        (12, 9),
        15.0,
        11.25,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
    )


def _build_board_frame(board: cv2.aruco.CharucoBoard) -> CameraFramePacket:
    """构造正视、零畸变的合成 ChArUco 相机帧。"""

    gray = board.generateImage((640, 480), marginSize=30, borderBits=1)
    color_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _build_frame(color_bgr)


def _build_blank_frame() -> CameraFramePacket:
    """构造不含 marker 的白色相机帧。"""

    return _build_frame(np.full((480, 640, 3), 255, dtype=np.uint8))


def _build_frame(color_bgr: np.ndarray) -> CameraFramePacket:
    """为给定彩色图补齐相机帧协议字段。"""

    return CameraFramePacket(
        frame_id=1,
        camera_name="test_camera",
        timestamp_ms=0.0,
        color_bgr=np.asarray(color_bgr, dtype=np.uint8),
        depth_mm=np.full(color_bgr.shape[:2], 1000, dtype=np.uint16),
        fx=800.0,
        fy=800.0,
        cx=320.0,
        cy=240.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


# endregion


def main() -> None:
    """在 IDE 中直接执行无硬件 ChArUco 合成图验证。"""

    test_detects_generated_board_without_debug_copy()
    test_debug_contains_marker_charuco_and_pose_overlay()
    test_pipeline_context_retries_until_fifth_stable_frame()
    test_pipeline_context_returns_last_missing_result_at_limit()
    logger.success("ChArUco 合成图、debug 与 5 帧重试验证通过")
    logger.warning("本结果未连接真实相机，未验证现场光照、畸变参数和实时耗时")


if __name__ == "__main__":
    main()
