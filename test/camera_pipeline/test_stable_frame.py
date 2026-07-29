from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.stable_frame import StableFrameConfig, StableFrameDetector

DEFAULT_FRAME_INTERVAL_MS = 100.0
"合成帧间隔，单位 ms。"

DEFAULT_STABLE_DURATION_S = 1.0
"连续稳定判定时间，单位 s。"


@dataclass(frozen=True, slots=True)
class _TestFrame:
    """稳定帧算法使用的最小合成帧。"""

    frame_id: int
    timestamp_ms: float
    color_bgr: npt.NDArray[np.uint8]
    depth_mm: npt.NDArray[np.uint16]


def test_returns_midpoint_frame_id_after_complete_stable_window() -> None:
    detector = _build_detector()
    result = None
    for frame_id in range(11):
        result = detector.update(
            _build_frame(frame_id, frame_id * DEFAULT_FRAME_INTERVAL_MS)
        )

    assert result == 5


def test_motion_resets_stable_window() -> None:
    detector = _build_detector()
    results: list[int | None] = []
    for frame_id in range(16):
        color_offset = 0
        if frame_id == 5:
            color_offset = 80
        results.append(
            detector.update(
                _build_frame(
                    frame_id,
                    frame_id * DEFAULT_FRAME_INTERVAL_MS,
                    color_offset=color_offset,
                )
            )
        )

    assert all(result is None for result in results)


def test_depth_change_resets_stable_window() -> None:
    detector = _build_detector()
    for frame_id in range(5):
        assert (
            detector.update(
                _build_frame(frame_id, frame_id * DEFAULT_FRAME_INTERVAL_MS)
            )
            is None
        )

    changed_depth = np.full((48, 64), 1100, dtype=np.uint16)
    changed_frame = _build_frame(5, 500.0, depth_mm=changed_depth)
    assert detector.update(changed_frame) is None


def test_color_only_stability_ignores_depth_changes() -> None:
    detector = _build_detector()
    result = None
    for frame_id in range(11):
        depth = np.full((48, 64), 1000 + frame_id * 100, dtype=np.uint16)
        result = detector.update_color(
            _build_frame(
                frame_id,
                frame_id * DEFAULT_FRAME_INTERVAL_MS,
                depth_mm=depth,
            )
        )

    assert result == 5


def main() -> None:
    """在 IDE 中直接执行稳定帧最小验证。"""

    test_returns_midpoint_frame_id_after_complete_stable_window()
    test_motion_resets_stable_window()
    test_depth_change_resets_stable_window()
    test_color_only_stability_ignores_depth_changes()
    logger.success("stable_frame 合成帧验证通过")
    logger.warning("本结果未连接真实相机，稳定性阈值仍需使用现场数据标定")


def _build_detector() -> StableFrameDetector:
    return StableFrameDetector(
        StableFrameConfig(
            stable_duration_s=DEFAULT_STABLE_DURATION_S,
            image_scale=1.0,
            max_frame_gap_ms=150.0,
        )
    )


def _build_frame(
    frame_id: int,
    timestamp_ms: float,
    *,
    color_offset: int = 0,
    depth_mm: npt.NDArray[np.uint16] | None = None,
) -> _TestFrame:
    color = np.zeros((48, 64, 3), dtype=np.uint8)
    color[12:36, 16:48] = np.uint8(color_offset)
    depth = np.full((48, 64), 1000, dtype=np.uint16) if depth_mm is None else depth_mm
    return _TestFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        color_bgr=color,
        depth_mm=depth,
    )


if __name__ == "__main__":
    main()
