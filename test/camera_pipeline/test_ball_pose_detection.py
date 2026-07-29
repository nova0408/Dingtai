from __future__ import annotations

from pathlib import Path
import sys

import cv2
from loguru import logger
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.ball_pose_detection.detector import (
    BallPoseDetector,
    _BallDetection,
    _ColorCandidate,
    _reference_hsv_ranges,
)
from camera_pipeline.ball_pose_detection.priors import BallPosePrior
from camera_pipeline.ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPosePriorInfo,
)
from camera_pipeline.ball_pose_detection.service import BallPoseDetectionService
from camera_pipeline.ball_pose_detection.types import BallPoseDetectionConfig
from camera_pipeline.client import CameraName
from camera_pipeline.protocol import CameraFramePacket, RgbdFrameProtocol


class _FixedCenterDetector(BallPoseDetector):
    """为候选直径硬校验提供固定三维球心的无硬件测试检测器。"""

    def _estimate_center_mm(
        self,
        frame: RgbdFrameProtocol,
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, int]:
        """返回固定球心，隔离颜色 mask 与球面拟合对本测试的影响。"""

        del frame, mask
        return np.asarray([0.0, 0.0, 1000.0], dtype=np.float64), 100


def test_relative_geometry_selects_true_ball_before_higher_color_score() -> None:
    """几何一致候选必须优先于颜色轮廓分数更高的错误色块。"""

    detector = BallPoseDetector()
    priors = _build_metric_priors()
    ranked = {
        "#ffff00": [_build_detection("#ffff00", (0.0, 0.0, 1000.0), 0.70)],
        "#ff0000": [
            _build_detection("#ff0000", (20.0, 0.0, 1000.0), 0.99),
            _build_detection("#ff0000", (100.0, 0.0, 1000.0), 0.55),
        ],
        "#ff00ff": [_build_detection("#ff00ff", (0.0, 80.0, 1000.0), 0.70)],
    }

    selected = detector._select_prior_consistent_detections(ranked, priors)

    assert selected[1].center_mm is not None
    assert np.allclose(selected[1].center_mm, (100.0, 0.0, 1000.0))
    assert all(item.detected for item in selected)


def test_relative_geometry_mismatch_returns_all_balls_missing() -> None:
    """没有几何一致完整组合时不得输出任何错误三球坐标。"""

    detector = BallPoseDetector()
    priors = _build_metric_priors()
    ranked = {
        "#ffff00": [_build_detection("#ffff00", (0.0, 0.0, 1000.0), 0.95)],
        "#ff0000": [_build_detection("#ff0000", (220.0, 0.0, 1000.0), 0.95)],
        "#ff00ff": [_build_detection("#ff00ff", (0.0, 200.0, 1000.0), 0.95)],
    }

    selected = detector._select_prior_consistent_detections(ranked, priors)

    assert all(not item.detected for item in selected)
    assert all(item.center_mm is None for item in selected)
    assert all(item.status == "relative_geometry_mismatch" for item in selected)


def test_diameter_mismatch_is_rejected_instead_of_soft_scored() -> None:
    """估计直径明显偏离先验时，单个颜色候选必须直接判定未检出。"""

    detector = _FixedCenterDetector(
        BallPoseDetectionConfig(max_diameter_error_ratio=0.20)
    )
    prior = BallPosePrior(
        color_hex="#ff0000",
        diameter_mm=20.0,
        model_center_mm=np.zeros((3,), dtype=np.float64),
    )
    candidate = _build_color_candidate(radius_px=60.0)

    ranked = detector._rank_ball_candidates(_build_frame(), prior, [candidate])

    assert len(ranked) == 1
    assert not ranked[0].detected
    assert ranked[0].center_mm is None
    assert ranked[0].status == "diameter_mismatch"


def test_prior_capture_prefers_diameter_before_appearance_score() -> None:
    """首次先验采集应优先选择直径准确而非外观分数更高的色块。"""

    detector = _FixedCenterDetector()
    prior = BallPosePrior(
        color_hex="#ff0000",
        diameter_mm=20.0,
        model_center_mm=np.zeros((3,), dtype=np.float64),
    )
    exact_diameter = _build_color_candidate(
        radius_px=10.0,
        circularity=0.46,
        fill_ratio=0.34,
    )
    better_appearance = _build_color_candidate(
        radius_px=12.0,
        circularity=1.0,
        fill_ratio=1.0,
    )

    ranked = detector._rank_ball_candidates(
        _build_frame(),
        prior,
        [better_appearance, exact_diameter],
    )

    assert len(ranked) == 2
    assert ranked[0].diameter_error_ratio == 0.0
    assert ranked[0].radius_px == 10.0


def test_calibrated_hsv_range_excludes_reference_color_false_block() -> None:
    """每球标定窄范围应排除仍落在参考宽范围内的相近错误色块。"""

    detector = BallPoseDetector()
    hsv = np.zeros((40, 80, 3), dtype=np.uint8)
    hsv[:, :40] = (2, 220, 220)
    hsv[:, 40:] = (9, 220, 220)
    color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    converted = detector._convert_to_hsv(color_bgr)

    mask = detector._build_color_mask(
        converted,
        ((0, 180, 180, 4, 255, 255),),
    )

    assert int(np.count_nonzero(mask[:, :35])) > 0
    assert int(np.count_nonzero(mask[:, 45:])) == 0


@pytest.mark.parametrize(
    ("color_hex", "true_hsv", "background_hsv"),
    [
        ("#ffff00", (30, 220, 250), (20, 105, 180)),
        ("#ff00ff", (150, 220, 250), (150, 70, 45)),
    ],
)
def test_reference_range_rejects_scene_block_but_keeps_ball(
    color_hex: str,
    true_hsv: tuple[int, int, int],
    background_hsv: tuple[int, int, int],
) -> None:
    """首次检测参考范围应保留高显色球体并排除现场低显色背景。"""

    detector = BallPoseDetector()
    hsv = np.zeros((80, 160, 3), dtype=np.uint8)
    hsv[:, :80] = background_hsv
    cv2.circle(hsv, (120, 40), 20, true_hsv, thickness=cv2.FILLED)
    color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    converted = detector._convert_to_hsv(color_bgr)

    mask = detector._build_color_mask(
        converted,
        _reference_hsv_ranges(color_hex, detector._config),
    )

    assert int(np.count_nonzero(mask[:, :70])) == 0
    assert int(np.count_nonzero(mask[25:55, 105:135])) > 0


def test_observed_hue_uses_circular_mean_across_red_wrap() -> None:
    """红色像素跨越 Hue 首尾时，实测中心应保持在红色附近。"""

    detector = BallPoseDetector()
    hsv = np.asarray(
        [[[179, 220, 220], [1, 220, 220]]] * 20,
        dtype=np.uint8,
    )
    mask = np.full(hsv.shape[:2], 255, dtype=np.uint8)

    observed = detector._estimate_observed_hsv(hsv, mask)

    assert observed is not None
    assert observed[0] < 2.0 or observed[0] > 178.0


def test_prior_capture_without_relative_positions_requires_debug() -> None:
    """带占位位置的首次先验采集关闭 debug 时必须被服务拒绝。"""

    service = BallPoseDetectionService()

    with pytest.raises(ValueError, match="requires enable_debug=True"):
        service.compute(
            _build_frame(),
            BallPoseDetectionRequest(
                request_id=1,
                camera_name=CameraName.LEFT_ARM,
                frame_id=1,
                enable_debug=False,
                priors=_build_placeholder_protocol_priors(),
            ),
        )


def test_prior_capture_with_debug_always_returns_overlay() -> None:
    """首次先验采集启用 debug 后必须返回可落盘的 overlay。"""

    response = BallPoseDetectionService().compute(
        _build_frame(),
        BallPoseDetectionRequest(
            request_id=2,
            camera_name=CameraName.LEFT_ARM,
            frame_id=1,
            enable_debug=True,
            priors=_build_placeholder_protocol_priors(),
        ),
    )

    assert len(response.debug_artifacts) == 1
    assert response.debug_artifacts[0].overlay_bgr.shape == (480, 640, 3)


def test_no_prior_smoke_path_returns_empty_result_without_debug() -> None:
    """完全不带先验时应返回空结果，且不触发先验采集 debug 约束。"""

    response = BallPoseDetectionService().compute(
        _build_frame(),
        BallPoseDetectionRequest(
            request_id=3,
            camera_name=CameraName.LEFT_ARM,
            frame_id=1,
            enable_debug=False,
            priors=(),
        ),
    )

    assert response.matched_count == 0
    assert response.detections == ()
    assert response.debug_artifacts == ()


def _build_metric_priors() -> list[BallPosePrior]:
    """构造具有实际毫米尺度的黄、红、紫三球模型先验。"""

    return [
        BallPosePrior("#ffff00", 20.0, np.asarray([0.0, 0.0, 0.0])),
        BallPosePrior("#ff0000", 20.0, np.asarray([100.0, 0.0, 0.0])),
        BallPosePrior("#ff00ff", 20.0, np.asarray([0.0, 80.0, 0.0])),
    ]


def _build_placeholder_protocol_priors() -> tuple[BallPosePriorInfo, ...]:
    """构造仅含颜色和物理直径、不含有效相对位置关系的采集先验。"""

    return (
        BallPosePriorInfo("#ffff00", 20.0, (0.0, 0.0, 0.0)),
        BallPosePriorInfo("#ff0000", 20.0, (1.0, 0.0, 0.0)),
        BallPosePriorInfo("#ff00ff", 20.0, (0.0, 1.0, 0.0)),
    )


def _build_detection(
    color_hex: str,
    center_mm: tuple[float, float, float],
    score: float,
) -> _BallDetection:
    """构造已通过深度和直径校验的内部候选。"""

    return _BallDetection(
        color_hex=color_hex,
        detected=True,
        status="detected",
        center_mm=np.asarray(center_mm, dtype=np.float64),
        center_px=(0.0, 0.0),
        radius_px=20.0,
        physical_diameter_mm=20.0,
        diameter_error_ratio=0.0,
        depth_points=100,
        score=score,
        contour=np.zeros((4, 2), dtype=np.int32),
        mask=np.ones((2, 2), dtype=np.uint8),
        center_norm=np.zeros((2,), dtype=np.float64),
        radius_norm=20.0,
        observed_hsv=np.asarray([0.0, 220.0, 220.0]),
    )


def _build_color_candidate(
    radius_px: float,
    circularity: float = 1.0,
    fill_ratio: float = 1.0,
) -> _ColorCandidate:
    """构造指定像素半径的颜色候选。"""

    return _ColorCandidate(
        color_hex="#ff0000",
        contour=np.zeros((4, 2), dtype=np.int32),
        mask=np.ones((2, 2), dtype=np.uint8),
        color_sample_mask=np.ones((2, 2), dtype=np.uint8),
        center_px=(320.0, 240.0),
        radius_px=radius_px,
        center_norm=np.asarray([320.0, 240.0], dtype=np.float64),
        radius_norm=radius_px,
        area_px=100,
        circularity=circularity,
        fill_ratio=fill_ratio,
        observed_hsv=np.asarray([0.0, 220.0, 220.0]),
    )


def _build_frame() -> CameraFramePacket:
    """构造焦距 1000 pixel、深度 1000 mm 的最小合成帧。"""

    return CameraFramePacket(
        frame_id=1,
        camera_name="test_camera",
        timestamp_ms=0.0,
        color_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_mm=np.full((480, 640), 1000, dtype=np.uint16),
        fx=1000.0,
        fy=1000.0,
        cx=320.0,
        cy=240.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def main() -> None:
    """在 IDE 中直接执行无硬件三球候选选择验证。"""

    test_relative_geometry_selects_true_ball_before_higher_color_score()
    test_relative_geometry_mismatch_returns_all_balls_missing()
    test_diameter_mismatch_is_rejected_instead_of_soft_scored()
    test_prior_capture_prefers_diameter_before_appearance_score()
    test_calibrated_hsv_range_excludes_reference_color_false_block()
    test_reference_range_rejects_scene_block_but_keeps_ball(
        "#ffff00",
        (30, 220, 250),
        (20, 105, 180),
    )
    test_reference_range_rejects_scene_block_but_keeps_ball(
        "#ff00ff",
        (140, 135, 215),
        (140, 70, 45),
    )
    test_observed_hue_uses_circular_mean_across_red_wrap()
    test_prior_capture_without_relative_positions_requires_debug()
    test_prior_capture_with_debug_always_returns_overlay()
    test_no_prior_smoke_path_returns_empty_result_without_debug()
    logger.success("三球几何、直径、精确 HSV 与先验采集模式验证通过")
    logger.warning("本结果未连接真实相机，未验证现场颜色、深度噪声和实时耗时")


if __name__ == "__main__":
    main()
