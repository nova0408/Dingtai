"""标定算法入口。"""

from .charuco import (
    CharucoPoseEstimator,
    CharucoPoseResult,
    CHARUCO_200_12_9,
)
from .hand_eye import (
    HandEyeCalibrationResult,
    HandEyeResidualStats,
    calibrate_hand_eye_ax_xb,
    calibrate_hand_eye_from_pose_sequences,
    evaluate_hand_eye_solution,
    generate_synthetic_motion_pairs,
    make_relative_motion_pairs,
)

__all__ = [
    "CharucoPoseEstimator",
    "CharucoPoseResult",
    "CHARUCO_200_12_9",
    "HandEyeResidualStats",
    "HandEyeCalibrationResult",
    "make_relative_motion_pairs",
    "calibrate_hand_eye_ax_xb",
    "calibrate_hand_eye_from_pose_sequences",
    "evaluate_hand_eye_solution",
    "generate_synthetic_motion_pairs",
]
