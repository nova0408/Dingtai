"""ChArUco 纠偏检测、历史安全门与本轮 runtime 缓存。"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation
from qmlinker import QMHead

from .board_detector_gateway import BoardDetectionConfig, CameraPipelineBoardDetector
from .runtime import ReplayRuntime
from .settings import OffsetConfig, ReplayServiceSettings

CHARUCO_HISTORY_FIELDS: tuple[str, ...] = (
    "source_file",
    "captured_at",
    "arm_side",
    "x_mm",
    "y_mm",
    "z_mm",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "translation_norm_mm",
    "rotation_norm_deg",
    "accepted",
    "decision_reason",
)


class CharucoOffsetInitializer:
    """按已验证 CLI 的流程初始化活动机械臂的 ChArUco offset。"""

    def __init__(self, config: OffsetConfig, settings: ReplayServiceSettings) -> None:
        self._config = config
        self._settings = settings

    def initialize(self, runtimes: list[ReplayRuntime]) -> None:
        """设置头部姿态、检测目标板并在安全门通过后缓存 offset。"""

        if not runtimes:
            raise ValueError("初始化 ChArUco offset 时缺少活动 runtime")
        if any(runtime.charuco_cartesian_offset is not None for runtime in runtimes):
            raise RuntimeError("本轮 ChArUco offset 已初始化，拒绝重复检测目标板")
        settings = self._settings.offset
        detector = CameraPipelineBoardDetector(
            BoardDetectionConfig(
                max_frames=settings.charuco_max_frame_count,
                min_charuco_corners=settings.charuco_min_corners,
                stable_timeout_s=settings.charuco_camera_timeout_s,
                rpc_timeout_s=settings.charuco_rpc_timeout_s,
                timeout_retry_count=settings.charuco_timeout_retry_count,
                timeout_retry_delay_s=settings.charuco_timeout_retry_delay_s,
            )
        )
        rejected: list[str] = []
        for attempt in range(1, settings.charuco_safety_attempt_count + 1):
            current_camera_board_m = self._detect_current_board(detector, runtimes[0])
            decisions = [
                self._precheck(runtime, current_camera_board_m)
                for runtime in runtimes
            ]
            failures = [reason for accepted, reason in decisions if not accepted]
            if failures:
                detail = f"attempt={attempt}/{settings.charuco_safety_attempt_count} failures={' | '.join(failures)}"
                rejected.append(detail)
                if attempt < settings.charuco_safety_attempt_count:
                    logger.warning(
                        "ChArUco offset 安全检查未通过，重新检测 {} delay={:.1f}s",
                        detail,
                        settings.charuco_safety_retry_delay_s,
                    )
                    time.sleep(settings.charuco_safety_retry_delay_s)
                    continue
                raise RuntimeError(
                    "ChArUco offset 连续安全检查均被拒绝：" + "; ".join(rejected)
                )
            for runtime in runtimes:
                runtime.charuco_cartesian_offset = self._calculate_offset(
                    runtime,
                    current_camera_board_m,
                )
            logger.success(
                "本轮 ChArUco offset 初始化完成 attempt={}/{}，后续 CSV 使用缓存结果",
                attempt,
                settings.charuco_safety_attempt_count,
            )
            return
        raise RuntimeError("ChArUco offset 安全检查流程意外结束")

    def _detect_current_board(
        self,
        detector: CameraPipelineBoardDetector,
        runtime: ReplayRuntime,
    ) -> np.ndarray:
        head = QMHead(runtime.hand_body.body_channel)
        head_settings = self._settings.offset
        head.set_head_yaw(head_settings.charuco_head_yaw_deg)
        head.set_head_pitch(head_settings.charuco_head_pitch_deg)
        time.sleep(head_settings.charuco_head_settle_s)
        logger.info(
            "头部 ChArUco 检测姿态已设置 yaw={:.1f} deg pitch={:.1f} deg",
            head.get_head_yaw(),
            head.get_head_pitch(),
        )
        board_mm = np.asarray(detector.detect_t_camera_board_mm(), dtype=np.float64)
        if board_mm.shape != (4, 4) or not np.all(np.isfinite(board_mm)):
            raise ValueError(f"CameraPipeline 返回的 T_camera_board 无效：shape={board_mm.shape}")
        board_m = board_mm.copy()
        board_m[:3, 3] *= 0.001
        return board_m

    def _calculate_offset(
        self,
        runtime: ReplayRuntime,
        current_camera_board_m: np.ndarray,
    ) -> tuple[tuple[float, float, float, float], ...]:
        base_camera_path = (
            self._config.left_head_base_camera_path
            if runtime.connected_arm.arm_side == "left"
            else self._config.right_head_base_camera_path
        )
        if base_camera_path is None:
            raise RuntimeError(f"{runtime.connected_arm.arm_side} 臂未配置 T_base_camera 路径")
        prior_board_path = self._config.charuco_prior_path
        if prior_board_path is None:
            raise RuntimeError("未配置 ChArUco 先验路径")
        base_camera_m = _load_matrix(base_camera_path, "T_base_camera")
        prior_camera_board_m = _load_prior_board(prior_board_path)
        prior_base_board_m = base_camera_m @ prior_camera_board_m
        current_base_board_m = base_camera_m @ current_camera_board_m
        offset_matrix_m = current_base_board_m @ np.linalg.inv(prior_base_board_m)
        accepted, reason = self._evaluate(runtime.connected_arm.arm_side, offset_matrix_m)
        if not accepted:
            raise RuntimeError(f"ChArUco offset 安全检查拒绝执行：{reason}")
        logger.info("ChArUco offset 安全检查通过 {}", reason)
        return _matrix_to_tuple(offset_matrix_m)

    def _precheck(
        self,
        runtime: ReplayRuntime,
        current_camera_board_m: np.ndarray,
    ) -> tuple[bool, str]:
        base_camera_path = (
            self._config.left_head_base_camera_path
            if runtime.connected_arm.arm_side == "left"
            else self._config.right_head_base_camera_path
        )
        if base_camera_path is None or self._config.charuco_prior_path is None:
            raise RuntimeError(f"{runtime.connected_arm.arm_side} 臂 ChArUco 配置不完整")
        base_camera_m = _load_matrix(base_camera_path, "T_base_camera")
        prior_camera_board_m = _load_prior_board(self._config.charuco_prior_path)
        offset_matrix_m = (
            base_camera_m @ current_camera_board_m
        ) @ np.linalg.inv(base_camera_m @ prior_camera_board_m)
        return self._evaluate(runtime.connected_arm.arm_side, offset_matrix_m)

    def _evaluate(self, arm_side: str, offset_matrix_m: np.ndarray) -> tuple[bool, str]:
        history_path = self._config.charuco_history_path
        if history_path is None:
            raise RuntimeError("未配置 ChArUco offset 历史路径")
        settings = self._settings.offset
        history = _load_history(history_path, arm_side)
        if history.shape[0] < settings.charuco_history_min_accepted_samples:
            return False, (
                f"{arm_side} 臂有效历史样本不足：{history.shape[0]} < "
                f"{settings.charuco_history_min_accepted_samples}"
            )
        values = _xyzrpy_mm_deg(offset_matrix_m)
        means = np.mean(history, axis=0)
        deviations = np.std(history, axis=0, ddof=1)
        sigma = settings.charuco_sigma_limit
        lower = means - sigma * deviations
        upper = means + sigma * deviations
        labels = ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")
        violations = [
            f"{label}={value:.3f} 不在 [{minimum:.3f}, {maximum:.3f}]"
            for label, value, minimum, maximum in zip(labels, values, lower, upper, strict=True)
            if value < minimum or value > maximum
        ]
        translation_norms = np.linalg.norm(history[:, :3], axis=1)
        rotation_norms = np.linalg.norm(history[:, 3:], axis=1)
        translation_limit = min(float(np.mean(translation_norms) + sigma * np.std(translation_norms, ddof=1)), settings.charuco_max_translation_norm_mm)
        rotation_limit = min(float(np.mean(rotation_norms) + sigma * np.std(rotation_norms, ddof=1)), settings.charuco_max_rotation_norm_deg)
        translation_norm = float(np.linalg.norm(values[:3]))
        rotation_norm = float(np.linalg.norm(values[3:]))
        if translation_norm > translation_limit:
            violations.append(f"translation_norm_mm={translation_norm:.3f} > {translation_limit:.3f}")
        if rotation_norm > rotation_limit:
            violations.append(f"rotation_norm_deg={rotation_norm:.3f} > {rotation_limit:.3f}")
        summary = (
            f"arm_side={arm_side} history_count={history.shape[0]} sigma={sigma:.1f} "
            f"translation_limit_mm={translation_limit:.3f} rotation_limit_deg={rotation_limit:.3f}"
        )
        return (not violations, f"{summary}; " + ("; ".join(violations) if violations else "within_normal_range"))


def _load_matrix(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} 格式无效：shape={matrix.shape}, path={path}")
    return matrix


def _load_prior_board(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ChArUco 先验根节点必须为 object：{path}")
    matrix = np.asarray(payload.get("camera_board_transform"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"ChArUco 先验缺少有效 camera_board_transform：{path}")
    if payload.get("translation_unit") != "mm":
        raise ValueError(f"ChArUco 先验平移单位不是 mm：{payload.get('translation_unit')!r}")
    result = matrix.copy()
    result[:3, 3] *= 0.001
    return result


def _load_history(path: Path, arm_side: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"ChArUco offset 历史文件不存在：{path}")
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CHARUCO_HISTORY_FIELDS:
            raise RuntimeError(f"ChArUco offset 历史 CSV 字段不符合约定：{path}")
        for row in reader:
            if row["arm_side"] != arm_side or row["accepted"].strip().lower() != "true":
                continue
            values.append([float(row[key]) for key in CHARUCO_HISTORY_FIELDS[3:9]])
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    if result.shape[1] != 6 or not np.all(np.isfinite(result)):
        raise RuntimeError(f"ChArUco offset 历史 CSV 包含非有限数值：{path}")
    return result


def _xyzrpy_mm_deg(matrix: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return np.asarray(
        [matrix[0, 3] * 1000.0, matrix[1, 3] * 1000.0, matrix[2, 3] * 1000.0, *rotation],
        dtype=np.float64,
    )


def _matrix_to_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())  # type: ignore[return-value]
