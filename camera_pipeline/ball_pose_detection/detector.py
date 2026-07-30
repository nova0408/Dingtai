from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass, field
import itertools
import cv2
import numpy as np
from loguru import logger

from ..protocol import RgbdFrameProtocol

from .priors import BallPosePrior
from .types import BallObservation, BallPoseDetectionConfig, BallPoseDetectionResult


@dataclass(frozen=True, slots=True)
class _ColorCandidate:
    """颜色分割产生的单个圆形候选。"""

    color_hex: str
    contour: np.ndarray
    mask: np.ndarray
    color_sample_mask: np.ndarray
    center_px: tuple[float, float]
    radius_px: float
    center_norm: np.ndarray
    radius_norm: float
    area_px: int
    circularity: float
    fill_ratio: float
    observed_hsv: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _BallDetection:
    """完成深度估计和评分后的内部球检测结果。"""

    color_hex: str
    detected: bool
    status: str
    center_mm: np.ndarray | None
    center_px: tuple[float, float] | None
    radius_px: float
    physical_diameter_mm: float
    diameter_error_ratio: float
    depth_points: int
    score: float
    contour: np.ndarray | None
    mask: np.ndarray | None
    center_norm: np.ndarray | None
    radius_norm: float
    observed_hsv: np.ndarray | None
    failure_reasons: list[str] = field(default_factory=list)


class BallPoseDetector:
    """根据传入的 frame 与小球先验，完成多球检测和位姿求解。"""

    def __init__(self, config: BallPoseDetectionConfig | None = None) -> None:
        self._config = BallPoseDetectionConfig() if config is None else config

    def detect(
        self, frame: RgbdFrameProtocol, priors: list[BallPosePrior]
    ) -> BallPoseDetectionResult:
        logger.info(
            "ball detector started camera_name={} frame_id={} prior_count={}",
            frame.camera_name,
            frame.frame_id,
            len(priors),
        )
        _validate_camera_intrinsics(frame)
        hsv = self._convert_to_hsv(frame.color_bgr)
        ranked: dict[str, list[_BallDetection]] = {}
        for prior in priors:
            ranges = (
                prior.hsv_ranges
                if prior.hsv_ranges
                else _reference_hsv_ranges(prior.color_hex, self._config)
            )
            mask = self._build_color_mask(hsv, ranges)
            candidates = self._collect_color_candidates(
                prior.color_hex,
                mask,
                hsv,
            )
            if candidates:
                logger.info(
                    "ball detector color candidates frame_id={} color={} candidate_count={}",
                    frame.frame_id,
                    prior.color_hex,
                    len(candidates),
                )
            else:
                logger.warning(
                    "ball detector color candidates missing frame_id={} color={}",
                    frame.frame_id,
                    prior.color_hex,
                )
            ranked[prior.color_hex] = self._rank_ball_candidates(
                frame, prior, candidates
            )
        detections = self._select_prior_consistent_detections(ranked, priors)
        observations = [
            self._to_observation(detection, priors) for detection in detections
        ]
        debug_colors = {
            prior.color_hex: _hex_to_bgr(prior.color_hex) for prior in priors
        }
        debug_diameters = {
            prior.color_hex: float(prior.diameter_mm) for prior in priors
        }
        debug_positions = {
            item.color_hex: np.asarray(item.center_mm, dtype=np.float64).copy()
            for item in observations
            if item.center_mm is not None
        }
        debug_model_positions = {
            prior.color_hex: np.asarray(prior.model_center_mm, dtype=np.float64).copy()
            for prior in priors
        }
        result = BallPoseDetectionResult(
            detections=observations,
            matched_count=sum(
                1
                for item in observations
                if item.detected and item.center_mm is not None
            ),
            debug_ball_colors_bgr=debug_colors,
            debug_ball_diameters_mm=debug_diameters,
            debug_ball_positions_mm=debug_positions,
            debug_ball_model_positions_mm=debug_model_positions,
            status=(
                "detected" if any(item.detected for item in observations) else "missing"
            ),
            timings_ms={},
        )
        logger.info(
            "ball detector completed camera_name={} frame_id={} status={} matched_count={}/{}",
            frame.camera_name,
            frame.frame_id,
            result.status,
            result.matched_count,
            len(priors),
        )
        return result

    @staticmethod
    def _convert_to_hsv(color_bgr: np.ndarray) -> np.ndarray:
        """将 BGR 图像平滑后转换为 OpenCV HSV 图像。"""

        blurred = cv2.GaussianBlur(np.asarray(color_bgr, dtype=np.uint8), (5, 5), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    @staticmethod
    def _build_color_mask(
        hsv: np.ndarray,
        ranges: tuple[tuple[int, int, int, int, int, int], ...],
    ) -> np.ndarray:
        """按单球专属或参考 HSV 范围构造颜色掩码。"""

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for range_item in ranges:
            if not (
                0 <= range_item[0] <= range_item[3] <= 179
                and 0 <= range_item[1] <= range_item[4] <= 255
                and 0 <= range_item[2] <= range_item[5] <= 255
            ):
                raise ValueError(f"invalid HSV range: {range_item}")
            lower = np.asarray(range_item[:3], dtype=np.uint8)
            upper = np.asarray(range_item[3:], dtype=np.uint8)
            combined = cv2.bitwise_or(
                combined,
                cv2.inRange(hsv, lower, upper),
            )
        kernel = np.ones((5, 5), dtype=np.uint8)
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        return cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

    def _collect_color_candidates(
        self,
        color_hex: str,
        mask: np.ndarray,
        hsv: np.ndarray,
    ) -> list[_ColorCandidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[_ColorCandidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(self._config.min_component_area_px):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < float(self._config.min_circularity):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius <= 1.0:
                continue
            circle_area = float(np.pi * radius * radius)
            fill_ratio = float(area / max(1.0, circle_area))
            if fill_ratio < float(self._config.min_fill_ratio):
                continue
            candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(
                candidate_mask,
                [contour],
                -1,
                (255.0,),
                thickness=cv2.FILLED,
            )
            color_sample_mask = cv2.bitwise_and(candidate_mask, mask)
            center_norm, radius_norm = self._normalize_geometry(contour)
            candidates.append(
                _ColorCandidate(
                    color_hex=color_hex,
                    contour=contour.reshape(-1, 2).astype(np.int32),
                    mask=candidate_mask,
                    color_sample_mask=color_sample_mask,
                    center_px=(float(cx), float(cy)),
                    radius_px=float(radius),
                    center_norm=center_norm,
                    radius_norm=radius_norm,
                    area_px=int(round(area)),
                    circularity=circularity,
                    fill_ratio=fill_ratio,
                    observed_hsv=self._estimate_observed_hsv(
                        hsv,
                        color_sample_mask,
                    ),
                )
            )
        candidates.sort(key=lambda item: item.area_px, reverse=True)
        return candidates[: int(self._config.max_color_components)]

    def _rank_ball_candidates(
        self,
        frame: RgbdFrameProtocol,
        prior: BallPosePrior,
        candidates: list[_ColorCandidate],
    ) -> list[_BallDetection]:
        if not candidates:
            return [self._missing_detection(prior, "no_color_component")]
        detections: list[_BallDetection] = []
        rejected_by_depth = 0
        rejected_by_diameter = 0
        for candidate in candidates:
            center_mm, ball_points = self._estimate_center_mm(frame, candidate.mask)
            if center_mm is None:
                rejected_by_depth += 1
                continue
            physical_diameter = self._estimate_physical_diameter_mm(
                center_mm=center_mm,
                radius_px=candidate.radius_px,
                intrinsics=frame,
            )
            diameter_error_ratio = abs(
                physical_diameter - prior.diameter_mm
            ) / max(
                1.0e-6, prior.diameter_mm
            )
            if (
                not np.isfinite(physical_diameter)
                or diameter_error_ratio
                > self._config.max_diameter_error_ratio
            ):
                rejected_by_diameter += 1
                continue
            diameter_score = max(0.0, 1.0 - diameter_error_ratio)
            depth_score = (
                0.0
                if ball_points <= 0
                else min(1.0, ball_points / max(1, self._config.min_depth_points * 12))
            )
            circle_score = float(np.clip(candidate.circularity, 0.0, 1.0))
            fill_score = float(np.clip(candidate.fill_ratio, 0.0, 1.0))
            border_score = self._score_inside_image(
                candidate.center_px, candidate.radius_px, frame
            )
            score = (
                0.44 * diameter_score
                + 0.06 * depth_score
                + 0.22 * circle_score
                + 0.18 * fill_score
                + 0.10 * border_score
            )
            detections.append(
                _BallDetection(
                    color_hex=prior.color_hex,
                    detected=True,
                    status="detected",
                    center_mm=center_mm,
                    center_px=candidate.center_px,
                    radius_px=candidate.radius_px,
                    physical_diameter_mm=physical_diameter,
                    diameter_error_ratio=diameter_error_ratio,
                    depth_points=int(ball_points),
                    score=float(score),
                    contour=candidate.contour,
                    mask=candidate.mask,
                    center_norm=candidate.center_norm,
                    radius_norm=candidate.radius_norm,
                    observed_hsv=candidate.observed_hsv,
                    failure_reasons=[],
                )
            )
        if not detections:
            if rejected_by_diameter > 0:
                logger.warning(
                    "ball detector rejected all candidates by diameter color={} rejected_count={} prior_diameter_mm={:.3f}",
                    prior.color_hex,
                    rejected_by_diameter,
                    prior.diameter_mm,
                )
                return [
                    self._missing_detection(
                        prior,
                        "diameter_mismatch",
                        status="diameter_mismatch",
                    )
                ]
            if rejected_by_depth > 0:
                return [
                    self._missing_detection(
                        prior,
                        "depth_weak",
                        status="depth_weak",
                    )
                ]
            return [self._missing_detection(prior, "no_valid_candidate")]
        # 首次先验采集没有球间位置尺度，必须先选物理直径最接近先验的候选。
        detections.sort(
            key=lambda item: (item.diameter_error_ratio, -item.score)
        )
        return detections

    def _estimate_center_mm(
        self, frame: RgbdFrameProtocol, mask: np.ndarray
    ) -> tuple[np.ndarray | None, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None, 0
        depth = np.asarray(frame.depth_mm, dtype=np.float64)
        sampled_depth = depth[ys, xs]
        valid = np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
        if int(np.count_nonzero(valid)) < int(self._config.min_depth_points):
            return None, int(np.count_nonzero(valid))
        sampled_depth = sampled_depth[valid]
        ys = ys[valid]
        xs = xs[valid]
        lo = float(np.quantile(sampled_depth, self._config.depth_trim_ratio))
        hi = float(np.quantile(sampled_depth, 1.0 - self._config.depth_trim_ratio))
        keep = (sampled_depth >= lo) & (sampled_depth <= hi)
        if int(np.count_nonzero(keep)) < int(self._config.min_depth_points):
            return None, int(np.count_nonzero(keep))
        xs = xs[keep]
        ys = ys[keep]
        sampled_depth = sampled_depth[keep]
        x = (xs.astype(np.float64) - float(frame.cx)) * sampled_depth / float(frame.fx)
        y = (ys.astype(np.float64) - float(frame.cy)) * sampled_depth / float(frame.fy)
        xyz = np.stack([x, y, sampled_depth], axis=1)
        xyz = self._smooth_surface_points(xyz)
        if xyz.shape[0] < int(self._config.min_depth_points):
            return None, int(xyz.shape[0])
        center_mm = self._fit_sphere_center(xyz)
        if center_mm is None:
            return None, int(xyz.shape[0])
        return center_mm.astype(np.float64), int(xyz.shape[0])

    def _select_prior_consistent_detections(
        self,
        ranked_by_color: dict[str, list[_BallDetection]],
        priors: list[BallPosePrior],
    ) -> list[_BallDetection]:
        """优先按三球模型相对距离选择完整候选组合。

        Parameters
        ----------
        ranked_by_color:
            各颜色通过深度和直径硬校验后的候选列表，列表内按单球外观分数降序排列。
        priors:
            按业务球顺序排列的先验，`model_center_mm` 位于同一模型坐标系，单位 mm。

        Returns
        -------
        list[_BallDetection]
            与先验顺序一致的三球检测。有效尺度先验下若没有完整几何一致组合，
            所有球均返回 `relative_geometry_mismatch`，避免输出错误三球位置。

        Notes
        -----
        三球边长在刚体变换下保持不变，因此无需已知相机外参。几何误差先于颜色、
        圆度和填充率评分参与排序；占位先验没有实际尺度时退回逐球选择，但候选仍已
        通过物理直径硬校验。
        """

        fallback: list[_BallDetection] = []
        for prior in priors:
            ranked = ranked_by_color.get(prior.color_hex, [])
            fallback.append(
                ranked[0] if ranked else self._missing_detection(prior, "missing")
            )
        if len(priors) != 3 or not self.has_metric_relative_prior(priors):
            return fallback

        candidate_groups = [
            [
                detection
                for detection in ranked_by_color.get(prior.color_hex, [])
                if detection.detected and detection.center_mm is not None
            ]
            for prior in priors
        ]
        if any(not group for group in candidate_groups):
            return self._relative_geometry_mismatch(priors)

        best_combo: tuple[_BallDetection, ...] | None = None
        best_rank: tuple[float, float, float] | None = None
        for combo in itertools.product(*candidate_groups):
            max_error_ratio, mean_error_ratio = self._relative_distance_errors(
                combo, priors
            )
            if (
                max_error_ratio
                > self._config.max_relative_distance_error_ratio
            ):
                continue
            # 几何最大误差和平均误差是主排序键；单球外观分数只打破几何近似平局。
            appearance_score = float(np.mean([item.score for item in combo]))
            rank = (max_error_ratio, mean_error_ratio, -appearance_score)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_combo = combo

        if best_combo is None:
            logger.warning(
                "ball detector rejected all three-ball combinations by relative geometry tolerance={:.3f}",
                self._config.max_relative_distance_error_ratio,
            )
            return self._relative_geometry_mismatch(priors)
        return list(best_combo)

    def has_metric_relative_prior(self, priors: list[BallPosePrior]) -> bool:
        """判断三球模型中心是否包含可用于距离约束的实际尺度。

        Parameters
        ----------
        priors:
            三球模型先验，中心单位 mm，直径单位 mm。

        Returns
        -------
        bool
            三条模型边长均明显大于占位尺度时为 `True`。
        """

        if len(priors) != 3:
            return False
        model_centers = [
            np.asarray(prior.model_center_mm, dtype=np.float64) for prior in priors
        ]
        if any(
            center.shape != (3,) or not np.all(np.isfinite(center))
            for center in model_centers
        ):
            return False
        max_diameter_mm = max(prior.diameter_mm for prior in priors)
        minimum_metric_distance_mm = (
            max_diameter_mm * self._config.min_center_distance_ratio
        )
        for left_index, right_index in itertools.combinations(range(3), 2):
            model_distance_mm = float(
                np.linalg.norm(
                    model_centers[left_index] - model_centers[right_index]
                )
            )
            if model_distance_mm < minimum_metric_distance_mm:
                return False
        return True

    @staticmethod
    def _relative_distance_errors(
        detections: tuple[_BallDetection, ...],
        priors: list[BallPosePrior],
    ) -> tuple[float, float]:
        """计算候选三球与模型三条边长的相对误差。

        Parameters
        ----------
        detections:
            与先验顺序一致且包含相机坐标球心的三个候选，中心单位 mm。
        priors:
            三球模型先验，模型中心单位 mm。

        Returns
        -------
        max_error_ratio:
            三条边中的最大相对误差。
        mean_error_ratio:
            三条边相对误差的平均值。
        """

        error_ratios: list[float] = []
        for left_index, right_index in itertools.combinations(range(3), 2):
            left_center = detections[left_index].center_mm
            right_center = detections[right_index].center_mm
            if left_center is None or right_center is None:
                return float("inf"), float("inf")
            observed_distance_mm = float(
                np.linalg.norm(left_center - right_center)
            )
            model_distance_mm = float(
                np.linalg.norm(
                    priors[left_index].model_center_mm
                    - priors[right_index].model_center_mm
                )
            )
            error_ratios.append(
                abs(observed_distance_mm - model_distance_mm)
                / max(1.0e-6, model_distance_mm)
            )
        return max(error_ratios), float(np.mean(error_ratios))

    def _relative_geometry_mismatch(
        self, priors: list[BallPosePrior]
    ) -> list[_BallDetection]:
        """构造三球相对位置不一致时的保守未检出结果。"""

        return [
            self._missing_detection(
                prior,
                "relative_geometry_mismatch",
                status="relative_geometry_mismatch",
            )
            for prior in priors
        ]

    def _missing_detection(
        self,
        prior: BallPosePrior,
        reason: str,
        *,
        status: str = "missing",
    ) -> _BallDetection:
        return _BallDetection(
            color_hex=prior.color_hex,
            detected=False,
            status=status,
            center_mm=None,
            center_px=None,
            radius_px=0.0,
            physical_diameter_mm=0.0,
            diameter_error_ratio=float("inf"),
            depth_points=0,
            score=0.0,
            contour=None,
            mask=None,
            center_norm=None,
            radius_norm=0.0,
            observed_hsv=None,
            failure_reasons=[reason],
        )

    def _to_observation(
        self, detection: _BallDetection, priors: list[BallPosePrior]
    ) -> BallObservation:
        prior = next(item for item in priors if item.color_hex == detection.color_hex)
        return BallObservation(
            color_hex=prior.color_hex,
            detected=detection.detected,
            center_px=(
                None
                if detection.center_px is None
                else np.asarray(detection.center_px, dtype=np.float64)
            ),
            center_mm=(
                None
                if detection.center_mm is None
                else np.asarray(detection.center_mm, dtype=np.float64)
            ),
            diameter_mm=(
                detection.physical_diameter_mm
                if detection.detected
                else float(prior.diameter_mm)
            ),
            radius_px=float(detection.radius_px),
            contour=detection.contour,
            mask=detection.mask,
            center_norm=(
                None
                if detection.center_norm is None
                else np.asarray(detection.center_norm, dtype=np.float64)
            ),
            radius_norm=float(detection.radius_norm),
            point_count=int(detection.depth_points),
            debug_bgr=_hex_to_bgr(prior.color_hex),
            status=detection.status,
            observed_hsv=(
                None
                if detection.observed_hsv is None
                else np.asarray(detection.observed_hsv, dtype=np.float64)
            ),
        )

    def _estimate_observed_hsv(
        self,
        hsv: np.ndarray,
        color_sample_mask: np.ndarray,
    ) -> np.ndarray | None:
        """估计候选颜色像素的稳健 HSV 中心。

        OpenCV Hue 的周期为 180，因此 H 使用圆均值，避免红色同时落在 0 和 179
        附近时得到错误的中间色相；S、V 使用中位数抑制高光与阴影。
        """

        pixels = np.asarray(hsv[color_sample_mask > 0], dtype=np.float64)
        if pixels.shape[0] < self._config.min_color_sample_pixels:
            return None
        hue_angles = pixels[:, 0] * (2.0 * np.pi / 180.0)
        hue_angle = np.arctan2(
            np.mean(np.sin(hue_angles)),
            np.mean(np.cos(hue_angles)),
        )
        hue = float((hue_angle % (2.0 * np.pi)) * 180.0 / (2.0 * np.pi))
        return np.asarray(
            [
                hue,
                float(np.median(pixels[:, 1])),
                float(np.median(pixels[:, 2])),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _estimate_physical_diameter_mm(
        center_mm: np.ndarray,
        radius_px: float,
        intrinsics: RgbdFrameProtocol,
    ) -> float:
        focal = 0.5 * (float(intrinsics.fx) + float(intrinsics.fy))
        return (
            2.0 * float(radius_px) * float(center_mm[2]) / max(1e-6, focal)
        )

    @staticmethod
    def _score_inside_image(
        center_px: tuple[float, float],
        radius_px: float,
        frame: RgbdFrameProtocol,
    ) -> float:
        h = int(np.asarray(frame.color_bgr).shape[0])
        w = int(np.asarray(frame.color_bgr).shape[1])
        cx, cy = center_px
        margin = float(radius_px) * 0.8
        if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
            return 0.35
        return 1.0

    @staticmethod
    def _normalize_geometry(contour: np.ndarray) -> tuple[np.ndarray, float]:
        points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] == 0:
            return np.zeros((2,), dtype=np.float64), 0.0
        center = np.mean(points, axis=0)
        centered = points - center.reshape(1, 2)
        return center, float(np.mean(np.linalg.norm(centered, axis=1)))

    @staticmethod
    def _smooth_surface_points(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] < 10:
            return pts
        center = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
        if not np.all(np.isfinite(dist)):
            return pts
        keep = dist <= float(np.quantile(dist, 0.85))
        if int(np.count_nonzero(keep)) < 10:
            return pts
        return pts[keep]

    @staticmethod
    def _fit_sphere_center(points: np.ndarray) -> np.ndarray | None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] < 4:
            return None
        a = np.column_stack([2.0 * pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        b = np.sum(pts * pts, axis=1)
        try:
            sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        center = np.asarray(sol[:3], dtype=np.float64)
        if not np.all(np.isfinite(center)):
            return None
        return center


def _validate_camera_intrinsics(frame: RgbdFrameProtocol) -> None:
    """拒绝无法用于三维反投影的相机内参。"""

    intrinsics = np.asarray(
        [frame.fx, frame.fy, frame.cx, frame.cy],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(intrinsics)):
        raise ValueError(
            "camera intrinsics must be finite: "
            f"fx={frame.fx} fy={frame.fy} cx={frame.cx} cy={frame.cy}"
        )
    if frame.fx <= 0.0 or frame.fy <= 0.0:
        raise ValueError(
            "camera intrinsics fx/fy must be greater than zero: "
            f"fx={frame.fx} fy={frame.fy} cx={frame.cx} cy={frame.cy}"
        )


def _hex_to_bgr(color_hex: str) -> np.ndarray:
    hex_text = color_hex.lstrip("#")
    if len(hex_text) != 6:
        raise ValueError(f"invalid color hex: {color_hex}")
    r = int(hex_text[0:2], 16)
    g = int(hex_text[2:4], 16)
    b = int(hex_text[4:6], 16)
    return np.asarray([b, g, r], dtype=np.uint8)


def _reference_hsv_ranges(
    color_hex: str,
    config: BallPoseDetectionConfig,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """根据请求传入的 RGB HEX 颜色生成首次检测使用的 HSV 宽范围。

    算法不维护任何具名球色表。Hue 在 OpenCV 的 0/179 边界处自动拆成两个范围，
    因此红色与任意用户选择的颜色都走同一条计算路径。
    """

    bgr = _hex_to_bgr(color_hex).reshape(1, 1, 3)
    hue, saturation, value = (
        int(component)
        for component in cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    )
    hue_min = hue - config.reference_hue_tolerance
    hue_max = hue + config.reference_hue_tolerance
    saturation_min = max(
        0,
        saturation - config.reference_saturation_tolerance,
    )
    value_min = max(0, value - config.reference_value_tolerance)
    if hue_min < 0:
        return (
            (0, saturation_min, value_min, hue_max, 255, 255),
            (180 + hue_min, saturation_min, value_min, 179, 255, 255),
        )
    if hue_max > 179:
        return (
            (hue_min, saturation_min, value_min, 179, 255, 255),
            (0, saturation_min, value_min, hue_max - 180, 255, 255),
        )
    return ((hue_min, saturation_min, value_min, hue_max, 255, 255),)
