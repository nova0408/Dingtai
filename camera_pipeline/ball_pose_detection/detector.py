from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .priors import BallPosePrior
from .types import BallObservation, BallPoseDetectionConfig, BallPoseDetectionResult


@dataclass(frozen=True)
class _PoseFitResult:
    rotation: np.ndarray
    translation_mm: np.ndarray
    residual_mm: float


@dataclass(frozen=True)
class _ColorCandidate:
    color_hex: str
    contour: np.ndarray
    mask: np.ndarray
    center_px: tuple[float, float]
    radius_px: float
    center_norm: np.ndarray
    radius_norm: float
    area_px: int
    circularity: float
    fill_ratio: float


@dataclass(frozen=True)
class _BallDetection:
    color_hex: str
    detected: bool
    status: str
    center_mm: np.ndarray | None
    center_px: tuple[float, float] | None
    radius_px: float
    physical_radius_mm: float
    depth_points: int
    score: float
    contour: np.ndarray | None
    mask: np.ndarray | None
    center_norm: np.ndarray | None
    radius_norm: float
    failure_reasons: list[str] = field(default_factory=list)


class BallPoseDetector:
    """根据传入的 frame 与小球先验，完成多球检测和位姿求解。"""

    def __init__(self, config: BallPoseDetectionConfig | None = None) -> None:
        self._config = BallPoseDetectionConfig() if config is None else config

    def detect(self, frame: Any, priors: list[BallPosePrior]) -> BallPoseDetectionResult:
        masks = self._build_color_masks(frame.color_bgr)
        ranked: dict[str, list[_BallDetection]] = {}
        for prior in priors:
            candidates = self._collect_color_candidates(prior.color_hex, masks[prior.color_hex])
            ranked[prior.color_hex] = self._rank_ball_candidates(frame, prior, candidates)
        detections = self._resolve_duplicates(ranked, priors)
        observations = [self._to_observation(detection, priors) for detection in detections]
        debug_colors = {prior.color_hex: _hex_to_bgr(prior.color_hex) for prior in priors}
        debug_radii = {prior.color_hex: float(prior.radius_mm) for prior in priors}
        debug_positions = {
            item.color_hex: np.asarray(item.center_mm, dtype=np.float64).copy()
            for item in observations
            if item.center_mm is not None
        }
        debug_model_positions = {prior.color_hex: np.asarray(prior.model_center_mm, dtype=np.float64).copy() for prior in priors}
        fit = self._estimate_pose(observations, priors)
        pose_transform = None
        pose_translation_mm = None
        pose_rotation = None
        residual_mm = None
        status = "pose_unavailable"
        if fit is not None:
            pose_rotation = fit.rotation
            pose_translation_mm = fit.translation_mm
            pose_transform = np.eye(4, dtype=np.float64)
            pose_transform[:3, :3] = pose_rotation
            pose_transform[:3, 3] = pose_translation_mm
            residual_mm = fit.residual_mm
            status = "pose_estimated"
        return BallPoseDetectionResult(
            detections=observations,
            pose_translation_mm=pose_translation_mm,
            pose_rotation=pose_rotation,
            pose_transform=pose_transform,
            residual_mm=residual_mm,
            matched_count=sum(1 for item in observations if item.detected and item.center_mm is not None),
            debug_ball_colors_bgr=debug_colors,
            debug_ball_radii_mm=debug_radii,
            debug_ball_positions_mm=debug_positions,
            debug_ball_model_positions_mm=debug_model_positions,
            status=status,
            timings_ms={},
        )

    def _estimate_pose(self, detections: list[BallObservation], priors: list[BallPosePrior]) -> _PoseFitResult | None:
        model_points: list[np.ndarray] = []
        camera_points: list[np.ndarray] = []
        for prior in priors:
            match = next((item for item in detections if item.color_hex == prior.color_hex and item.center_mm is not None), None)
            if match is None:
                continue
            model_points.append(np.asarray(prior.model_center_mm, dtype=np.float64))
            camera_points.append(np.asarray(match.center_mm, dtype=np.float64))
        if len(model_points) < 3:
            return None
        model_stack = np.stack(model_points, axis=0)
        camera_stack = np.stack(camera_points, axis=0)
        anchor_fit = self._fit_rigid_transform(model_stack[:3], camera_stack[:3])
        if model_stack.shape[0] == 3:
            return anchor_fit
        full_fit = self._fit_rigid_transform(model_stack, camera_stack)
        if full_fit.residual_mm <= anchor_fit.residual_mm + 1e-6:
            return full_fit
        return anchor_fit

    def _build_color_masks(self, color_bgr: np.ndarray) -> dict[str, np.ndarray]:
        blurred = cv2.GaussianBlur(np.asarray(color_bgr, dtype=np.uint8), (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        masks: dict[str, np.ndarray] = {}
        kernel = np.ones((5, 5), dtype=np.uint8)
        for color_hex, ranges in self._config.color_ranges.items():
            combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for range_item in ranges:
                lower = np.asarray(range_item[:3], dtype=np.uint8)
                upper = np.asarray(range_item[3:], dtype=np.uint8)
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
            masks[color_hex] = combined
        return masks

    def _collect_color_candidates(self, color_hex: str, mask: np.ndarray) -> list[_ColorCandidate]:
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
            cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=cv2.FILLED)
            center_norm, radius_norm = self._normalize_geometry(contour)
            candidates.append(
                _ColorCandidate(
                    color_hex=color_hex,
                    contour=contour.reshape(-1, 2).astype(np.int32),
                    mask=candidate_mask,
                    center_px=(float(cx), float(cy)),
                    radius_px=float(radius),
                    center_norm=center_norm,
                    radius_norm=radius_norm,
                    area_px=int(round(area)),
                    circularity=circularity,
                    fill_ratio=fill_ratio,
                )
            )
        candidates.sort(key=lambda item: item.area_px, reverse=True)
        return candidates[: int(self._config.max_color_components)]

    def _rank_ball_candidates(self, frame: Any, prior: BallPosePrior, candidates: list[_ColorCandidate]) -> list[_BallDetection]:
        if not candidates:
            return [self._missing_detection(prior, "no_color_component")]
        detections: list[_BallDetection] = []
        for candidate in candidates:
            center_mm, ball_points = self._estimate_center_mm(frame, candidate.mask)
            physical_radius = 0.0 if center_mm is None else self._estimate_physical_radius_mm(center_mm=center_mm, radius_px=candidate.radius_px, intrinsics=frame)
            expected_radius_norm = float(prior.radius_mm) / max(1e-6, 0.5 * (float(frame.fx) + float(frame.fy)))
            radius_score = max(0.0, 1.0 - abs(candidate.radius_norm - expected_radius_norm) / max(1.0e-6, expected_radius_norm))
            depth_score = 0.0 if ball_points <= 0 else min(1.0, ball_points / max(1, self._config.min_depth_points * 12))
            circle_score = float(np.clip(candidate.circularity, 0.0, 1.0))
            fill_score = float(np.clip(candidate.fill_ratio, 0.0, 1.0))
            border_score = self._score_inside_image(candidate.center_px, candidate.radius_px, frame)
            score = 0.44 * radius_score + 0.06 * depth_score + 0.22 * circle_score + 0.18 * fill_score + 0.10 * border_score
            detections.append(
                _BallDetection(
                    color_hex=prior.color_hex,
                    detected=True,
                    status="detected" if center_mm is not None else "depth_weak",
                    center_mm=center_mm,
                    center_px=candidate.center_px,
                    radius_px=candidate.radius_px,
                    physical_radius_mm=physical_radius,
                    depth_points=int(ball_points),
                    score=float(score),
                    contour=candidate.contour,
                    mask=candidate.mask,
                    center_norm=candidate.center_norm,
                    radius_norm=candidate.radius_norm,
                    failure_reasons=[],
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections

    def _estimate_center_mm(self, frame: Any, mask: np.ndarray) -> tuple[np.ndarray | None, int]:
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
        return np.mean(xyz, axis=0).astype(np.float64), int(xyz.shape[0])

    def _resolve_duplicates(self, ranked_by_color: dict[str, list[_BallDetection]], priors: list[BallPosePrior]) -> list[_BallDetection]:
        resolved: dict[str, _BallDetection] = {}
        for prior in priors:
            ranked = ranked_by_color.get(prior.color_hex, [])
            resolved[prior.color_hex] = ranked[0] if ranked else self._missing_detection(prior, "missing")
        return list(resolved.values())

    def _missing_detection(self, prior: BallPosePrior, reason: str) -> _BallDetection:
        return _BallDetection(
            color_hex=prior.color_hex,
            detected=False,
            status="missing",
            center_mm=None,
            center_px=None,
            radius_px=0.0,
            physical_radius_mm=0.0,
            depth_points=0,
            score=0.0,
            contour=None,
            mask=None,
            center_norm=None,
            radius_norm=0.0,
            failure_reasons=[reason],
        )

    def _to_observation(self, detection: _BallDetection, priors: list[BallPosePrior]) -> BallObservation:
        prior = next(item for item in priors if item.color_hex == detection.color_hex)
        return BallObservation(
            color_hex=prior.color_hex,
            detected=detection.detected,
            center_px=None if detection.center_px is None else np.asarray(detection.center_px, dtype=np.float64),
            center_mm=None if detection.center_mm is None else np.asarray(detection.center_mm, dtype=np.float64),
            radius_mm=float(prior.radius_mm),
            radius_px=float(detection.radius_px),
            contour=detection.contour,
            mask=detection.mask,
            center_norm=None if detection.center_norm is None else np.asarray(detection.center_norm, dtype=np.float64),
            radius_norm=float(detection.radius_norm),
            point_count=int(detection.depth_points),
            debug_bgr=_hex_to_bgr(prior.color_hex),
            status=detection.status,
        )

    @staticmethod
    def _estimate_physical_radius_mm(center_mm: np.ndarray, radius_px: float, intrinsics: Any) -> float:
        focal = 0.5 * (float(intrinsics.fx) + float(intrinsics.fy))
        return float(radius_px) * float(center_mm[2]) / max(1e-6, focal)

    @staticmethod
    def _score_inside_image(center_px: tuple[float, float], radius_px: float, frame: Any) -> float:
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
    def _fit_rigid_transform(model_points: np.ndarray, camera_points: np.ndarray) -> _PoseFitResult:
        model_center = np.mean(model_points, axis=0)
        camera_center = np.mean(camera_points, axis=0)
        model_centered = model_points - model_center.reshape(1, 3)
        camera_centered = camera_points - camera_center.reshape(1, 3)
        h = model_centered.T @ camera_centered
        u, _, vt = np.linalg.svd(h)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        translation = camera_center - rotation @ model_center
        aligned = (model_points @ rotation.T) + translation.reshape(1, 3)
        residual = float(np.sqrt(np.mean(np.sum((aligned - camera_points) ** 2, axis=1))))
        return _PoseFitResult(rotation=rotation.astype(np.float64), translation_mm=translation.astype(np.float64), residual_mm=residual)


def _hex_to_bgr(color_hex: str) -> np.ndarray:
    hex_text = color_hex.lstrip("#")
    if len(hex_text) != 6:
        raise ValueError(f"invalid color hex: {color_hex}")
    r = int(hex_text[0:2], 16)
    g = int(hex_text[2:4], 16)
    b = int(hex_text[4:6], 16)
    return np.asarray([b, g, r], dtype=np.uint8)
