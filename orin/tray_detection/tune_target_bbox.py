from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig

from .detector import TrayPointExcluder
from .types import TrayDetection, TrayDetectionConfig


# region 常量
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "target_bbox_tuning"
DEFAULT_TARGET_BOX_XYXY = (620, 335, 1060, 473)
DEFAULT_SAMPLE_FRAME_COUNT = 3
DEFAULT_EXPECTED_TRAY_COUNT = 2
DEFAULT_CAMERA_CONFIG = CameraStreamRuntimeConfig()
# endregion


# region 数据结构
@dataclass(frozen=True)
class CandidateSpec:
    """单个候选配置。"""

    name: str
    prompt: str
    target_keywords: str
    min_confidence: float
    topk_objects: int
    combine_prompts_forward: bool
    detect_max_side: int
    strict_target_filter: bool = True
    max_targets: int = 1
    box_threshold: float = 0.16
    text_threshold: float = 0.08


@dataclass(frozen=True)
class CandidateFrameScore:
    """单帧评分结果。"""

    frame_id: int
    score: float
    count_score: float
    region_score: float
    ordering_score: float
    split_score: float
    confidence: float
    has_detection: bool
    tray_count: int
    bbox_xywh_list: list[tuple[int, int, int, int]]
    label_texts: list[str]


@dataclass(frozen=True)
class CandidateRunSummary:
    """单个候选配置的总体结果。"""

    name: str
    prompt: str
    target_keywords: str
    min_confidence: float
    topk_objects: int
    combine_prompts_forward: bool
    detect_max_side: int
    strict_target_filter: bool
    mean_score: float
    max_score: float
    mean_count_score: float
    mean_region_score: float
    mean_ordering_score: float
    mean_split_score: float
    mean_confidence: float
    detection_rate: float
    frame_scores: list[CandidateFrameScore]
    selected_bbox_xywh_list: list[tuple[int, int, int, int]]
    selected_label_texts: list[str]
    selected_frame_id: int
    overlay_path: str
# endregion


# region 主流程
def main(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_frame_count: int = DEFAULT_SAMPLE_FRAME_COUNT,
    target_box_xyxy: tuple[int, int, int, int] = DEFAULT_TARGET_BOX_XYXY,
    expected_tray_count: int = DEFAULT_EXPECTED_TRAY_COUNT,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _capture_sample_frames(sample_frame_count)
    if len(frames) == 0:
        raise RuntimeError("未能从真实相机流中获取样本帧。")
    candidate_specs = _build_candidate_specs()
    summaries: list[CandidateRunSummary] = []
    for candidate in candidate_specs:
        summaries.append(_run_candidate(candidate, frames, target_box_xyxy, expected_tray_count, output_dir))
    summaries.sort(key=lambda item: item.mean_score, reverse=True)
    _write_summary_files(summaries, frames, target_box_xyxy, output_dir)
    _print_top_results(summaries)
    return 0


def _capture_sample_frames(sample_frame_count: int) -> list:
    runtime = CameraStreamRuntime(DEFAULT_CAMERA_CONFIG)
    runtime.start()
    frames = []
    seen_frame_ids: set[int] = set()
    try:
        if not runtime.wait_until_ready(timeout_s=10.0):
            raise RuntimeError("共享相机流在超时时间内未准备好。")
        while len(frames) < max(1, int(sample_frame_count)):
            frame = runtime.get_latest_frame()
            if frame is None:
                continue
            if int(frame.frame_id) in seen_frame_ids:
                continue
            seen_frame_ids.add(int(frame.frame_id))
            frames.append(frame)
    finally:
        runtime.stop()
    return frames


def _run_candidate(
    candidate: CandidateSpec,
    frames: list,
    target_box_xyxy: tuple[int, int, int, int],
    expected_tray_count: int,
    output_dir: Path,
) -> CandidateRunSummary:
    config = TrayDetectionConfig(
        prompt=str(candidate.prompt),
        target_keywords=str(candidate.target_keywords),
        strict_target_filter=bool(candidate.strict_target_filter),
        max_targets=int(candidate.max_targets),
        use_sam=False,
        min_confidence=float(candidate.min_confidence),
        topk_objects=int(candidate.topk_objects),
        sam_max_boxes=1,
        sam_primary_only=True,
        combine_prompts_forward=bool(candidate.combine_prompts_forward),
        detect_max_side=int(candidate.detect_max_side),
        box_threshold=float(candidate.box_threshold),
        text_threshold=float(candidate.text_threshold),
    )
    detector = TrayPointExcluder(config)
    frame_scores: list[CandidateFrameScore] = []
    selected_overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
    selected_bbox_list: list[tuple[int, int, int, int]] = []
    selected_label_texts: list[str] = []
    selected_frame_id = -1
    best_frame_score = -1.0
    try:
        for frame in frames:
            detections = detector.detect(np.asarray(frame.color_bgr, dtype=np.uint8))
            frame_score, overlay = _score_frame(candidate, frame, detections, target_box_xyxy, expected_tray_count)
            frame_scores.append(frame_score)
            if frame_score.score > best_frame_score:
                best_frame_score = float(frame_score.score)
                selected_overlay = overlay
                selected_bbox_list = list(frame_score.bbox_xywh_list)
                selected_label_texts = list(frame_score.label_texts)
                selected_frame_id = int(frame.frame_id)
    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    overlay_path = output_dir / f"{candidate.name}.jpg"
    cv2.imwrite(str(overlay_path), selected_overlay)
    mean_score = float(np.mean([item.score for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    mean_count_score = float(np.mean([item.count_score for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    mean_region_score = float(np.mean([item.region_score for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    mean_ordering_score = float(np.mean([item.ordering_score for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    mean_split_score = float(np.mean([item.split_score for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    mean_confidence = float(np.mean([item.confidence for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    detection_rate = float(np.mean([1.0 if item.has_detection else 0.0 for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    max_score = float(np.max([item.score for item in frame_scores])) if len(frame_scores) > 0 else 0.0
    return CandidateRunSummary(
        name=str(candidate.name),
        prompt=str(candidate.prompt),
        target_keywords=str(candidate.target_keywords),
        min_confidence=float(candidate.min_confidence),
        topk_objects=int(candidate.topk_objects),
        combine_prompts_forward=bool(candidate.combine_prompts_forward),
        detect_max_side=int(candidate.detect_max_side),
        strict_target_filter=bool(candidate.strict_target_filter),
        mean_score=mean_score,
        max_score=max_score,
        mean_count_score=mean_count_score,
        mean_region_score=mean_region_score,
        mean_ordering_score=mean_ordering_score,
        mean_split_score=mean_split_score,
        mean_confidence=mean_confidence,
        detection_rate=detection_rate,
        frame_scores=frame_scores,
        selected_bbox_xywh_list=selected_bbox_list,
        selected_label_texts=selected_label_texts,
        selected_frame_id=selected_frame_id,
        overlay_path=str(overlay_path),
    )


def _score_frame(
    candidate: CandidateSpec,
    frame,
    detections: list[TrayDetection],
    target_box_xyxy: tuple[int, int, int, int],
    expected_tray_count: int,
) -> tuple[CandidateFrameScore, np.ndarray]:
    overlay = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
    _draw_target_box(overlay, target_box_xyxy)
    sorted_detections = sorted(list(detections), key=lambda det: _mask_center_uv(np.asarray(det.mask, dtype=np.uint8))[0])
    if len(sorted_detections) == 0:
        cv2.putText(overlay, candidate.name, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return (
            CandidateFrameScore(
                frame_id=int(frame.frame_id),
                score=0.0,
                count_score=0.0,
                region_score=0.0,
                ordering_score=0.0,
                split_score=0.0,
                confidence=0.0,
                has_detection=False,
                tray_count=0,
                bbox_xywh_list=[],
                label_texts=[],
            ),
            overlay,
        )
    candidate_score = _compute_candidate_frame_score(
        frame_id=int(frame.frame_id),
        detections=sorted_detections,
        target_box_xyxy=target_box_xyxy,
        expected_tray_count=expected_tray_count,
    )
    for det_index, det in enumerate(sorted_detections):
        mask = np.asarray(det.mask, dtype=np.uint8)
        color = _debug_color_bgr(det_index)
        overlay = _blend_mask_overlay(overlay, mask, color, 0.22)
        _draw_mask_outline(overlay, mask, color)
        x, y, w, h = _mask_bbox_xywh(mask)
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"tray_{det_index} {float(det.confidence_2d):.2f}",
            (x, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        "{0} score {1:.3f} count {2}".format(candidate.name, candidate_score.score, candidate_score.tray_count),
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "region {0:.2f} order {1:.2f} split {2:.2f}".format(
            candidate_score.region_score,
            candidate_score.ordering_score,
            candidate_score.split_score,
        ),
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return candidate_score, overlay


def _write_summary_files(
    summaries: list[CandidateRunSummary],
    frames: list,
    target_box_xyxy: tuple[int, int, int, int],
    output_dir: Path,
) -> None:
    payload = {
        "target_box_xyxy": list(target_box_xyxy),
        "frame_ids": [int(frame.frame_id) for frame in frames],
        "results": [_summary_to_json(item) for item in summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "target_box_xyxy={0}".format(target_box_xyxy),
        "frame_ids={0}".format([int(frame.frame_id) for frame in frames]),
        "",
    ]
    for index, item in enumerate(summaries, start=1):
        lines.append(
            "{0}. {1} mean_score={2:.4f} count={3:.4f} region={4:.4f} order={5:.4f} split={6:.4f} conf={7:.4f} rate={8:.2f}".format(
                index,
                item.name,
                item.mean_score,
                item.mean_count_score,
                item.mean_region_score,
                item.mean_ordering_score,
                item.mean_split_score,
                item.mean_confidence,
                item.detection_rate,
            )
        )
        lines.append("   prompt={0}".format(item.prompt))
        lines.append("   bboxes={0} labels={1} frame_id={2}".format(item.selected_bbox_xywh_list, item.selected_label_texts, item.selected_frame_id))
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _print_top_results(summaries: list[CandidateRunSummary]) -> None:
    top_items = summaries[:5]
    for index, item in enumerate(top_items, start=1):
        print(
            "{0}. {1} mean_score={2:.4f} count={3:.4f} region={4:.4f} order={5:.4f} split={6:.4f} conf={7:.4f} rate={8:.2f} bboxes={9}".format(
                index,
                item.name,
                item.mean_score,
                item.mean_count_score,
                item.mean_region_score,
                item.mean_ordering_score,
                item.mean_split_score,
                item.mean_confidence,
                item.detection_rate,
                item.selected_bbox_xywh_list,
            )
        )

# endregion


# region 评分与可视化
def _compute_candidate_frame_score(
    frame_id: int,
    detections: list[TrayDetection],
    target_box_xyxy: tuple[int, int, int, int],
    expected_tray_count: int,
) -> CandidateFrameScore:
    target_x1, target_y1, target_x2, target_y2 = target_box_xyxy
    bbox_list = [_mask_bbox_xywh(np.asarray(det.mask, dtype=np.uint8)) for det in detections]
    label_texts = [str(det.label_text) for det in detections]
    confidences = [float(det.confidence_2d) for det in detections]
    centers = [(_mask_center_uv(np.asarray(det.mask, dtype=np.uint8))) for det in detections]
    tray_count = len(bbox_list)
    count_score = max(0.0, 1.0 - abs(tray_count - int(expected_tray_count)) / max(1.0, float(expected_tray_count)))
    inside_scores: list[float] = []
    for center_x, center_y in centers:
        inside_scores.append(1.0 if (target_x1 <= center_x <= target_x2 and target_y1 <= center_y <= target_y2) else 0.0)
    region_score = float(np.mean(inside_scores)) if len(inside_scores) > 0 else 0.0
    ordering_score = 0.0
    split_score = 0.0
    if tray_count >= 2:
        xs = [item[0] for item in centers]
        ordering_score = 1.0 if xs == sorted(xs) else 0.0
        normalized_gap = float(xs[-1] - xs[0]) / max(1.0, float(target_x2 - target_x1))
        split_score = float(np.clip(normalized_gap / 0.35, 0.0, 1.0))
    elif tray_count == 1:
        box_x, _box_y, box_w, _box_h = bbox_list[0]
        width_ratio = float(box_w) / max(1.0, float(target_x2 - target_x1))
        split_score = float(max(0.0, 1.0 - width_ratio))
        ordering_score = 0.0
    mean_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    score = (
        0.35 * count_score
        + 0.25 * region_score
        + 0.25 * split_score
        + 0.10 * ordering_score
        + 0.05 * mean_confidence
    )
    if tray_count < int(expected_tray_count):
        score -= 0.20
    if tray_count == 1 and len(bbox_list) > 0:
        only_box = bbox_list[0]
        width_ratio = float(only_box[2]) / max(1.0, float(target_x2 - target_x1))
        if width_ratio > 0.55:
            score -= 0.15
    score = float(np.clip(score, 0.0, 1.0))
    return CandidateFrameScore(
        frame_id=int(frame_id),
        score=score,
        count_score=float(count_score),
        region_score=float(region_score),
        ordering_score=float(ordering_score),
        split_score=float(split_score),
        confidence=float(mean_confidence),
        has_detection=True,
        tray_count=int(tray_count),
        bbox_xywh_list=[
            (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
            for item in bbox_list
        ],
        label_texts=label_texts,
    )


def _draw_target_box(image_bgr: np.ndarray, target_box_xyxy: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = target_box_xyxy
    cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, "target", (int(x1), max(16, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size == 0:
        return 0, 0, 0, 0
    x1 = int(np.min(xs))
    x2 = int(np.max(xs))
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _mask_center_uv(mask: np.ndarray) -> tuple[float, float]:
    x, y, w, h = _mask_bbox_xywh(mask)
    return float(x + 0.5 * w), float(y + 0.5 * h)


def _intersection_area_xyxy(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    x1 = max(int(box_a[0]), int(box_b[0]))
    y1 = max(int(box_a[1]), int(box_b[1]))
    x2 = min(int(box_a[2]), int(box_b[2]))
    y2 = min(int(box_a[3]), int(box_b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def _box_iou_xyxy(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    inter = float(_intersection_area_xyxy(box_a, box_b))
    area_a = float(max(0, int(box_a[2]) - int(box_a[0])) * max(0, int(box_a[3]) - int(box_a[1])))
    area_b = float(max(0, int(box_b[2]) - int(box_b[0])) * max(0, int(box_b[3]) - int(box_b[1])))
    denom = max(1.0, area_a + area_b - inter)
    return inter / denom


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _blend_mask_overlay(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    mask_bool = mask_u8 > 0
    if not np.any(mask_bool):
        return base_bgr
    result = np.asarray(base_bgr, dtype=np.float32).copy()
    color = np.asarray(color_bgr, dtype=np.float32)
    result[mask_bool] = result[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _debug_color_bgr(index: int) -> tuple[int, int, int]:
    palette = (
        (0, 220, 255),
        (80, 200, 120),
        (255, 170, 0),
        (255, 110, 180),
        (140, 180, 255),
    )
    return palette[int(index) % len(palette)]


def _summary_to_json(item: CandidateRunSummary) -> dict:
    data = asdict(item)
    data["frame_scores"] = [asdict(score) for score in item.frame_scores]
    return data
# endregion


# region 候选配置
def _build_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="baseline_fast_prompt",
            prompt="black tray,black pallet,rectangular black tray",
            target_keywords="rectangular black tray,black tray,black pallet",
            min_confidence=0.18,
            topk_objects=6,
            combine_prompts_forward=True,
            detect_max_side=512,
            max_targets=3,
            box_threshold=0.12,
            text_threshold=0.06,
        ),
        CandidateSpec(
            name="separate_prompts_highres",
            prompt="black tray,black pallet,rectangular black tray",
            target_keywords="rectangular black tray,black tray,black pallet",
            min_confidence=0.22,
            topk_objects=8,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.12,
            text_threshold=0.05,
        ),
        CandidateSpec(
            name="plastic_rectangular_tray",
            prompt="rectangular black tray,black plastic tray,black pallet,black tray",
            target_keywords="rectangular black tray,black plastic tray,black tray,black pallet",
            min_confidence=0.22,
            topk_objects=8,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.12,
            text_threshold=0.05,
        ),
        CandidateSpec(
            name="rectangular_container_focus",
            prompt="rectangular black tray,black plastic container,black tray,black pallet",
            target_keywords="rectangular black tray,black plastic container,black tray,black pallet",
            min_confidence=0.20,
            topk_objects=8,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.11,
            text_threshold=0.05,
        ),
        CandidateSpec(
            name="tray_bin_mix",
            prompt="rectangular black tray,black bin tray,black plastic tray,black pallet",
            target_keywords="rectangular black tray,black bin tray,black plastic tray,black pallet",
            min_confidence=0.20,
            topk_objects=8,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.11,
            text_threshold=0.05,
        ),
        CandidateSpec(
            name="tray_only_strict",
            prompt="rectangular black tray,black tray,black plastic tray",
            target_keywords="rectangular black tray,black tray,black plastic tray",
            min_confidence=0.18,
            topk_objects=8,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.10,
            text_threshold=0.05,
        ),
        CandidateSpec(
            name="pallet_only_strict",
            prompt="black pallet,rectangular black pallet,black tray pallet",
            target_keywords="black pallet,rectangular black pallet,black tray pallet",
            min_confidence=0.18,
            topk_objects=8,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.10,
            text_threshold=0.05,
        ),
        CandidateSpec(
            name="wide_prompt_lowconf",
            prompt="rectangular black tray,black tray,black plastic tray,black pallet,black bin tray",
            target_keywords="rectangular black tray,black tray,black plastic tray,black pallet,black bin tray",
            min_confidence=0.16,
            topk_objects=10,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.09,
            text_threshold=0.04,
        ),
        CandidateSpec(
            name="tray_dup_prompt",
            prompt="black tray,rectangular black tray,black tray,rectangular black tray",
            target_keywords="rectangular black tray,black tray",
            min_confidence=0.16,
            topk_objects=10,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.09,
            text_threshold=0.04,
        ),
        CandidateSpec(
            name="tray_pallet_dup_prompt",
            prompt="black tray,rectangular black tray,black pallet,black tray,rectangular black tray,black pallet",
            target_keywords="rectangular black tray,black tray,black pallet",
            min_confidence=0.16,
            topk_objects=10,
            combine_prompts_forward=False,
            detect_max_side=640,
            max_targets=3,
            box_threshold=0.09,
            text_threshold=0.04,
        ),
    ]
# endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于目标框的托盘检测参数筛选")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-frame-count", type=int, default=DEFAULT_SAMPLE_FRAME_COUNT)
    parser.add_argument("--target-x1", type=int, default=DEFAULT_TARGET_BOX_XYXY[0])
    parser.add_argument("--target-y1", type=int, default=DEFAULT_TARGET_BOX_XYXY[1])
    parser.add_argument("--target-x2", type=int, default=DEFAULT_TARGET_BOX_XYXY[2])
    parser.add_argument("--target-y2", type=int, default=DEFAULT_TARGET_BOX_XYXY[3])
    parser.add_argument("--expected-tray-count", type=int, default=DEFAULT_EXPECTED_TRAY_COUNT)
    args = parser.parse_args()
    raise SystemExit(
        main(
            output_dir=Path(args.output_dir),
            sample_frame_count=int(args.sample_frame_count),
            target_box_xyxy=(int(args.target_x1), int(args.target_y1), int(args.target_x2), int(args.target_y2)),
            expected_tray_count=int(args.expected_tray_count),
        )
    )
