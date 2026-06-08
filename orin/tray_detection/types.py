from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrayDetectionConfig:
    gd_model_id: str = "IDEA-Research/grounding-dino-base"
    sam_model_id: str = "facebook/sam-vit-base"
    hf_cache_dir: str | None = None
    hf_local_files_only: bool = True
    device: str = "cuda:0"
    proxy_url: str = "http://127.0.0.1:4444"
    prompt: str = "black tray,black pallet,rectangular black tray"
    target_keywords: str = "rectangular black tray,black tray,black pallet"
    strict_target_filter: bool = True
    max_targets: int = 1
    use_sam: bool = False
    box_threshold: float = 0.16
    text_threshold: float = 0.08
    min_confidence: float = 0.20
    topk_objects: int = 2
    sam_max_boxes: int = 1
    sam_primary_only: bool = True
    sam_secondary_conf_threshold: float = 0.55
    combine_prompts_forward: bool = True
    min_mask_pixels: int = 300
    mask_iou_suppress: float = 0.65
    detect_max_side: int = 384


@dataclass(frozen=True)
class TrayDetection:
    label_text: str
    confidence_2d: float
    contour: np.ndarray
    mask: np.ndarray
    excluded_points: int = 0


@dataclass(frozen=True)
class TrayPipelineConfig:
    detect_every_n: int = 6
    motion_downsample: float = 0.25
    motion_smooth_alpha: float = 0.60
    motion_max_shift_px: float = 36.0
    fast_gray_percentile: float = 48.0
    fast_top_crop_ratio: float = 0.36


@dataclass
class TrayRuntimeState:
    cached_detections: list[TrayDetection] = field(default_factory=list)
    cached_ok: bool = False
    compute_count: int = 0
    detect_inflight: bool = False
    lock: Any = field(default=None)
    prev_motion_gray_small: np.ndarray | None = None
    motion_dx_smooth: float = 0.0
    motion_dy_smooth: float = 0.0

    def __post_init__(self) -> None:
        if self.lock is None:
            import threading

            self.lock = threading.Lock()
