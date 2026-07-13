from __future__ import annotations

import threading
from _thread import LockType
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _default_hf_cache_dir() -> str:
    """返回仓库内统一 Hugging Face 缓存目录。"""

    return str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface")


@dataclass(frozen=True, slots=True)
class TrayDetectionConfig:
    """托盘检测模型、阈值、设备和缓存配置。"""

    gd_model_id: str = "IDEA-Research/grounding-dino-base"
    sam_model_id: str = "facebook/sam-vit-base"
    hf_cache_dir: str = field(default_factory=_default_hf_cache_dir)
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


@dataclass(frozen=True, slots=True)
class TrayDetection:
    """单个托盘的内部轮廓、mask 和置信度结果。"""

    label_text: str
    confidence_2d: float
    contour: np.ndarray
    mask: np.ndarray
    excluded_points: int = 0


@dataclass(frozen=True, slots=True)
class TrayPipelineConfig:
    """完整检测与快速帧更新的运行参数。"""

    detect_every_n: int = 6
    motion_downsample: float = 0.25
    motion_smooth_alpha: float = 0.60
    motion_max_shift_px: float = 36.0
    fast_gray_percentile: float = 48.0
    fast_top_crop_ratio: float = 0.36


@dataclass(slots=True)
class TrayRuntimeState:
    """托盘跨帧缓存、计数器和运动估计状态。"""

    cached_detections: list[TrayDetection] = field(default_factory=list)
    cached_ok: bool = False
    compute_count: int = 0
    detect_inflight: bool = False
    lock: LockType = field(default_factory=threading.Lock)
    prev_motion_gray_small: np.ndarray | None = None
    motion_dx_smooth: float = 0.0
    motion_dy_smooth: float = 0.0
