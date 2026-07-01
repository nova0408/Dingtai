from __future__ import annotations
# pyright: reportMissingImports=false

import contextlib
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

from .model_cache import apply_download_proxy, load_pretrained_with_project_cache, prepare_hf_cache_dir
from .types import TrayDetection, TrayDetectionConfig
from .utils import (
    build_combined_prompt,
    build_rect_mask,
    mask_to_contour,
    merge_label_text,
    normalize_label_list,
    parse_keywords,
    resize_for_detection,
    scale_box_xyxy,
    suppress_masks_by_iou,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE_DIR = str(PROJECT_ROOT / ".cache" / "huggingface")


class TrayPointExcluder:
    def __init__(self, config: TrayDetectionConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.prompt = str(config.prompt).strip()
        self.prompt_terms = parse_keywords(config.prompt)
        if len(self.prompt_terms) == 0:
            self.prompt_terms = [self.prompt if len(self.prompt) > 0 else "object"]
        self.combined_prompt = build_combined_prompt(self.prompt_terms)
        self.target_keywords = parse_keywords(config.target_keywords)
        self.strict_target_filter = bool(config.strict_target_filter)
        self.max_targets = int(max(1, config.max_targets))
        self.use_sam = bool(config.use_sam)
        self.box_threshold = float(config.box_threshold)
        self.text_threshold = float(config.text_threshold)
        self.min_target_conf = float(np.clip(config.min_confidence, 0.0, 1.0))
        self.topk_objects = int(max(1, config.topk_objects))
        self.sam_max_boxes = int(max(1, config.sam_max_boxes))
        self.sam_primary_only = bool(config.sam_primary_only)
        self.sam_secondary_conf_threshold = float(np.clip(config.sam_secondary_conf_threshold, 0.0, 1.0))
        self.combine_prompts_forward = bool(config.combine_prompts_forward)
        self.min_mask_pixels = int(max(20, config.min_mask_pixels))
        self.mask_iou_suppress = float(np.clip(config.mask_iou_suppress, 0.1, 0.95))
        self.detect_max_side = int(max(128, config.detect_max_side))
        cache_dir = config.hf_cache_dir if config.hf_cache_dir is not None else DEFAULT_HF_CACHE_DIR
        self.hf_cache_dir = prepare_hf_cache_dir(cache_dir)
        apply_download_proxy(str(config.proxy_url).strip())
        self.gd_processor = load_pretrained_with_project_cache(
            loader=AutoProcessor.from_pretrained,
            model_id=config.gd_model_id,
            cache_dir=self.hf_cache_dir,
            local_files_only=config.hf_local_files_only,
            role="grounding_dino_processor",
        )
        self.gd_model = load_pretrained_with_project_cache(
            loader=AutoModelForZeroShotObjectDetection.from_pretrained,
            model_id=config.gd_model_id,
            cache_dir=self.hf_cache_dir,
            local_files_only=config.hf_local_files_only,
            role="grounding_dino_model",
        ).to(self.device)
        self.gd_model.eval()
        self._gd_postprocess_uses_box_threshold = "box_threshold" in inspect.signature(
            self.gd_processor.post_process_grounded_object_detection
        ).parameters
        self._use_cuda = str(self.device).startswith("cuda")
        self.sam_processor = None
        self.sam_model = None
        if self.use_sam:
            self.sam_processor = load_pretrained_with_project_cache(
                loader=SamProcessor.from_pretrained,
                model_id=config.sam_model_id,
                cache_dir=self.hf_cache_dir,
                local_files_only=config.hf_local_files_only,
                role="sam_processor",
            )
            self.sam_model = load_pretrained_with_project_cache(
                loader=SamModel.from_pretrained,
                model_id=config.sam_model_id,
                cache_dir=self.hf_cache_dir,
                local_files_only=config.hf_local_files_only,
                role="sam_model",
            ).to(self.device)
            self.sam_model.eval()

    def detect(self, frame_bgr: np.ndarray) -> list[TrayDetection]:
        h, w = frame_bgr.shape[:2]
        det_bgr, inv_scale = resize_for_detection(frame_bgr=frame_bgr, detect_max_side=self.detect_max_side)
        det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
        pil_det = Image.fromarray(det_rgb)
        box_items = self._detect_prompt_boxes(pil_img=pil_det)
        if len(box_items) == 0:
            return []
        filtered_items = self._filter_target_boxes(box_items)
        if len(filtered_items) == 0:
            return []
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        if self.use_sam:
            filtered_items = filtered_items[: self.sam_max_boxes]
            full_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            full_pil = None
        candidates: list[tuple[np.ndarray, float, str, np.ndarray, int]] = []
        for idx, (box_det, score, merged_label) in enumerate(filtered_items):
            box_full = scale_box_xyxy(box_xyxy=box_det, scale=inv_scale, w=w, h=h)
            mask = self._build_candidate_mask(idx, float(score), full_pil, box_full, h, w)
            if mask is None:
                continue
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            area = int(np.count_nonzero(mask))
            if area < self.min_mask_pixels:
                continue
            contour = mask_to_contour(mask=mask, min_mask_pixels=self.min_mask_pixels)
            if contour.shape[0] < 3:
                continue
            candidates.append((mask, float(score), str(merged_label), contour, area))
        kept = suppress_masks_by_iou(candidates, self.mask_iou_suppress, self.topk_objects)
        return [
            TrayDetection(label_text=label_text, confidence_2d=float(score), contour=contour, mask=mask, excluded_points=0)
            for mask, score, label_text, contour, _area in kept[: self.max_targets]
        ]

    def _detect_prompt_boxes(self, pil_img: Image.Image) -> list[tuple[np.ndarray, float, str]]:
        if self.combine_prompts_forward:
            box_items = self._grounding_detect_topk(pil_img=pil_img, prompt_text=self.combined_prompt, topk=self.topk_objects)
            return [(b, s, merge_label_text(raw_label=lb, prompt_term="")) for b, s, lb in box_items]
        per_prompt_topk = int(max(1, min(self.topk_objects, max(2, self.max_targets))))
        box_items: list[tuple[np.ndarray, float, str]] = []
        for term in self.prompt_terms:
            term_items = self._grounding_detect_topk(pil_img=pil_img, prompt_text=term, topk=per_prompt_topk)
            for box_det, score, label_text in term_items:
                box_items.append((box_det, float(score), merge_label_text(raw_label=label_text, prompt_term=term)))
        box_items.sort(key=lambda x: x[1], reverse=True)
        return box_items[: self.topk_objects * 3]

    def _filter_target_boxes(self, box_items: list[tuple[np.ndarray, float, str]]) -> list[tuple[np.ndarray, float, str]]:
        filtered_items: list[tuple[np.ndarray, float, str]] = []
        for box_det, score, merged_label in box_items:
            if float(score) < self.min_target_conf:
                continue
            is_target = self._is_target_label(merged_label)
            if self.strict_target_filter and not is_target:
                continue
            filtered_items.append((box_det, float(score), str(merged_label)))
        return filtered_items

    def _grounding_detect_topk(self, pil_img: Image.Image, prompt_text: str, topk: int) -> list[tuple[np.ndarray, float, str]]:
        text = prompt_text if str(prompt_text).endswith(".") else f"{prompt_text}."
        inputs = self.gd_processor(images=pil_img, text=text, return_tensors="pt")
        inputs = _tensor_dict_to_device(inputs, self.device)
        with torch.inference_mode():
            with self._autocast_ctx():
                outputs = self.gd_model(**inputs)
        target_sizes = [(pil_img.height, pil_img.width)]
        postprocess_kwargs = {
            "outputs": outputs,
            "input_ids": inputs["input_ids"],
            "text_threshold": self.text_threshold,
            "target_sizes": target_sizes,
        }
        if self._gd_postprocess_uses_box_threshold:
            postprocess_kwargs["box_threshold"] = self.box_threshold
        else:
            postprocess_kwargs["threshold"] = self.box_threshold
        post = self.gd_processor.post_process_grounded_object_detection(**postprocess_kwargs)
        if len(post) == 0:
            return []
        p0 = post[0]
        boxes = p0.get("boxes", None)
        scores = p0.get("scores", None)
        labels = p0.get("labels", None)
        if boxes is None or scores is None or len(scores) == 0:
            return []
        if hasattr(scores, "detach"):
            score_arr = scores.detach().cpu().numpy().astype(np.float32)
            box_arr = boxes.detach().cpu().numpy().astype(np.float32)
        else:
            score_arr = np.asarray(scores, dtype=np.float32)
            box_arr = np.asarray(boxes, dtype=np.float32)
        label_arr = normalize_label_list(labels=labels, expect_len=len(score_arr))
        order = np.argsort(-score_arr)
        keep = order[: int(max(1, topk))]
        return [(box_arr[int(i)], float(score_arr[int(i)]), label_arr[int(i)]) for i in keep]

    def _autocast_ctx(self):
        if self._use_cuda:
            return torch.autocast(device_type="cuda")
        return contextlib.nullcontext()

    def _is_target_label(self, label_text: str) -> bool:
        if len(self.target_keywords) == 0:
            return True
        t = str(label_text).strip().lower()
        if len(t) == 0:
            return not self.strict_target_filter
        return any(kw in t for kw in self.target_keywords)

    def _build_candidate_mask(
        self,
        candidate_index: int,
        score: float,
        pil_img: Image.Image | None,
        box_xyxy: np.ndarray,
        out_h: int,
        out_w: int,
    ) -> np.ndarray | None:
        use_sam_for_this = False
        if self.use_sam:
            if not self.sam_primary_only:
                use_sam_for_this = True
            else:
                use_sam_for_this = int(candidate_index) == 0 or float(score) >= self.sam_secondary_conf_threshold
        if use_sam_for_this:
            return self._sam_segment_box(pil_img=pil_img, box_xyxy=box_xyxy, out_h=out_h, out_w=out_w)
        return build_rect_mask(box_xyxy=box_xyxy, h=out_h, w=out_w)

    def _sam_segment_box(self, pil_img: Image.Image | None, box_xyxy: np.ndarray, out_h: int, out_w: int) -> np.ndarray | None:
        if pil_img is None or self.sam_processor is None or self.sam_model is None:
            return None
        box = np.asarray(box_xyxy, dtype=np.float32).reshape(1, 4)
        sam_inputs = self.sam_processor(images=pil_img, input_boxes=[[box[0].tolist()]], return_tensors="pt")
        sam_inputs = _tensor_dict_to_device(sam_inputs, self.device)
        with torch.inference_mode():
            with self._autocast_ctx():
                outputs = self.sam_model(**sam_inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu(),
        )
        if len(masks) == 0:
            return None
        mask = masks[0][0][0].numpy().astype(np.float32)
        mask_u8 = (mask > 0.0).astype(np.uint8) * 255
        if mask_u8.shape[0] != out_h or mask_u8.shape[1] != out_w:
            mask_u8 = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return mask_u8


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。")
    return "cpu" if d == "cpu" else d


def _tensor_dict_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
