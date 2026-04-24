from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch  # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
from loguru import logger
from PIL import Image
from pyorbbecsdk import OBFormat
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import (
    OrbbecSession,
    SessionOptions,
    filter_valid_points,
    normalize_points,
    set_point_cloud_filter_format,
)

# region 默认参数（优先在这里直接改）
DEFAULT_GD_MODEL_ID = "IDEA-Research/grounding-dino-base"  # GroundingDINO 模型 ID（HuggingFace）
DEFAULT_SAM_MODEL_ID = "facebook/sam-vit-base"  # SAM 模型 ID（HuggingFace）
DEFAULT_DEVICE = "cuda:0"  # 推理设备，建议 GPU：cuda:0
DEFAULT_PROXY_URL = "http://127.0.0.1:4444"  # 模型下载代理地址，留空则不注入
# autocorrect:false
DEFAULT_PROMPT = "black tray,black pallet,rectangular black tray"  # 零训练目标提示词（逗号分隔，可多词并行）
DEFAULT_TARGET_KEYWORDS = "rectangular black tray,black tray,black pallet"  # 目标关键词（逗号分隔）
# autocorrect:true
DEFAULT_STRICT_TARGET_FILTER = True  # 是否严格按关键词过滤（True 时非目标一律丢弃）
DEFAULT_MAX_TARGETS = 2  # 最多保留目标数量（料盘场景通常为 2）
DEFAULT_USE_SAM = True  # 是否启用 SAM 精细分割
DEFAULT_BOX_THRESHOLD = 0.16  # GroundingDINO 框阈值（召回优先）
DEFAULT_TEXT_THRESHOLD = 0.08  # GroundingDINO 文本阈值（召回优先）
DEFAULT_MIN_TARGET_CONF = 0.35  # 目标最小置信度阈值（稳定优先，避免边缘分数频繁丢检）
DEFAULT_TOPK_OBJECTS = 4  # 每帧最多保留目标数
DEFAULT_SAM_MAX_BOXES = 2  # 每帧最多进入 SAM 的候选框数量
DEFAULT_MIN_MASK_PIXELS = 300  # 2D 掩码最小像素面积
DEFAULT_MASK_IOU_SUPPRESS = 0.65  # 掩码抑制阈值（避免重复目标）
DEFAULT_DETECT_MAX_SIDE = 512  # 2D 检测最长边缩放尺寸，越小越快
DEFAULT_DETECT_INTERVAL = 4  # 每 N 帧做一次检测，其余帧复用最近结果
DEFAULT_ENABLE_ADAPTIVE_INTERVAL = True  # 是否启用自适应检测频率（稳态提速、失稳增密）
DEFAULT_ADAPTIVE_INTERVAL_MAX = 6  # 自适应检测频率上限（越大越省算力）
DEFAULT_ADAPTIVE_CONF_HIGH = 0.43  # 高置信阈值：高于该值倾向拉长检测间隔
DEFAULT_ADAPTIVE_CONF_LOW = 0.33  # 低置信阈值：低于该值倾向缩短检测间隔
DEFAULT_DEBUG_LOG_EVERY_DETECT = 5  # 每 N 次检测打印一次详细候选日志
DEFAULT_CACHE_HOLD_DETECT_ROUNDS = 5  # 连续空检测时保留历史结果的检测轮数（抗抖动）
DEFAULT_COMBINE_PROMPTS_FORWARD = False  # True=单次合并前向(更快), False=逐关键词前向(更准)

DEFAULT_TIMEOUT_MS = 120  # 等待帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 5000.0  # 深度裁剪上限，单位 mm
DEFAULT_MAX_PREVIEW_POINTS = 90_000  # 预览点上限
DEFAULT_MIN_OBJECT_POINTS = 60  # 3D 显示最小点数

DEFAULT_ALPHA = 0.42  # 2D/3D叠加半透明权重
DEFAULT_WINDOW_WIDTH = 1440  # 3D 窗口宽度
DEFAULT_WINDOW_HEIGHT = 900  # 3D 窗口高度
DEFAULT_POINT_SIZE = 1.5  # 3D 点大小
DEFAULT_BACKGROUND_COLOR = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)  # 3D 背景色
DEFAULT_BASE_COLOR = np.asarray([0.22, 0.22, 0.22], dtype=np.float64)  # 基础点云颜色
DEFAULT_2D_WINDOW_NAME = "Orbbec object partition detector"  # 2D 窗口名
# endregion


# region 数据结构
@dataclass
class Detection2D:
    det_id: int
    confidence_2d: float
    label_text: str
    contour: np.ndarray
    mask: np.ndarray
    area_pixels: int


# endregion


# region 零训练检测器
class ZeroShotObjectPartitionDetector:
    def __init__(
        self,
        gd_model_id: str,
        sam_model_id: str,
        device: str,
        proxy_url: str,
        prompt: str,
        target_keywords: str,
        strict_target_filter: bool,
        max_targets: int,
        use_sam: bool,
        box_threshold: float,
        text_threshold: float,
        min_target_conf: float,
        topk_objects: int,
        sam_max_boxes: int,
        combine_prompts_forward: bool,
        min_mask_pixels: int,
        mask_iou_suppress: float,
        detect_max_side: int,
    ) -> None:
        if torch is None or Image is None or AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            raise RuntimeError("缺少零训练依赖。请安装：pip install transformers pillow accelerate")
        if bool(use_sam) and (SamProcessor is None or SamModel is None):
            raise RuntimeError("当前环境缺少 SAM 组件，请安装匹配版本或加参数 --no-use-sam")

        self.device = self._resolve_device(device)
        self.prompt = str(prompt).strip()
        self.prompt_terms = _parse_keywords(prompt)
        if len(self.prompt_terms) == 0:
            self.prompt_terms = [self.prompt if len(self.prompt) > 0 else "object"]
        self.combined_prompt = _build_combined_prompt(self.prompt_terms)
        self.target_keywords = _parse_keywords(target_keywords)
        self.strict_target_filter = bool(strict_target_filter)
        self.max_targets = int(max(1, max_targets))
        self.use_sam = bool(use_sam)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.min_target_conf = float(np.clip(min_target_conf, 0.0, 1.0))
        self.topk_objects = int(max(1, topk_objects))
        self.sam_max_boxes = int(max(1, sam_max_boxes))
        self.combine_prompts_forward = bool(combine_prompts_forward)
        self.min_mask_pixels = int(max(20, min_mask_pixels))
        self.mask_iou_suppress = float(np.clip(mask_iou_suppress, 0.1, 0.95))
        self.detect_max_side = int(max(128, detect_max_side))
        self.detect_count = 0

        _apply_download_proxy(str(proxy_url).strip())

        logger.info(f"加载 GroundingDINO：{gd_model_id}")
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(self.device)
        self.gd_model.eval()
        self._use_cuda = str(self.device).startswith("cuda")
        if self._use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam_processor = None
        self.sam_model = None
        if self.use_sam:
            logger.info(f"加载 SAM：{sam_model_id}")
            self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
            self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
            self.sam_model.eval()
        else:
            logger.info("跳过 SAM：使用检测框掩码（更快）")

        logger.info(f"零训练划分检测器就绪：device {self.device}")

    def _resolve_device(self, device: str) -> torch.device:
        d = str(device).strip().lower()
        if d.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。")
        if d == "cpu":
            return torch.device("cpu")
        return torch.device(d)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]:
        self.detect_count += 1
        h, w = frame_bgr.shape[:2]
        det_bgr, inv_scale = _resize_for_detection(frame_bgr=frame_bgr, detect_max_side=self.detect_max_side)
        det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
        pil_det = Image.fromarray(det_rgb)
        box_items: list[tuple[np.ndarray, float, str]] = []
        if self.combine_prompts_forward:
            box_items = self._grounding_detect_topk(
                pil_img=pil_det,
                prompt_text=self.combined_prompt,
                topk=self.topk_objects,
            )
            box_items = [(b, s, _merge_label_text(raw_label=lb, prompt_term="")) for b, s, lb in box_items]
        else:
            # 稳定优先：逐关键词识别，但每词限制候选数量，控制推理开销。
            per_prompt_topk = int(max(1, min(self.topk_objects, max(2, self.max_targets))))
            for term in self.prompt_terms:
                term_items = self._grounding_detect_topk(
                    pil_img=pil_det,
                    prompt_text=term,
                    topk=per_prompt_topk,
                )
                for box_det, score, label_text in term_items:
                    box_items.append((box_det, float(score), _merge_label_text(raw_label=label_text, prompt_term=term)))
        if len(box_items) == 0:
            return []
        box_items.sort(key=lambda x: x[1], reverse=True)
        if len(box_items) > self.topk_objects * 3:
            box_items = box_items[: self.topk_objects * 3]
        if self.detect_count % DEFAULT_DEBUG_LOG_EVERY_DETECT == 1:
            preview = " | ".join(
                [f"label={str(lbl)}:{float(sc):.2f}" for _, sc, lbl in box_items[: min(8, len(box_items))]]
            )
            logger.info(f"检测候选标签（未过滤）：{preview}")

        candidates: list[tuple[np.ndarray, float, str, np.ndarray, int]] = []
        full_pil = None
        filtered_items: list[tuple[np.ndarray, float, str]] = []
        for box_det, score, merged_label in box_items:
            if float(score) < self.min_target_conf:
                continue
            is_target = self._is_target_label(merged_label)
            if self.strict_target_filter and not is_target:
                continue
            filtered_items.append((box_det, float(score), str(merged_label)))

        if len(filtered_items) == 0:
            return []

        filtered_items.sort(key=lambda x: x[1], reverse=True)
        if self.use_sam:
            filtered_items = filtered_items[: self.sam_max_boxes]
            full_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        for box_det, score, merged_label in filtered_items:
            box_full = _scale_box_xyxy(box_xyxy=box_det, scale=inv_scale, w=w, h=h)
            if self.use_sam:
                mask = self._sam_segment_box(pil_img=full_pil, box_xyxy=box_full, out_h=h, out_w=w)
                if mask is None:
                    continue
            else:
                mask = _build_rect_mask(box_xyxy=box_full, h=h, w=w)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
            area = int(np.count_nonzero(mask))
            if area < self.min_mask_pixels:
                continue
            contour = _mask_to_contour(mask)
            if contour.shape[0] < 3:
                continue
            candidates.append((mask, float(score), str(merged_label), contour, area))

        if len(candidates) == 0:
            return []

        # 按置信度排序并做掩码 IoU 抑制，避免一个物体重复分配。
        candidates.sort(key=lambda x: x[1], reverse=True)
        kept: list[tuple[np.ndarray, float, str, np.ndarray, int]] = []
        for cand in candidates:
            keep = True
            for k in kept:
                iou = _mask_iou(cand[0], k[0])
                if iou >= self.mask_iou_suppress:
                    keep = False
                    break
            if keep:
                kept.append(cand)
            if len(kept) >= self.topk_objects:
                break

        dets: list[Detection2D] = []
        for i, (mask, score, label_text, contour, area) in enumerate(kept):
            dets.append(
                Detection2D(
                    det_id=i,
                    confidence_2d=float(score),
                    label_text=label_text,
                    contour=contour,
                    mask=mask,
                    area_pixels=int(area),
                )
            )
        dets.sort(key=lambda d: d.confidence_2d, reverse=True)
        return dets[: self.max_targets]

    def _grounding_detect_topk(
        self, pil_img: Image.Image, prompt_text: str, topk: int
    ) -> list[tuple[np.ndarray, float, str]]:
        text = prompt_text if str(prompt_text).endswith(".") else f"{prompt_text}."
        inputs = self.gd_processor(images=pil_img, text=text, return_tensors="pt")
        inputs = _tensor_dict_to_device(inputs, self.device)

        with torch.inference_mode():
            with self._autocast_ctx():
                outputs = self.gd_model(**inputs)

        target_sizes = [(pil_img.height, pil_img.width)]
        post = self.gd_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )
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
        label_arr = _normalize_label_list(labels=labels, expect_len=len(score_arr))

        order = np.argsort(-score_arr)
        keep = order[: int(max(1, topk))]
        out: list[tuple[np.ndarray, float, str]] = []
        for i in keep:
            ii = int(i)
            out.append((box_arr[ii], float(score_arr[ii]), label_arr[ii]))
        return out

    def _autocast_ctx(self):
        if self._use_cuda:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _is_target_label(self, label_text: str) -> bool:
        if len(self.target_keywords) == 0:
            return True
        t = str(label_text).strip().lower()
        if len(t) == 0:
            return not self.strict_target_filter
        for kw in self.target_keywords:
            if kw in t:
                return True
        return False

    def _sam_segment_box(
        self, pil_img: Image.Image | None, box_xyxy: np.ndarray, out_h: int, out_w: int
    ) -> np.ndarray | None:
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


# endregion


# region 主流程
def main(
    gd_model_id: str = DEFAULT_GD_MODEL_ID,
    sam_model_id: str = DEFAULT_SAM_MODEL_ID,
    device: str = DEFAULT_DEVICE,
    proxy_url: str = DEFAULT_PROXY_URL,
    prompt: str = DEFAULT_PROMPT,
    target_keywords: str = DEFAULT_TARGET_KEYWORDS,
    strict_target_filter: bool = DEFAULT_STRICT_TARGET_FILTER,
    max_targets: int = DEFAULT_MAX_TARGETS,
    use_sam: bool = DEFAULT_USE_SAM,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    min_target_conf: float = DEFAULT_MIN_TARGET_CONF,
    topk_objects: int = DEFAULT_TOPK_OBJECTS,
    sam_max_boxes: int = DEFAULT_SAM_MAX_BOXES,
    min_mask_pixels: int = DEFAULT_MIN_MASK_PIXELS,
    mask_iou_suppress: float = DEFAULT_MASK_IOU_SUPPRESS,
    detect_max_side: int = DEFAULT_DETECT_MAX_SIDE,
    detect_interval: int = DEFAULT_DETECT_INTERVAL,
    enable_adaptive_interval: bool = DEFAULT_ENABLE_ADAPTIVE_INTERVAL,
    adaptive_interval_max: int = DEFAULT_ADAPTIVE_INTERVAL_MAX,
    adaptive_conf_high: float = DEFAULT_ADAPTIVE_CONF_HIGH,
    adaptive_conf_low: float = DEFAULT_ADAPTIVE_CONF_LOW,
    cache_hold_detect_rounds: int = DEFAULT_CACHE_HOLD_DETECT_ROUNDS,
    combine_prompts_forward: bool = DEFAULT_COMBINE_PROMPTS_FORWARD,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    min_object_points: int = DEFAULT_MIN_OBJECT_POINTS,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    detector = ZeroShotObjectPartitionDetector(
        gd_model_id=gd_model_id,
        sam_model_id=sam_model_id,
        device=device,
        proxy_url=proxy_url,
        prompt=prompt,
        target_keywords=target_keywords,
        strict_target_filter=strict_target_filter,
        max_targets=max_targets,
        use_sam=use_sam,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        min_target_conf=min_target_conf,
        topk_objects=topk_objects,
        sam_max_boxes=sam_max_boxes,
        combine_prompts_forward=combine_prompts_forward,
        min_mask_pixels=min_mask_pixels,
        mask_iou_suppress=mask_iou_suppress,
        detect_max_side=detect_max_side,
    )

    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
    with OrbbecSession(options=options) as session:
        cam = session.get_camera_param()
        ci = cam.rgb_intrinsic if session.has_color_sensor else cam.depth_intrinsic
        img_w = int(max(32, ci.width))
        img_h = int(max(32, ci.height))
        fx = float(ci.fx)
        fy = float(ci.fy)
        cx = float(ci.cx)
        cy = float(ci.cy)

        logger.info("启动零训练物体划分（不强制红绿类别）")
        logger.info(
            f"参数：prompt {prompt}, target_keywords {target_keywords}, strict_target_filter {strict_target_filter}, "
            f"max_targets {max_targets}, use_sam {use_sam}, box_threshold {box_threshold:.2f}, text_threshold {text_threshold:.2f}, "
            f"min_target_conf {min_target_conf:.2f}, sam_max_boxes {sam_max_boxes}, "
            f"detect_max_side {detect_max_side}, detect_interval {detect_interval}, "
            f"enable_adaptive_interval {enable_adaptive_interval}, adaptive_interval_max {adaptive_interval_max}, "
            f"adaptive_conf_high {adaptive_conf_high:.2f}, adaptive_conf_low {adaptive_conf_low:.2f}, "
            f"cache_hold_detect_rounds {cache_hold_detect_rounds}, topk_objects {topk_objects}, "
            f"combine_prompts_forward {combine_prompts_forward}"
        )

        _run_loop(
            session=session,
            detector=detector,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_w=img_w,
            img_h=img_h,
            detect_interval=max(1, int(detect_interval)),
            enable_adaptive_interval=bool(enable_adaptive_interval),
            adaptive_interval_max=max(1, int(adaptive_interval_max)),
            adaptive_conf_high=float(np.clip(adaptive_conf_high, 0.0, 1.0)),
            adaptive_conf_low=float(np.clip(adaptive_conf_low, 0.0, 1.0)),
            cache_hold_detect_rounds=max(0, int(cache_hold_detect_rounds)),
            min_object_points=max(1, int(min_object_points)),
            alpha=float(np.clip(alpha, 0.0, 1.0)),
        )


# endregion


# region 实时循环
def _run_loop(
    session: OrbbecSession,
    detector: ZeroShotObjectPartitionDetector,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    detect_interval: int,
    enable_adaptive_interval: bool,
    adaptive_interval_max: int,
    adaptive_conf_high: float,
    adaptive_conf_low: float,
    cache_hold_detect_rounds: int,
    min_object_points: int,
    alpha: float,
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Orbbec 零训练物体划分 3D", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = DEFAULT_POINT_SIZE
        render_opt.background_color = DEFAULT_BACKGROUND_COLOR

    base_pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axis)
    vis.add_geometry(base_pcd)
    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)

    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)
    point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())

    frame_idx = 0
    cached_dets: list[Detection2D] = []
    consecutive_empty_detect_rounds = 0
    current_detect_interval = max(1, int(detect_interval))
    adaptive_interval_cap = max(current_detect_interval, int(adaptive_interval_max))
    detect_countdown = 0
    if adaptive_conf_low > adaptive_conf_high:
        adaptive_conf_low = adaptive_conf_high
    try:
        while True:
            if stop["flag"]:
                break

            points, color_bgr = _capture_preview_with_color_once(session=session, point_filter=point_filter)
            if points is None or len(points) == 0:
                alive = vis.poll_events()
                vis.update_renderer()
                if not alive:
                    break
                continue

            frame_idx += 1
            xyz = np.asarray(points[:, :3], dtype=np.float64)
            rgb = _extract_rgb(points)
            uv, valid_proj = _project_points_to_image(xyz=xyz, fx=fx, fy=fy, cx=cx, cy=cy, w=img_w, h=img_h)
            rgb_img = _rasterize_rgb(xyz=xyz, rgb=rgb, uv=uv, valid_proj=valid_proj, w=img_w, h=img_h)
            base_2d = (
                cv2.resize(color_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                if color_bgr is not None
                else rgb_img
            )

            if detect_countdown <= 0:
                dets_new = detector.detect(base_2d)
                if len(dets_new) > 0:
                    cached_dets = dets_new
                    consecutive_empty_detect_rounds = 0
                    if enable_adaptive_interval:
                        best_conf = max([float(d.confidence_2d) for d in dets_new])
                        if best_conf >= adaptive_conf_high:
                            current_detect_interval = min(adaptive_interval_cap, current_detect_interval + 1)
                        elif best_conf <= adaptive_conf_low:
                            current_detect_interval = max(1, current_detect_interval - 1)
                else:
                    consecutive_empty_detect_rounds += 1
                    if enable_adaptive_interval:
                        current_detect_interval = max(1, current_detect_interval - 1)
                    if consecutive_empty_detect_rounds > cache_hold_detect_rounds:
                        cached_dets = []
                dets = cached_dets
                detect_countdown = max(0, current_detect_interval - 1)
                if frame_idx % (max(1, detect_interval) * DEFAULT_DEBUG_LOG_EVERY_DETECT) == 1:
                    if len(dets) > 0:
                        mapping: list[str] = []
                        for d in dets:
                            bgr = _palette_bgr(d.det_id)
                            mapping.append(
                                f"O{d.det_id}/BGR{bgr}/label={d.label_text}/conf={d.confidence_2d:.2f}/area={d.area_pixels}"
                            )
                        logger.info(
                            f"颜色 - 关键词映射：{' | '.join(mapping)} | detect_interval_now={current_detect_interval}"
                        )
                    else:
                        logger.info(f"颜色 - 关键词映射：当前无有效目标 | detect_interval_now={current_detect_interval}")
            else:
                dets = cached_dets
                detect_countdown -= 1

            labels = np.full((xyz.shape[0],), -1, dtype=np.int32)
            show_dets: list[Detection2D] = []
            for det in dets:
                ids = _collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=det.mask)
                if ids.size < min_object_points:
                    continue
                labels[ids] = int(det.det_id)
                show_dets.append(det)

            _update_3d_cloud(base_pcd=base_pcd, xyz=xyz, labels=labels, alpha=alpha)
            vis.update_geometry(base_pcd)

            overlay = _draw_2d_overlay(rgb_img=base_2d, dets=show_dets, alpha=alpha)
            cv2.imshow(DEFAULT_2D_WINDOW_NAME, overlay)
            key = cv2.waitKey(1)
            if key == 27:
                stop["flag"] = True

            if frame_idx % 15 == 0:
                obj_cnt = len(show_dets)
                assigned = int(np.sum(labels >= 0))
                logger.info(f"帧 {frame_idx}：objects {obj_cnt}, assigned_points {assigned}")

            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
            if cv2.getWindowProperty(DEFAULT_2D_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        vis.destroy_window()
        cv2.destroyWindow(DEFAULT_2D_WINDOW_NAME)


# endregion


# region 工具函数
def _tensor_dict_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _apply_download_proxy(proxy_url: str) -> None:
    px = str(proxy_url).strip()
    if len(px) == 0:
        logger.info("未注入下载代理（proxy_url 为空）")
        return
    os.environ["HTTP_PROXY"] = px
    os.environ["HTTPS_PROXY"] = px
    os.environ["ALL_PROXY"] = px
    logger.info(f"已注入下载代理：{px}")


def _parse_keywords(text: str) -> list[str]:
    raw = str(text).replace("，", ",").split(",")
    out: list[str] = []
    for s in raw:
        v = s.strip().lower()
        if len(v) > 0:
            out.append(v)
    return out


def _build_combined_prompt(terms: list[str]) -> str:
    clean = [str(t).strip() for t in terms if len(str(t).strip()) > 0]
    if len(clean) == 0:
        return "object"
    return ". ".join(clean)


def _normalize_label_list(labels, expect_len: int) -> list[str]:
    if labels is None:
        return ["" for _ in range(expect_len)]
    if isinstance(labels, np.ndarray):
        vals = [str(x) for x in labels.tolist()]
    elif isinstance(labels, (list, tuple)):
        vals = [str(x) for x in labels]
    else:
        try:
            vals = [str(x) for x in labels]
        except Exception:
            vals = []
    if len(vals) < expect_len:
        vals.extend([""] * (expect_len - len(vals)))
    return vals[:expect_len]


def _merge_label_text(raw_label: str, prompt_term: str) -> str:
    r = str(raw_label).strip().lower()
    p = str(prompt_term).strip()
    if len(r) == 0:
        return p
    if r in {"object", "objects", "thing", "item"}:
        return p if len(p) > 0 else str(raw_label).strip()
    return str(raw_label).strip()


def _resize_for_detection(frame_bgr: np.ndarray, detect_max_side: int) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= int(detect_max_side):
        return frame_bgr, 1.0
    scale = float(detect_max_side) / float(m)
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    inv_scale = 1.0 / scale
    return resized, inv_scale


def _scale_box_xyxy(box_xyxy: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
    b = np.asarray(box_xyxy, dtype=np.float32).copy()
    b *= float(scale)
    b[0] = np.clip(b[0], 0, max(0, w - 1))
    b[2] = np.clip(b[2], 0, max(0, w - 1))
    b[1] = np.clip(b[1], 0, max(0, h - 1))
    b[3] = np.clip(b[3], 0, max(0, h - 1))
    if b[2] < b[0]:
        b[0], b[2] = b[2], b[0]
    if b[3] < b[1]:
        b[1], b[3] = b[3], b[1]
    return b


def _build_rect_mask(box_xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    x1 = int(np.floor(float(box_xyxy[0])))
    y1 = int(np.floor(float(box_xyxy[1])))
    x2 = int(np.ceil(float(box_xyxy[2])))
    y2 = int(np.ceil(float(box_xyxy[3])))
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    x2 = int(np.clip(x2, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    y2 = int(np.clip(y2, 0, max(0, h - 1)))
    out = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return out
    out[y1 : y2 + 1, x1 : x2 + 1] = 255
    return out


def _mask_to_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.int32)
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < float(DEFAULT_MIN_MASK_PIXELS):
        return np.empty((0, 2), dtype=np.int32)
    hull = cv2.convexHull(c).reshape(-1, 2)
    return hull.astype(np.int32)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _capture_preview_with_color_once(
    session: OrbbecSession, point_filter
) -> tuple[np.ndarray | None, np.ndarray | None]:
    frames = session.wait_for_frames()
    if frames is None:
        return None, None

    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None, None
    color_frame = frames.get_color_frame()
    color_bgr = _decode_color_frame_bgr(color_frame)

    point_frames, use_color = session.prepare_frame_for_point_cloud(frames)
    set_point_cloud_filter_format(
        point_filter,
        depth_scale=float(depth_frame.get_depth_scale()),
        use_color=use_color,
    )
    cloud_frame = point_filter.process(point_frames)
    if cloud_frame is None:
        return None, color_bgr

    raw = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
    normalized = normalize_points(raw)
    valid, _ = filter_valid_points(normalized, max_depth_mm=DEFAULT_MAX_DEPTH_MM)
    if len(valid) == 0:
        return None, color_bgr

    if len(valid) > DEFAULT_MAX_PREVIEW_POINTS:
        step = max(1, len(valid) // DEFAULT_MAX_PREVIEW_POINTS)
        valid = valid[::step]
    return valid, color_bgr


def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)
    return np.full((points.shape[0], 3), 0.7, dtype=np.float32)


def _decode_color_frame_bgr(color_frame) -> np.ndarray | None:
    if color_frame is None:
        return None
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    if width <= 0 or height <= 0:
        return None

    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())
    if data.size == 0:
        return None

    if color_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3)).copy()
    if color_format in (OBFormat.YUYV, OBFormat.YUY2):
        yuy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return None


def _project_points_to_image(
    xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = xyz[:, 2]
    valid = z > 1e-6
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)

    if np.any(valid):
        x = xyz[valid, 0]
        y = xyz[valid, 1]
        zz = z[valid]
        uu = np.rint(fx * x / zz + cx).astype(np.int32)
        vv = np.rint(fy * y / zz + cy).astype(np.int32)
        in_bounds = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        idx = np.where(valid)[0][in_bounds]
        u[idx] = uu[in_bounds]
        v[idx] = vv[in_bounds]

    uv = np.stack([u, v], axis=1)
    return uv, (u >= 0) & (v >= 0)


def _rasterize_rgb(
    xyz: np.ndarray, rgb: np.ndarray, uv: np.ndarray, valid_proj: np.ndarray, w: int, h: int
) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return out

    u = uv[idx, 0].astype(np.int32)
    v = uv[idx, 1].astype(np.int32)
    z = xyz[idx, 2].astype(np.float32)
    linear = v * int(w) + u
    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    idx_sorted = idx[order]
    first = np.unique(linear_sorted, return_index=True)[1]
    chosen = idx_sorted[first]

    u_c = uv[chosen, 0].astype(np.int32)
    v_c = uv[chosen, 1].astype(np.int32)
    out[v_c, u_c, :] = np.clip(rgb[chosen] * 255.0, 0.0, 255.0).astype(np.uint8)[:, ::-1]
    return out


def _collect_indices_in_mask(uv: np.ndarray, valid_proj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return np.empty((0,), dtype=np.int32)
    u = uv[idx, 0]
    v = uv[idx, 1]
    inside = mask[v, u] > 0
    return idx[inside].astype(np.int32)


def _palette_color(idx: int) -> np.ndarray:
    hue = (37 * int(idx)) % 180
    hsv = np.asarray([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0, :]
    return np.asarray([float(bgr[2]) / 255.0, float(bgr[1]) / 255.0, float(bgr[0]) / 255.0], dtype=np.float64)


def _palette_bgr(idx: int) -> tuple[int, int, int]:
    c = _palette_color(idx)
    return (int(c[2] * 255.0), int(c[1] * 255.0), int(c[0] * 255.0))


def _update_3d_cloud(base_pcd: o3d.geometry.PointCloud, xyz: np.ndarray, labels: np.ndarray, alpha: float) -> None:
    base_rgb = np.tile(DEFAULT_BASE_COLOR.reshape(1, 3), (xyz.shape[0], 1))
    uniq = np.unique(labels)
    uniq = uniq[uniq >= 0]
    for obj_id in uniq:
        color = _palette_color(int(obj_id))
        m = labels == int(obj_id)
        if np.any(m):
            base_rgb[m] = (1.0 - alpha) * base_rgb[m] + alpha * color.reshape(1, 3)

    base_pcd.points = o3d.utility.Vector3dVector(xyz)
    base_pcd.colors = o3d.utility.Vector3dVector(np.clip(base_rgb, 0.0, 1.0))


def _draw_2d_overlay(rgb_img: np.ndarray, dets: list[Detection2D], alpha: float) -> np.ndarray:
    overlay = rgb_img.copy()

    for det in dets:
        c = _palette_color(det.det_id)
        bgr = (int(c[2] * 255.0), int(c[1] * 255.0), int(c[0] * 255.0))
        if det.contour.shape[0] >= 3:
            cv2.fillConvexPoly(overlay, det.contour.astype(np.int32), bgr)
            cv2.polylines(overlay, [det.contour.astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
            center = np.mean(det.contour, axis=0).astype(np.int32)
            cv2.putText(
                overlay,
                f"O{det.det_id}:{det.confidence_2d:.2f} {det.label_text}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    blended = cv2.addWeighted(overlay, float(alpha), rgb_img, float(1.0 - alpha), 0.0)
    return blended


# endregion


# region CLI
def _parse_cli() -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    bool,
    int,
    bool,
    float,
    float,
    float,
    int,
    int,
    int,
    float,
    int,
    int,
    int,
    bool,
    int,
    float,
    float,
    bool,
    int,
    int,
    int,
    float,
]:
    parser = argparse.ArgumentParser(description="零训练物体划分（GroundingDINO + 可选 SAM + 3D 投影可视化）")
    parser.add_argument("--gd-model-id", type=str, default=DEFAULT_GD_MODEL_ID, help="GroundingDINO 模型 ID")
    parser.add_argument("--sam-model-id", type=str, default=DEFAULT_SAM_MODEL_ID, help="SAM 模型 ID")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="推理设备，示例 cuda:0 / cpu")
    parser.add_argument("--proxy-url", type=str, default=DEFAULT_PROXY_URL, help="模型下载代理地址，留空表示不注入")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="目标提示词，支持逗号分隔多词，例如 tray,pallet,material tray",
    )
    parser.add_argument(
        "--target-keywords", type=str, default=DEFAULT_TARGET_KEYWORDS, help="仅保留包含这些关键词的目标标签"
    )
    parser.add_argument(
        "--strict-target-filter",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STRICT_TARGET_FILTER,
        help="是否严格按关键词过滤",
    )
    parser.add_argument("--max-targets", type=int, default=DEFAULT_MAX_TARGETS, help="最多保留目标数量")
    parser.add_argument(
        "--use-sam",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_SAM,
        help="是否启用 SAM 精细分割（禁用可明显提速）",
    )
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD, help="GroundingDINO 框阈值")
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD, help="GroundingDINO 文本阈值")
    parser.add_argument("--min-target-conf", type=float, default=DEFAULT_MIN_TARGET_CONF, help="目标最小置信度阈值")
    parser.add_argument("--topk-objects", type=int, default=DEFAULT_TOPK_OBJECTS, help="每帧最多保留目标数")
    parser.add_argument(
        "--sam-max-boxes", type=int, default=DEFAULT_SAM_MAX_BOXES, help="每帧最多进入 SAM 的候选框数量"
    )
    parser.add_argument("--min-mask-pixels", type=int, default=DEFAULT_MIN_MASK_PIXELS, help="掩码最小像素面积")
    parser.add_argument("--mask-iou-suppress", type=float, default=DEFAULT_MASK_IOU_SUPPRESS, help="掩码抑制 IoU 阈值")
    parser.add_argument("--detect-max-side", type=int, default=DEFAULT_DETECT_MAX_SIDE, help="检测缩放最长边")
    parser.add_argument("--detect-interval", type=int, default=DEFAULT_DETECT_INTERVAL, help="每 N 帧检测一次")
    parser.add_argument(
        "--enable-adaptive-interval",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_ADAPTIVE_INTERVAL,
        help="是否启用自适应检测频率",
    )
    parser.add_argument(
        "--adaptive-interval-max",
        type=int,
        default=DEFAULT_ADAPTIVE_INTERVAL_MAX,
        help="自适应检测频率上限（仅在启用自适应时生效）",
    )
    parser.add_argument(
        "--adaptive-conf-high",
        type=float,
        default=DEFAULT_ADAPTIVE_CONF_HIGH,
        help="高置信阈值（高于该阈值时可增大检测间隔）",
    )
    parser.add_argument(
        "--adaptive-conf-low",
        type=float,
        default=DEFAULT_ADAPTIVE_CONF_LOW,
        help="低置信阈值（低于该阈值时减小检测间隔）",
    )
    parser.add_argument(
        "--cache-hold-detect-rounds",
        type=int,
        default=DEFAULT_CACHE_HOLD_DETECT_ROUNDS,
        help="连续空检测时保留历史结果的检测轮数",
    )
    parser.add_argument(
        "--combine-prompts-forward",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COMBINE_PROMPTS_FORWARD,
        help="是否将多个提示词合并为单次前向（更快但可能降稳）",
    )

    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames 超时（ms）")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="请求采集帧率（fps）")
    parser.add_argument("--min-object-points", type=int, default=DEFAULT_MIN_OBJECT_POINTS, help="3D 显示最小点数")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="半透明叠加权重 0~1")
    args = parser.parse_args()

    return (
        str(args.gd_model_id),
        str(args.sam_model_id),
        str(args.device),
        str(args.proxy_url),
        str(args.prompt),
        str(args.target_keywords),
        bool(args.strict_target_filter),
        int(args.max_targets),
        bool(args.use_sam),
        float(args.box_threshold),
        float(args.text_threshold),
        float(args.min_target_conf),
        int(args.topk_objects),
        int(args.sam_max_boxes),
        int(args.min_mask_pixels),
        float(args.mask_iou_suppress),
        int(args.detect_max_side),
        int(args.detect_interval),
        bool(args.enable_adaptive_interval),
        int(args.adaptive_interval_max),
        float(args.adaptive_conf_high),
        float(args.adaptive_conf_low),
        int(args.cache_hold_detect_rounds),
        bool(args.combine_prompts_forward),
        int(args.timeout_ms),
        int(args.capture_fps),
        int(args.min_object_points),
        float(args.alpha),
    )


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            (
                gd_model_id_arg,
                sam_model_id_arg,
                device_arg,
                proxy_url_arg,
                prompt_arg,
                target_keywords_arg,
                strict_target_filter_arg,
                max_targets_arg,
                use_sam_arg,
                box_th_arg,
                text_th_arg,
                min_target_conf_arg,
                topk_arg,
                sam_max_boxes_arg,
                min_mask_arg,
                iou_sup_arg,
                detect_side_arg,
                detect_interval_arg,
                enable_adaptive_interval_arg,
                adaptive_interval_max_arg,
                adaptive_conf_high_arg,
                adaptive_conf_low_arg,
                cache_hold_rounds_arg,
                combine_prompts_forward_arg,
                timeout_arg,
                fps_arg,
                min_obj_pts_arg,
                alpha_arg,
            ) = _parse_cli()
            main(
                gd_model_id=gd_model_id_arg,
                sam_model_id=sam_model_id_arg,
                device=device_arg,
                proxy_url=proxy_url_arg,
                prompt=prompt_arg,
                target_keywords=target_keywords_arg,
                strict_target_filter=strict_target_filter_arg,
                max_targets=max_targets_arg,
                use_sam=use_sam_arg,
                box_threshold=box_th_arg,
                text_threshold=text_th_arg,
                min_target_conf=min_target_conf_arg,
                topk_objects=topk_arg,
                sam_max_boxes=sam_max_boxes_arg,
                min_mask_pixels=min_mask_arg,
                mask_iou_suppress=iou_sup_arg,
                detect_max_side=detect_side_arg,
                detect_interval=detect_interval_arg,
                enable_adaptive_interval=enable_adaptive_interval_arg,
                adaptive_interval_max=adaptive_interval_max_arg,
                adaptive_conf_high=adaptive_conf_high_arg,
                adaptive_conf_low=adaptive_conf_low_arg,
                cache_hold_detect_rounds=cache_hold_rounds_arg,
                combine_prompts_forward=combine_prompts_forward_arg,
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
                min_object_points=min_obj_pts_arg,
                alpha=alpha_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
