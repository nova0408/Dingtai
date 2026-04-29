from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

from .hf_model_cache import apply_download_proxy, load_pretrained_with_project_cache, prepare_hf_cache_dir
from .tray_detection_types import TrayDetection, TrayDetectionConfig, TrayExclusionResult
from .tray_detection_utils import (
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
from .tray_projection import collect_indices_in_mask

# region 默认路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE_DIR = str(PROJECT_ROOT / ".cache" / "huggingface")
# endregion


# region 料盘识别与排除
class TrayPointExcluder:
    """料盘识别与点云排除器。

    该类在 `src.pointcloud` 内完整实现料盘识别流程，不依赖 `test` 或 `experiments` 模块。
    它负责加载 GroundingDINO、按 prompt 检测料盘候选、按需调用 SAM 生成精细 mask，
    并把 2D mask 映射到裁切点云的 `(N,) bool` 排除掩码。

    职责边界：
    - 负责 2D 识别、2D mask 后处理、mask IoU 抑制和 2D->3D 点索引映射。
    - 不负责相机采集、Open3D 预览、三平面拟合或坐标系计算。
    - 不保存历史帧，不做跟踪；实时管线是否跳帧由调用方控制。

    设计思想：
    - 把模型生命周期集中在一个显式对象中，避免每帧重复加载模型。
    - `detect` 输出纯 2D 结果，`exclude_points` 在调用方提供投影坐标后生成 3D 排除掩码。
    - SAM 可关闭；关闭时使用矩形 mask，牺牲精细边界以换取性能。

    继承关系：
    - 不继承任何业务基类，不使用魔术式动态分发。
    - 依赖 HuggingFace Transformers/PyTorch 的显式模型对象。

    线程与资源语义：
    - 实例持有 GPU/CPU 模型，建议在单个计算线程中复用。
    - 不保证同一实例可被多个线程同时调用；多线程时应由外层队列串行化 `detect`。
    - 模型随 Python 对象生命周期释放，未额外持有相机或文件句柄。
    """

    def __init__(self, config: TrayDetectionConfig) -> None:
        """初始化料盘识别器并加载模型。

        Parameters
        ----------
        config:
            料盘检测配置对象。该对象包含模型 ID、缓存目录、推理设备、prompt 和阈值。

        Raises
        ------
        RuntimeError
            当请求 CUDA 但环境没有 GPU，或本地模型缓存不可用时抛出。

        Notes
        -----
        初始化会加载 GroundingDINO；当 `config.use_sam=True` 时还会加载 SAM。
        该方法开销较大，应在工作线程启动时调用一次，而不是每帧调用。
        """
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
        self.detect_count = 0

        cache_dir = config.hf_cache_dir if config.hf_cache_dir is not None else DEFAULT_HF_CACHE_DIR
        self.hf_cache_dir = prepare_hf_cache_dir(cache_dir)
        apply_download_proxy(str(config.proxy_url).strip())

        logger.info(f"加载 GroundingDINO：{config.gd_model_id}")
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
        self._use_cuda = str(self.device).startswith("cuda")
        if self._use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam_processor = None
        self.sam_model = None
        if self.use_sam:
            logger.info(f"加载 SAM：{config.sam_model_id}")
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
        else:
            logger.info("跳过 SAM：使用检测框 mask（更快）")

        logger.success(f"料盘识别器初始化成功：device {self.device}, hf_cache_dir {self.hf_cache_dir}")

    def detect(self, frame_bgr: np.ndarray) -> list[TrayDetection]:
        """在 BGR 图像中检测料盘区域。

        Parameters
        ----------
        frame_bgr:
            OpenCV BGR 图像，形状为 `(H, W, 3)`，dtype 通常为 `uint8`，坐标单位为像素。

        Returns
        -------
        detections:
            料盘检测结果列表。每个结果的 `mask` 形状为 `(H, W)`，`contour` 形状为 `(K, 2)`。

        Notes
        -----
        该方法只生成 2D 检测结果，不直接访问点云。点云排除由 `exclude_points` 使用投影坐标完成。
        """
        self.detect_count += 1
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
            mask = self._build_candidate_mask(
                candidate_index=idx,
                score=float(score),
                pil_img=full_pil,
                box_xyxy=box_full,
                out_h=h,
                out_w=w,
            )
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

        kept = suppress_masks_by_iou(
            candidates=candidates,
            mask_iou_suppress=self.mask_iou_suppress,
            max_count=self.topk_objects,
        )
        return [
            TrayDetection(
                label_text=label_text,
                confidence_2d=float(score),
                contour=contour,
                mask=mask,
                excluded_points=0,
            )
            for mask, score, label_text, contour, _area in kept[: self.max_targets]
        ]

    def exclude_points(
        self,
        frame_bgr: np.ndarray,
        uv: np.ndarray,
        valid_proj: np.ndarray,
        total_points: int,
    ) -> TrayExclusionResult:
        """识别料盘并生成点云排除掩码。

        Parameters
        ----------
        frame_bgr:
            OpenCV BGR 图像，形状为 `(H, W, 3)`，dtype 通常为 `uint8`。
        uv:
            点云投影像素坐标，形状为 `(N, 2)`，dtype 为 `int32`。第 0 列为 U，第 1 列为 V。
        valid_proj:
            有效投影掩码，形状为 `(N,)`，dtype 为 `bool`。True 表示对应点可索引图像。
        total_points:
            当前裁切点云点数，单位 点，必须与 `uv.shape[0]` 对齐。

        Returns
        -------
        result:
            料盘排除结果。`excluded_mask` 形状为 `(N,)`，True 表示对应点属于料盘区域。

        Notes
        -----
        该方法捕获检测异常并返回全 False 掩码，避免单帧模型失败中断实时管线。
        """
        # excluded: (N,) bool，与当前裁切点云逐点对齐。
        excluded = np.zeros((int(total_points),), dtype=bool)
        out: list[TrayDetection] = []
        try:
            detections = self.detect(frame_bgr)
        except Exception as exc:
            logger.exception(f"料盘识别失败，本帧不执行料盘排除：{exc}")
            return TrayExclusionResult(excluded_mask=excluded, detections=[])

        for det in detections:
            # ids 是原始点云索引，不是压缩后的连续索引，可直接回写 excluded[ids]。
            ids = collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=det.mask)
            if ids.size == 0:
                continue
            excluded[ids] = True
            out.append(
                TrayDetection(
                    label_text=det.label_text,
                    confidence_2d=det.confidence_2d,
                    contour=det.contour,
                    mask=det.mask,
                    excluded_points=int(ids.size),
                )
            )
        return TrayExclusionResult(excluded_mask=excluded, detections=out)

    def _detect_prompt_boxes(self, pil_img: Image.Image) -> list[tuple[np.ndarray, float, str]]:
        """按配置 prompt 执行 GroundingDINO 候选框检测。

        Parameters
        ----------
        pil_img:
            RGB PIL 图像，尺寸为检测缩放后的 `(W, H)`。

        Returns
        -------
        box_items:
            候选框列表。每个元素为 `(box_xyxy, score, label)`，其中 `box_xyxy` 形状为 `(4,)`，
            dtype 为 `float32`，坐标单位为缩放后图像像素。
        """
        if self.combine_prompts_forward:
            box_items = self._grounding_detect_topk(
                pil_img=pil_img,
                prompt_text=self.combined_prompt,
                topk=self.topk_objects,
            )
            return [(b, s, merge_label_text(raw_label=lb, prompt_term="")) for b, s, lb in box_items]

        per_prompt_topk = int(max(1, min(self.topk_objects, max(2, self.max_targets))))
        box_items: list[tuple[np.ndarray, float, str]] = []
        for term in self.prompt_terms:
            term_items = self._grounding_detect_topk(pil_img=pil_img, prompt_text=term, topk=per_prompt_topk)
            for box_det, score, label_text in term_items:
                box_items.append((box_det, float(score), merge_label_text(raw_label=label_text, prompt_term=term)))
        box_items.sort(key=lambda x: x[1], reverse=True)
        return box_items[: self.topk_objects * 3]

    def _filter_target_boxes(
        self, box_items: list[tuple[np.ndarray, float, str]]
    ) -> list[tuple[np.ndarray, float, str]]:
        """按置信度和目标关键词过滤 DINO 候选框。

        Parameters
        ----------
        box_items:
            DINO 候选框列表。每个候选包含 `(4,) float32` 像素框、0-1 置信度和标签文本。

        Returns
        -------
        filtered_items:
            过滤后的候选框列表，仍保持 `(box_xyxy, score, label)` 结构。
        """
        filtered_items: list[tuple[np.ndarray, float, str]] = []
        for box_det, score, merged_label in box_items:
            if float(score) < self.min_target_conf:
                continue
            is_target = self._is_target_label(merged_label)
            if self.strict_target_filter and not is_target:
                continue
            filtered_items.append((box_det, float(score), str(merged_label)))
        return filtered_items

    def _grounding_detect_topk(
        self, pil_img: Image.Image, prompt_text: str, topk: int
    ) -> list[tuple[np.ndarray, float, str]]:
        """执行一次 GroundingDINO 前向并返回 top-k 框。

        Parameters
        ----------
        pil_img:
            RGB PIL 图像，尺寸为检测缩放后的 `(W, H)`。
        prompt_text:
            单个或合并后的检测提示词。
        topk:
            最多保留候选数量，单位 个。

        Returns
        -------
        items:
            候选列表。每个候选为 `(box_xyxy, score, label)`；`box_xyxy` 形状为 `(4,)`，
            坐标单位为 `pil_img` 像素。
        """
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
        label_arr = normalize_label_list(labels=labels, expect_len=len(score_arr))

        # order: (M,) int；按置信度降序排列候选索引。
        order = np.argsort(-score_arr)
        keep = order[: int(max(1, topk))]
        return [(box_arr[int(i)], float(score_arr[int(i)]), label_arr[int(i)]) for i in keep]

    def _autocast_ctx(self):
        """返回当前设备对应的自动混合精度上下文。

        Returns
        -------
        ctx:
            CUDA 下为 `torch.autocast`，CPU 下为空上下文。该返回值只用于 `with` 语句。
        """
        if self._use_cuda:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _is_target_label(self, label_text: str) -> bool:
        """判断候选标签是否属于目标料盘关键词。

        Parameters
        ----------
        label_text:
            DINO 返回或 prompt 合并后的标签文本。

        Returns
        -------
        is_target:
            True 表示该标签包含任一目标关键词，或未配置关键词时默认接受。
        """
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
        """根据候选框生成料盘 mask。

        Parameters
        ----------
        candidate_index:
            当前候选在过滤后列表中的顺序，0 表示主目标。
        score:
            当前候选 2D 置信度，范围 0-1。
        pil_img:
            原始尺寸 RGB PIL 图像；当不使用 SAM 时可以为 None。
        box_xyxy:
            原图坐标系候选框，形状为 `(4,)`，顺序为 x1/y1/x2/y2，单位 像素。
        out_h, out_w:
            输出 mask 高宽，单位 像素。

        Returns
        -------
        mask:
            料盘 mask，形状为 `(out_h, out_w)`，dtype 为 `uint8`；失败时返回 None。
        """
        use_sam_for_this = False
        if self.use_sam:
            if not self.sam_primary_only:
                use_sam_for_this = True
            else:
                use_sam_for_this = int(candidate_index) == 0 or float(score) >= self.sam_secondary_conf_threshold
        if use_sam_for_this:
            return self._sam_segment_box(pil_img=pil_img, box_xyxy=box_xyxy, out_h=out_h, out_w=out_w)
        return build_rect_mask(box_xyxy=box_xyxy, h=out_h, w=out_w)

    def _sam_segment_box(
        self, pil_img: Image.Image | None, box_xyxy: np.ndarray, out_h: int, out_w: int
    ) -> np.ndarray | None:
        """使用 SAM 对单个候选框做精细分割。

        Parameters
        ----------
        pil_img:
            原始尺寸 RGB PIL 图像。
        box_xyxy:
            候选框，形状为 `(4,)`，单位 像素。
        out_h, out_w:
            输出 mask 高宽，单位 像素。

        Returns
        -------
        mask:
            SAM 分割 mask，形状为 `(out_h, out_w)`，dtype 为 `uint8`；模型不可用时返回 None。
        """
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


# region 模型与数组工具
def _resolve_device(device: str) -> torch.device:
    """解析推理设备字符串。

    Parameters
    ----------
    device:
        设备字符串，例如 `cuda:0` 或 `cpu`。

    Returns
    -------
    torch_device:
        PyTorch 设备对象。

    Raises
    ------
    RuntimeError
        请求 CUDA 但当前环境没有可用 GPU 时抛出。
    """
    d = str(device).strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。")
    if d == "cpu":
        return torch.device("cpu")
    return torch.device(d)


def _tensor_dict_to_device(batch: dict, device: torch.device) -> dict:
    """把 Transformers batch 中的 tensor 移动到指定设备。

    Parameters
    ----------
    batch:
        Transformers processor 返回的字典。值可能是 torch tensor，也可能是普通对象。
    device:
        目标 PyTorch 设备。

    Returns
    -------
    out:
        与输入 key 相同的字典；tensor 值会被移动到 `device`。
    """
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# endregion
