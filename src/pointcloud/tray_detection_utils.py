from __future__ import annotations

import cv2
import numpy as np


# region 文本与标签工具
def parse_keywords(text: str) -> list[str]:
    """解析逗号分隔关键词列表。

    Parameters
    ----------
    text:
        用户或配置给出的关键词字符串，支持英文逗号和中文逗号分隔。

    Returns
    -------
    keywords:
        小写关键词列表。空白项会被丢弃，列表顺序与输入顺序一致。
    """
    raw = str(text).replace("，", ",").split(",")
    return [s.strip().lower() for s in raw if len(s.strip()) > 0]


def build_combined_prompt(terms: list[str]) -> str:
    """把多个 prompt 合并成 GroundingDINO 可用文本。

    Parameters
    ----------
    terms:
        prompt 词条列表，每个元素是一段目标描述文本。

    Returns
    -------
    prompt:
        GroundingDINO 可消费的文本提示词。多个词条使用句点分隔；输入为空时返回 `object`。
    """
    clean = [str(t).strip() for t in terms if len(str(t).strip()) > 0]
    if len(clean) == 0:
        return "object"
    return ". ".join(clean)


def normalize_label_list(labels, expect_len: int) -> list[str]:
    """把模型返回标签规范化为固定长度字符串列表。

    Parameters
    ----------
    labels:
        Transformers 后处理返回的标签集合，可能是 list、tuple、ndarray、可迭代对象或 None。
    expect_len:
        期望输出标签数量，单位 个。

    Returns
    -------
    normalized:
        长度为 `expect_len` 的字符串列表。不足时补空字符串，过长时截断。

    Notes
    -----
    该函数只做模型输出适配，不解释标签语义，也不参与目标过滤。
    """
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


def merge_label_text(raw_label: str, prompt_term: str) -> str:
    """合并模型标签和 prompt 词，避免泛化标签丢失目标语义。

    Parameters
    ----------
    raw_label:
        模型返回的原始标签文本。
    prompt_term:
        当前前向使用的 prompt 词条。

    Returns
    -------
    label_text:
        用于后续关键词过滤和日志输出的标签文本。模型只返回泛化标签时回退到 prompt。
    """
    r = str(raw_label).strip().lower()
    p = str(prompt_term).strip()
    if len(r) == 0:
        return p
    if r in {"object", "objects", "thing", "item"}:
        return p if len(p) > 0 else str(raw_label).strip()
    return str(raw_label).strip()


# endregion


# region 图像框与 mask 工具
def resize_for_detection(frame_bgr: np.ndarray, detect_max_side: int) -> tuple[np.ndarray, float]:
    """按最长边缩放检测图像。

    Parameters
    ----------
    frame_bgr:
        OpenCV BGR 图像，形状为 `(H, W, 3)`，dtype 通常为 `uint8`。
    detect_max_side:
        检测输入最长边上限，单位 像素。

    Returns
    -------
    resized:
        缩放后的 BGR 图像，形状为 `(H2, W2, 3)`。若原图未超过上限则返回原对象。
    restore_scale:
        从缩放图坐标映射回原图坐标的倍率。未缩放时为 `1.0`。
    """
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= int(detect_max_side):
        return frame_bgr, 1.0
    scale = float(detect_max_side) / float(m)
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, 1.0 / scale


def scale_box_xyxy(box_xyxy: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
    """把检测缩放图上的 xyxy 框映射回原图并裁剪到图像范围。

    Parameters
    ----------
    box_xyxy:
        检测框数组，形状为 `(4,)`，顺序为 `x1, y1, x2, y2`，单位 像素。
    scale:
        从检测图坐标恢复到原图坐标的倍率。
    w, h:
        原图宽高，单位 像素。

    Returns
    -------
    box:
        裁剪并修正端点顺序后的检测框，形状为 `(4,)`，dtype 为 `float32`。
    """
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


def build_rect_mask(box_xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    """根据 xyxy 框生成矩形 mask。

    Parameters
    ----------
    box_xyxy:
        检测框数组，形状为 `(4,)`，顺序为 `x1, y1, x2, y2`，单位 像素。
    h, w:
        输出 mask 高宽，单位 像素。

    Returns
    -------
    mask:
        二值 mask，形状为 `(H, W)`，dtype 为 `uint8`。目标区域为 255，背景为 0。
    """
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


def mask_to_contour(mask: np.ndarray, min_mask_pixels: int) -> np.ndarray:
    """从 mask 提取最大外轮廓凸包。

    Parameters
    ----------
    mask:
        二值 mask，形状为 `(H, W)`，dtype 通常为 `uint8`。
    min_mask_pixels:
        有效轮廓最小面积，单位 像素。

    Returns
    -------
    contour:
        最大外轮廓凸包点，形状为 `(K, 2)`，dtype 为 `int32`，坐标单位为像素。
        未找到有效轮廓时返回空数组。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.int32)
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < float(min_mask_pixels):
        return np.empty((0, 2), dtype=np.int32)
    hull = cv2.convexHull(c).reshape(-1, 2)
    return hull.astype(np.int32)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算两个二值 mask 的 IoU。

    Parameters
    ----------
    mask_a, mask_b:
        二值 mask，形状均为 `(H, W)`，非零像素表示目标区域。

    Returns
    -------
    iou:
        交并比，范围 0-1。两个 mask 均为空时返回 0。
    """
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def suppress_masks_by_iou(
    candidates: list[tuple[np.ndarray, float, str, np.ndarray, int]],
    mask_iou_suppress: float,
    max_count: int,
) -> list[tuple[np.ndarray, float, str, np.ndarray, int]]:
    """按置信度和 mask IoU 抑制重复候选。

    Parameters
    ----------
    candidates:
        候选列表。每个元素包含 `mask, confidence, label_text, contour, point_count`。
        mask 形状为 `(H, W)`，contour 形状为 `(K, 2)`。
    mask_iou_suppress:
        判定候选重复的 IoU 阈值，范围 0-1。
    max_count:
        最多保留的候选数量，单位 个。

    Returns
    -------
    kept:
        经置信度排序和 IoU 抑制后的候选列表。
    """
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept: list[tuple[np.ndarray, float, str, np.ndarray, int]] = []
    for cand in candidates:
        keep = True
        for item in kept:
            if mask_iou(cand[0], item[0]) >= float(mask_iou_suppress):
                keep = False
                break
        if keep:
            kept.append(cand)
        if len(kept) >= int(max_count):
            break
    return kept


# endregion
