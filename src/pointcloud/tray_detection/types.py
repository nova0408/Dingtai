from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# region 数据结构
@dataclass(frozen=True)
class TrayDetectionConfig:
    """料盘零训练检测配置。

    该配置对象描述 `TrayPointExcluder` 初始化 GroundingDINO 与可选 SAM 所需的全部参数，
    是调用层和检测层之间的稳定数据契约。

    设计思想：
    - 使用不可变 dataclass，把模型、阈值、缓存和 prompt 参数集中在一个显式结构中。
    - 默认值面向 Orbbec 料盘排除场景，调用方通常只需要覆盖 `use_sam` 或设备参数。
    - 默认值直接写在字段定义处，不再通过同文件 `DEFAULT_*` 常量二次转发，降低维护成本。
    - 参数保持语义完整，不使用无结构 dict 长距离透传。

    继承关系：
    - 不继承业务基类，不依赖动态分发。
    - 仅依赖 dataclass 生成初始化逻辑和只读字段约束。
    """

    gd_model_id: str = "IDEA-Research/grounding-dino-base"
    "GroundingDINO 模型 ID 或本地模型路径。"
    sam_model_id: str = "facebook/sam-vit-base"
    "SAM 模型 ID 或本地模型路径，仅 `use_sam=True` 时加载。"
    hf_cache_dir: str | None = None
    "HuggingFace 项目缓存目录；为 None 时使用 `.cache/huggingface`。"
    hf_local_files_only: bool = True
    "是否只使用本地缓存；True 时不会主动联网下载模型。"
    device: str = "cuda:0"
    "推理设备字符串，例如 `cuda:0` 或 `cpu`。"
    proxy_url: str = "http://127.0.0.1:4444"
    "模型下载代理地址；空字符串表示不注入代理环境变量。"
    prompt: str = "black tray,black pallet,rectangular black tray"
    "零训练检测提示词，使用逗号分隔多个目标。"
    target_keywords: str = "rectangular black tray,black tray,black pallet"
    "目标关键词，检测标签必须匹配这些关键词后才参与料盘排除。"
    strict_target_filter: bool = True
    "是否严格按 `target_keywords` 过滤非料盘候选。"
    max_targets: int = 2
    "每帧最多输出料盘目标数量，单位 个。"
    use_sam: bool = True
    "是否启用 SAM 精细分割；False 时使用检测框矩形 mask，速度更快。"
    box_threshold: float = 0.16
    "GroundingDINO box 阈值，范围通常为 0-1。"
    text_threshold: float = 0.08
    "GroundingDINO text 阈值，范围通常为 0-1。"
    min_confidence: float = 0.35
    "料盘候选最小置信度，范围 0-1。"
    topk_objects: int = 4
    "每次 DINO 前向最多保留的候选框数量，单位 个。"
    sam_max_boxes: int = 2
    "每帧最多进入 SAM 的候选框数量，单位 个。"
    sam_primary_only: bool = True
    "是否只对主目标默认执行 SAM；次目标可由置信度阈值决定。"
    sam_secondary_conf_threshold: float = 0.55
    "次目标进入 SAM 的最低置信度，范围 0-1。"
    combine_prompts_forward: bool = False
    "是否把多个 prompt 合并成一次 DINO 前向；False 更稳定，True 更快。"
    min_mask_pixels: int = 300
    "有效料盘 mask 的最小面积，单位 像素。"
    mask_iou_suppress: float = 0.65
    "候选 mask 去重 IoU 阈值，范围 0-1。"
    detect_max_side: int = 512
    "DINO 检测输入的最长边尺寸，单位 像素。"


@dataclass(frozen=True)
class TrayDetection:
    """料盘检测结果。

    该结构是 2D 零训练识别结果与 3D 点云排除流程之间的数据契约。

    设计思想：
    - 使用不可变 dataclass，保证检测线程和预览线程之间传递时不会被意外改写。
    - NumPy 数组字段保留图像坐标语义，点云索引由投影函数在外部计算。
    - `excluded_points` 在纯 2D 检测阶段为 0，在完成 2D mask 到 3D 点索引映射后回填。

    继承关系：
    - 不继承业务基类，避免引入隐式生命周期或魔术式动态分发。
    - 仅依赖 dataclass 的只读字段约束。
    """

    label_text: str
    "检测标签文本，来自 GroundingDINO 输出或 prompt 合并结果。"
    confidence_2d: float
    "2D 检测置信度，范围 0-1。"
    contour: np.ndarray
    "料盘轮廓点数组，形状为 `(K, 2)`，dtype 为 `int32`，坐标单位为像素。"
    mask: np.ndarray
    "料盘区域掩码，形状为 `(H, W)`，dtype 为 `uint8`，非零像素表示料盘区域。"
    excluded_points: int = 0
    "由该检测结果排除的点云数量，单位 点。"


@dataclass(frozen=True)
class TrayExclusionResult:
    """料盘点云排除结果。

    该结构封装一帧点云的料盘排除输出，供三平面位姿算法直接消费。

    设计思想：
    - 用 `(N,) bool` 掩码作为跨算法边界的数据契约，避免复制大点云。
    - 检测列表保留 2D 轮廓和 mask，便于测试脚本叠加预览。
    - 结果对象不可变，降低多线程管线中被后续流程改写的风险。

    继承关系：
    - 不继承业务基类。
    - 仅依赖 dataclass 生成初始化和只读字段约束。
    """

    excluded_mask: np.ndarray
    "点云排除掩码，形状为 `(N,)`，dtype 为 `bool`；True 表示该点属于料盘干扰区域。"
    detections: list[TrayDetection]
    "料盘检测结果列表，每个元素包含 2D mask、轮廓和对应排除点数量。"


# endregion
