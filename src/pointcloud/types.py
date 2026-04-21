# -*- coding: utf-8 -*-
from __future__ import annotations

# region 导入
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import open3d as o3d

# endregion


# region 协议类型
@runtime_checkable
class PointLike(Protocol):
    """三维点对象最小协议。"""

    def as_array(self) -> np.ndarray:
        """返回 ``(3,)`` 坐标数组。"""
        ...


@runtime_checkable
class CropBoxLike(Protocol):
    """裁剪盒对象最小协议。"""

    @property
    def center(self) -> PointLike:
        """裁剪盒中心。"""
        ...

    @property
    def bounds(self) -> tuple[float, float, float]:
        """裁剪盒 XYZ 三方向半尺寸。"""
        ...


# endregion


# region 调试与后处理参数
@dataclass(frozen=True)
class DebugOptions:
    """调试与可视化选项。"""

    enable: bool = False

    pairwise_vox: float = 1.5
    final_piece_vox: float = 2.0
    final_merged_vox: float = 1.0

    unattended: bool = False
    icp_log_candidates: bool = True
    icp_log_scales: bool = True
    icp_log_timing: bool = False

    vis_before_icp: bool = False
    vis_marker_pair: bool = False
    vis_pairwise: bool = False
    vis_icp_chain_before_pose_graph: bool = False
    vis_final: bool = True
    vis_dynamic_roi: bool = False
    vis_before_postprocess_merged: bool = False

    vis_postprocess_crop: bool = False
    vis_postprocess_denoise: bool = False
    vis_postprocess_voxel: bool = False


@dataclass(frozen=True)
class CropProcessOptions:
    enable: bool = True
    strict: bool = False


@dataclass(frozen=True)
class RadiusOutlierRemovalOptions:
    enable: bool = True
    radius: float = 1.0
    min_neighbors: int = 30
    max_nn: int = 64
    batch_size: int = 100000
    nprobe: int = 16
    temp_memory_mb: int = 256


@dataclass(frozen=True)
class VoxelDownsampleOptions:
    enable: bool = True
    voxel_size: float = 0.1
    keep: str = "centroid"
    xyz_only: bool = False


@dataclass(frozen=True)
class PointCloudPostProcessOptions:
    crop: CropProcessOptions = field(default_factory=CropProcessOptions)
    denoise: RadiusOutlierRemovalOptions = field(default_factory=RadiusOutlierRemovalOptions)
    voxel: VoxelDownsampleOptions = field(default_factory=VoxelDownsampleOptions)


@dataclass(frozen=True)
class RegistrationPostprocessOptions:
    piece: PointCloudPostProcessOptions = field(default_factory=PointCloudPostProcessOptions)
    merged: PointCloudPostProcessOptions = field(
        default_factory=lambda: PointCloudPostProcessOptions(
            crop=CropProcessOptions(enable=True, strict=False),
            denoise=RadiusOutlierRemovalOptions(),
            voxel=VoxelDownsampleOptions(enable=True, voxel_size=0.1, keep="centroid", xyz_only=False),
        )
    )


# endregion


# region 配准日志与元信息
@dataclass(frozen=True)
class ICPScaleLog:
    """单层 ICP 结果。"""

    voxel_size: float
    fitness: float
    rmse: float
    source_points: int
    target_points: int
    delta_from_previous: np.ndarray
    delta_from_init: np.ndarray


@dataclass(frozen=True)
class RegistrationEval:
    """固定阈值配准评估。"""

    threshold_mm: float
    fitness: float
    rmse: float
    source_points: int
    target_points: int


@dataclass(frozen=True)
class MarkerRegistrationMeta:
    """crop 初始配准调试信息。"""

    src_roi_idx: int
    tgt_roi_idx: int
    center_distance: float
    init_translation: np.ndarray
    transform_src_to_tgt: np.ndarray
    translation_error: float
    rotation_error_deg: float
    residual_translation: np.ndarray
    coarse_logs: tuple[ICPScaleLog, ...]
    fine_logs: tuple[ICPScaleLog, ...]
    evals: tuple[RegistrationEval, ...]
    fallback: bool
    attempted_icp: bool
    fallback_reason: str
    src_marker_points: int
    tgt_marker_points: int


@dataclass(frozen=True)
class DynamicRoiMeta:
    """动态 ROI 构造调试信息。"""

    voxel_size: float
    dynamic_init_transform: np.ndarray
    fast_transform_rel: np.ndarray
    fast_fitness: float
    fast_rmse: float
    matched_src_voxel_count: int
    matched_tgt_voxel_count: int
    matched_src_point_count: int
    matched_tgt_point_count: int
    src_fit_voxels_in_tgt: o3d.geometry.PointCloud
    tgt_fit_voxels: o3d.geometry.PointCloud
    src_roi_in_tgt: o3d.geometry.PointCloud
    tgt_roi: o3d.geometry.PointCloud
    roi_source_mode: str
    used_full_roi_fallback: bool
    build_elapsed_sec: float


@dataclass(frozen=True)
class PairDebugMeta:
    """pairwise 配准调试信息。"""

    src_idx: int
    tgt_idx: int
    is_loop: bool
    marker: MarkerRegistrationMeta
    dynamic_roi: DynamicRoiMeta
    roi_refine_transform_rel: np.ndarray
    roi_logs: tuple[ICPScaleLog, ...]
    roi_evals: tuple[RegistrationEval, ...]
    roi_delta_t: np.ndarray
    roi_strategy: str
    roi_dz_mse_mm2: float
    roi_recover_used: bool
    roi_relaxed_accept: bool
    roi_limit_mm: float
    roi_limit_relax_mm: float
    quality_weight: float


# endregion


# region 算法配置与流程数据
@dataclass(frozen=True)
class AlgoSettings:
    """配准算法参数集合。"""

    eval_threshold_mm: float = 0.10
    eval_vox: float = 0.3

    marker_coarse_vox: tuple[float, ...] = (0.8,)
    marker_coarse_maxd: tuple[float, ...] = (5.0,)
    marker_coarse_iters: tuple[int, ...] = (20,)
    marker_fine_vox: tuple[float, ...] = (0.20,)
    marker_fine_maxd: tuple[float, ...] = (1.0,)
    marker_fine_iters: tuple[int, ...] = (30,)
    marker_cap_pts: int = 200000
    marker_max_translation_component_mm: float = 5.0

    overlap_coarse_voxels: float = 1.0
    overlap_fallback_voxels: float = 0.8
    overlap_fallback_fast_fitness_min: float = 0.20
    overlap_icp_max_corr_mm: float = 3.0
    overlap_icp_max_iters: int = 40
    overlap_cap_pts: int = 150000
    overlap_fit_radius_mm: float = 3.0
    overlap_min_matched_voxels: int = 50
    overlap_min_matched_points: int = 300
    overlap_use_voxel_matched_roi: bool = True
    overlap_voxel_roi_min_points: int = 5000

    roi_fine_vox: tuple[float, ...] = (0.8, 0.30, 0.10, 0.04)
    roi_fine_maxd: tuple[float, ...] = (2.0, 0.8, 0.25, 0.08)
    roi_fine_iters: tuple[int, ...] = (80, 120, 180, 260)
    roi_icp_method: str = "pt2pt"
    roi_cap_pts: int = 220000
    roi_max_translation_component_mm: float = 5.0
    roi_limit_relax_component_mm: float = 6.0
    roi_recover_enable: bool = True
    roi_recover_vox: tuple[float, ...] = (1.2, 0.4)
    roi_recover_maxd: tuple[float, ...] = (3.0, 0.8)
    roi_recover_iters: tuple[int, ...] = (40, 30)
    roi_recover_cap_pts: int = 160000

    adaptive_escalation_enable: bool = True
    adaptive_escalation_dz_mse_mm2: float = 0.08
    adaptive_escalation_stepwise_stop: bool = True
    adaptive_escalation_early_stop_dz_mse_mm2: float = 0.04
    adaptive_eval_dz_mse_max_corr_mm: float = 1.0
    adaptive_eval_dz_mse_cap_points: int = 12000

    pg_max_corr_dist: float = 1.0
    edge_prune_th: float = 0.25
    pref_loop: float = 2.0
    w_info_marker: float = 12.0
    w_info_roi: float = 1.0
    loop_extra_scale: float = 0.6
    after_opt_eval_threshold_mm: float = 0.10

    postprocess: RegistrationPostprocessOptions = field(default_factory=RegistrationPostprocessOptions)
    debug: DebugOptions = field(default_factory=DebugOptions)


@dataclass(frozen=True)
class PairPlanItem:
    """单条 pairwise 配准计划项。"""

    src_idx: int
    tgt_idx: int
    is_odometry: bool
    relative_transform: np.ndarray


@dataclass
class BlockCloudRecord:
    """单个块在单个 angle 流程中的完整记录。"""

    local_idx: int
    source_index: int
    origin_local: o3d.geometry.PointCloud
    crop_local: o3d.geometry.PointCloud
    init_full: o3d.geometry.PointCloud
    crop_init_world: o3d.geometry.PointCloud
    T_init: np.ndarray


@dataclass(frozen=True)
class PairRegistrationResult:
    """单条边的配准结果。"""

    src_idx: int
    tgt_idx: int
    src_roi_idx: int
    tgt_roi_idx: int
    transform_src_to_tgt: np.ndarray
    information: np.ndarray
    is_loop: bool
    meta: PairDebugMeta


# endregion


# region 结构化候选与缓存边
@dataclass(frozen=True)
class PairRegistrationCandidate:
    """pairwise 候选分支结果。"""

    tag: str
    algo: AlgoSettings
    src_roi_idx: int
    tgt_roi_idx: int
    marker_meta: MarkerRegistrationMeta
    src_roi_base: o3d.geometry.PointCloud
    tgt_roi_base: o3d.geometry.PointCloud
    dynamic_roi: DynamicRoiMeta
    roi_refine_transform_rel: np.ndarray
    roi_logs: tuple[ICPScaleLog, ...]
    roi_delta_t: np.ndarray
    roi_strategy: str
    roi_dz_mse_mm2: float
    roi_recover_used: bool
    roi_relaxed_accept: bool
    final_transform: np.ndarray
    roi_evals: tuple[RegistrationEval, ...]


@dataclass(frozen=True)
class PairCacheEdge:
    """缓存文件中的位姿图边。"""

    src_idx: int
    tgt_idx: int
    is_odometry: bool
    transform_src_to_tgt: np.ndarray
    information: np.ndarray
    extra: dict[str, object] = field(default_factory=dict)


# endregion
