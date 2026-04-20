# -*- coding: utf-8 -*-
from .registration_icp import eval_multi_thresholds, icp_p2pl_multiscale, icp_pt2pt_multiscale, info_matrix
from .registration_marker import choose_best_marker_pair
from .registration_pairwise import build_pose_graph, register_pair_src_to_tgt
from .registration_pipeline import merge_piece_and_save, optimize_pose_graph, save_transforms
from .registration_postprocess import PointCloudPostProcessor
from .pointcloud_primitives import (
    T_init_for_block,
    T_translate,
    aabb_from_center_half,
    clone_pcd,
    extract_box_center_and_half,
    load_npy_as_pcd,
    transform_point,
)
from .registration_quality import compute_pair_quality_weight
from .registration_icp_registry import (
    get_roi_icp_runner,
    list_roi_icp_methods,
    register_roi_icp_method,
    resolve_roi_icp_runner,
)
from .pointcloud_visual import colorize_by_cycle
from .pointcloud_visual import height_to_color
from .pointcloud_io import load_pcd

__all__ = [
    "T_translate",
    "T_init_for_block",
    "clone_pcd",
    "extract_box_center_and_half",
    "aabb_from_center_half",
    "transform_point",
    "load_npy_as_pcd",
    "icp_pt2pt_multiscale",
    "icp_p2pl_multiscale",
    "eval_multi_thresholds",
    "info_matrix",
    "choose_best_marker_pair",
    "register_pair_src_to_tgt",
    "build_pose_graph",
    "optimize_pose_graph",
    "save_transforms",
    "merge_piece_and_save",
    "compute_pair_quality_weight",
    "PointCloudPostProcessor",
    "colorize_by_cycle",
    "height_to_color",
    "load_pcd",
    "register_roi_icp_method",
    "get_roi_icp_runner",
    "list_roi_icp_methods",
    "resolve_roi_icp_runner",
]

