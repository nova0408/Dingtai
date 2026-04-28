from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.datas.kinematics.se3 import SE3_string


def _preprocess_for_feature(
    pcd: o3d.geometry.PointCloud, voxel_size_mm: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    if len(pcd.points) == 0:
        raise RuntimeError("输入点云为空，无法做粗配准。")
    down = pcd.voxel_down_sample(voxel_size_mm)
    if len(down.points) == 0:
        down = pcd
    normal_radius = voxel_size_mm * 2.0
    feature_radius = voxel_size_mm * 5.0
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return down, feature


def coarse_register_ransac(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    min_scale_mm: float = 10.0,
    max_scale_mm: float = 80.0,
) -> np.ndarray:
    voxel_size_mm = float(np.clip(max_scale_mm * 0.25, min_scale_mm, max_scale_mm))
    src_down, src_fpfh = _preprocess_for_feature(source, voxel_size_mm)
    tgt_down, tgt_fpfh = _preprocess_for_feature(target, voxel_size_mm)
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        max_scale_mm,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_scale_mm),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(80_000, 0.999),
    )
    return np.asarray(res.transformation, dtype=np.float64)


def coarse_register_fgr(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    min_scale_mm: float = 10.0,
    max_scale_mm: float = 80.0,
) -> np.ndarray:
    voxel_size_mm = float(np.clip(max_scale_mm * 0.25, min_scale_mm, max_scale_mm))
    src_down, src_fpfh = _preprocess_for_feature(source, voxel_size_mm)
    tgt_down, tgt_fpfh = _preprocess_for_feature(target, voxel_size_mm)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_scale_mm,
        iteration_number=128,
        maximum_tuple_count=5000,
        tuple_scale=0.95,
    )
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        option,
    )
    return np.asarray(res.transformation, dtype=np.float64)


def coarse_refine_with_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_error_mm: float = 20.0,
    max_iteration: int = 30,
) -> np.ndarray:
    src = o3d.geometry.PointCloud(source)
    tgt = o3d.geometry.PointCloud(target)
    normal_radius = max(max_error_mm * 2.0, 5.0)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    res = o3d.pipelines.registration.registration_colored_icp(
        src,
        tgt,
        max_error_mm,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    return np.asarray(res.transformation, dtype=np.float64)


def _weighted_kabsch(source_xyz: np.ndarray, target_xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.eye(4, dtype=np.float64)
    w = w / w_sum
    src_centroid = np.sum(source_xyz * w[:, None], axis=0)
    tgt_centroid = np.sum(target_xyz * w[:, None], axis=0)
    src_centered = source_xyz - src_centroid
    tgt_centered = target_xyz - tgt_centroid
    h = src_centered.T @ (tgt_centered * w[:, None])
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = tgt_centroid - r @ src_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform


def coarse_register_custom_rgb_guided(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray | None = None,
) -> np.ndarray:
    src = source.voxel_down_sample(5.0)
    tgt = target.voxel_down_sample(5.0)
    if len(src.points) == 0 or len(tgt.points) == 0:
        return np.eye(4, dtype=np.float64)
    src_xyz = np.asarray(src.points, dtype=np.float64)
    tgt_xyz = np.asarray(tgt.points, dtype=np.float64)
    src_rgb = np.asarray(src.colors, dtype=np.float64)
    tgt_rgb = np.asarray(tgt.colors, dtype=np.float64)
    if src_rgb.shape[0] != src_xyz.shape[0]:
        src_rgb = np.zeros((src_xyz.shape[0], 3), dtype=np.float64)
    if tgt_rgb.shape[0] != tgt_xyz.shape[0]:
        tgt_rgb = np.zeros((tgt_xyz.shape[0], 3), dtype=np.float64)
    tree = cKDTree(tgt_xyz)
    transform = (
        np.eye(4, dtype=np.float64) if init_transform is None else np.asarray(init_transform, dtype=np.float64).copy()
    )

    for max_dist_mm in (30.0, 20.0, 12.0, 8.0):
        color_thr = 0.35
        sigma_dist = max_dist_mm * 0.5
        sigma_color = 0.20
        for _ in range(12):
            src_xyz_tf = (transform[:3, :3] @ src_xyz.T).T + transform[:3, 3]
            dist, idx = tree.query(src_xyz_tf, k=1, distance_upper_bound=max_dist_mm)
            valid = np.isfinite(dist) & (idx < tgt_xyz.shape[0])
            if int(np.sum(valid)) < 30:
                break
            s = src_xyz_tf[valid]
            t = tgt_xyz[idx[valid]]
            color_delta = np.linalg.norm(src_rgb[valid] - tgt_rgb[idx[valid]], axis=1)
            color_valid = color_delta <= color_thr
            if int(np.sum(color_valid)) < 30:
                break
            s = s[color_valid]
            t = t[color_valid]
            d = dist[valid][color_valid]
            c = color_delta[color_valid]
            w_dist = np.exp(-(d * d) / (2.0 * sigma_dist * sigma_dist + 1e-12))
            w_color = np.exp(-(c * c) / (2.0 * sigma_color * sigma_color + 1e-12))
            delta = _weighted_kabsch(s, t, w_dist * w_color)
            transform = delta @ transform
            trans_step = float(np.linalg.norm(delta[:3, 3]))
            trace_val = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
            rot_step_deg = float(np.degrees(np.arccos(trace_val)))
            if trans_step < 0.01 and rot_step_deg < 0.01:
                break
    return transform


def _kmeans_rgb_labels(colors: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    n = colors.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(42)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = colors[init_idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist2 = np.sum((colors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = np.mean(colors[mask], axis=0)
            else:
                centers[i] = colors[rng.integers(0, n)]
    return labels


def coarse_register_by_color_clusters(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    num_clusters: int = 4,
) -> np.ndarray:
    src = source.voxel_down_sample(5.0)
    tgt = target.voxel_down_sample(5.0)
    if len(src.points) < 100 or len(tgt.points) < 100:
        return np.eye(4, dtype=np.float64)
    src_xyz = np.asarray(src.points, dtype=np.float64)
    tgt_xyz = np.asarray(tgt.points, dtype=np.float64)
    src_rgb = np.asarray(src.colors, dtype=np.float64)
    tgt_rgb = np.asarray(tgt.colors, dtype=np.float64)
    if src_rgb.shape[0] != src_xyz.shape[0] or tgt_rgb.shape[0] != tgt_xyz.shape[0]:
        return np.eye(4, dtype=np.float64)

    k = int(max(3, min(num_clusters, src_xyz.shape[0] // 200, tgt_xyz.shape[0] // 200)))
    src_labels = _kmeans_rgb_labels(src_rgb, k=k)
    tgt_labels = _kmeans_rgb_labels(tgt_rgb, k=k)

    src_c_xyz = np.zeros((k, 3), dtype=np.float64)
    tgt_c_xyz = np.zeros((k, 3), dtype=np.float64)
    src_c_rgb = np.zeros((k, 3), dtype=np.float64)
    tgt_c_rgb = np.zeros((k, 3), dtype=np.float64)
    src_w = np.zeros((k,), dtype=np.float64)
    tgt_w = np.zeros((k,), dtype=np.float64)
    for i in range(k):
        s_mask = src_labels == i
        t_mask = tgt_labels == i
        if np.any(s_mask):
            src_c_xyz[i] = np.mean(src_xyz[s_mask], axis=0)
            src_c_rgb[i] = np.mean(src_rgb[s_mask], axis=0)
            src_w[i] = float(np.sum(s_mask))
        if np.any(t_mask):
            tgt_c_xyz[i] = np.mean(tgt_xyz[t_mask], axis=0)
            tgt_c_rgb[i] = np.mean(tgt_rgb[t_mask], axis=0)
            tgt_w[i] = float(np.sum(t_mask))

    src_idx = np.where(src_w > 0)[0]
    tgt_idx = np.where(tgt_w > 0)[0]
    if src_idx.size < 3 or tgt_idx.size < 3:
        return np.eye(4, dtype=np.float64)
    src_cols = src_c_rgb[src_idx]
    tgt_cols = tgt_c_rgb[tgt_idx]
    src_counts = src_w[src_idx]
    tgt_counts = tgt_w[tgt_idx]

    color_cost = np.linalg.norm(src_cols[:, None, :] - tgt_cols[None, :, :], axis=2)
    count_ratio = np.abs(src_counts[:, None] - tgt_counts[None, :]) / np.maximum(
        np.maximum(src_counts[:, None], tgt_counts[None, :]), 1.0
    )
    cost = color_cost + 0.25 * count_ratio
    rows, cols = linear_sum_assignment(cost)
    if rows.size < 3:
        return np.eye(4, dtype=np.float64)

    src_corr = src_c_xyz[src_idx[rows]]
    tgt_corr = tgt_c_xyz[tgt_idx[cols]]
    color_diff = color_cost[rows, cols]
    pair_weight = np.minimum(src_counts[rows], tgt_counts[cols]) * np.exp(-(color_diff**2) / (2.0 * 0.2 * 0.2))
    init_transform = _weighted_kabsch(src_corr, tgt_corr, pair_weight)
    return coarse_register_custom_rgb_guided(source, target, init_transform=init_transform)


def evaluate_transform(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    strict_mm: float = 5.0,
    wide_mm: float = 10.0,
) -> tuple[o3d.pipelines.registration.RegistrationResult, o3d.pipelines.registration.RegistrationResult]:
    strict_eval = o3d.pipelines.registration.evaluate_registration(source, target, strict_mm, transform)
    wide_eval = o3d.pipelines.registration.evaluate_registration(source, target, wide_mm, transform)
    return strict_eval, wide_eval


def visualize_one_method(
    method_name: str,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
) -> None:
    raw_src = o3d.geometry.PointCloud(source)
    raw_tgt = o3d.geometry.PointCloud(target)
    reg_src = o3d.geometry.PointCloud(source)
    reg_tgt = o3d.geometry.PointCloud(target)
    reg_src.transform(transform)
    offset = np.array([0.0, 1000.0, 0.0], dtype=np.float64)
    reg_src.translate(offset)
    reg_tgt.translate(offset)

    axis_raw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    axis_reg = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=offset.tolist())

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(f"粗配准方法：{method_name} (下：原始，上：粗配准)", 1440, 900)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.set_background(np.array([0, 0, 0, 0]), None)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 1.5
    vis.add_geometry("raw_target", raw_tgt, mat)
    vis.add_geometry("raw_source", raw_src, mat)
    vis.add_geometry("reg_target", reg_tgt, mat)
    vis.add_geometry("reg_source", reg_src, mat)
    vis.add_geometry("axis_raw", axis_raw)
    vis.add_geometry("axis_reg", axis_reg)
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def main() -> None:
    pcd_1_path = PROJECT_ROOT / "experiments" / "pcd1.pcd"
    pcd_2_path = PROJECT_ROOT / "experiments" / "pcd2.pcd"
    if not pcd_1_path.exists() or not pcd_2_path.exists():
        raise FileNotFoundError(f"未找到缓存点云：{pcd_1_path} / {pcd_2_path}")

    pcd_1 = o3d.io.read_point_cloud(pcd_1_path)
    pcd_2 = o3d.io.read_point_cloud(pcd_2_path)
    if len(pcd_1.points) == 0 or len(pcd_2.points) == 0:
        raise RuntimeError("缓存点云为空。")

    ransac_t = coarse_register_ransac(pcd_2, pcd_1)
    fgr_t = coarse_register_fgr(pcd_2, pcd_1)
    ransac_col_t = coarse_refine_with_colored_icp(pcd_2, pcd_1, ransac_t)
    fgr_col_t = coarse_refine_with_colored_icp(pcd_2, pcd_1, fgr_t)
    custom_rgb_t = coarse_register_custom_rgb_guided(pcd_2, pcd_1)
    cluster_rgb_t = coarse_register_by_color_clusters(pcd_2, pcd_1)

    methods: list[tuple[str, np.ndarray]] = [
        ("fgr_fpfh", fgr_t),
        ("fgr_fpfh_colored", fgr_col_t),
        ("custom_rgb_guided", custom_rgb_t),
    ]

    for name, transform in methods:
        strict_eval, wide_eval = evaluate_transform(pcd_2, pcd_1, transform)
        logger.success(f"[{name}] T(PCD2->PCD1): {SE3_string(transform)}")
        logger.info(
            f"[{name}] strict@5mm fitness={strict_eval.fitness:.5f}, rmse={strict_eval.inlier_rmse:.5f} | "
            f"wide@10mm fitness={wide_eval.fitness:.5f}, rmse={wide_eval.inlier_rmse:.5f}"
        )
        visualize_one_method(name, pcd_2, pcd_1, transform)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断。")
    except Exception as exc:
        logger.error(f"运行失败：{exc}")
        traceback.print_exc()
