from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TeaserLikeParams:
    voxel_mm: float = 8.0
    max_corr_mm: float = 120.0
    color_gate: float = 0.60
    fpfh_ratio_test: float = 0.92
    noise_bound_mm: float = 0.0
    cbar2: float = 1.0
    min_nodes: int = 8
    kcore_expand_ratio: float = 0.55
    gnc_factor: float = 1.4
    gnc_iters: int = 100
    gnc_cost_tol: float = 1e-6
    joint_schedule_mm: tuple[float, ...] = (35.0, 22.0, 14.0, 9.0)
    sigma_color: float = 0.20
    color_weight_power: float = 1.8
    colored_icp_mm: float = 15.0
    colored_icp_iters: int = 25
    strict_eval_mm: float = 5.0
    wide_eval_mm: float = 10.0
    score_color_weight: float = 1.10


@dataclass(frozen=True)
class TeaserLikeCoarseResult:
    transform: np.ndarray
    method_name: str
    used_noise_bound_mm: float
    num_raw_matches: int
    num_filtered_matches: int
    num_inlier_nodes: int
    fitness_strict: float
    rmse_strict: float
    fitness_wide: float
    rmse_wide: float
    color_residual: float


def _preprocess_feature_cloud(
    pcd: o3d.geometry.PointCloud, voxel_mm: float
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    down = pcd.voxel_down_sample(voxel_mm)
    if len(down.points) == 0:
        down = pcd
    normal_radius = max(2.0 * voxel_mm, 5.0)
    fpfh_radius = max(5.0 * voxel_mm, 10.0)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100),
    )
    return down, np.asarray(feat.data, dtype=np.float64).T


def _preprocess_for_global_registration(
    pcd: o3d.geometry.PointCloud, voxel_mm: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    down = pcd.voxel_down_sample(voxel_mm)
    if len(down.points) == 0:
        down = pcd
    normal_radius = max(2.0 * voxel_mm, 5.0)
    fpfh_radius = max(5.0 * voxel_mm, 10.0)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100),
    )
    return down, feat


def _weighted_kabsch(src_xyz: np.ndarray, tgt_xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.eye(4, dtype=np.float64)
    w = w / w_sum
    src_centroid = np.sum(src_xyz * w[:, None], axis=0)
    tgt_centroid = np.sum(tgt_xyz * w[:, None], axis=0)
    src_c = src_xyz - src_centroid
    tgt_c = tgt_xyz - tgt_centroid
    h = src_c.T @ (tgt_c * w[:, None])
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = tgt_centroid - r @ src_centroid
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = r
    delta[:3, 3] = t
    return delta


def _build_color_aware_correspondences(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    params: TeaserLikeParams,
) -> tuple[np.ndarray, np.ndarray]:
    src_down, src_fpfh = _preprocess_feature_cloud(src, params.voxel_mm)
    tgt_down, tgt_fpfh = _preprocess_feature_cloud(tgt, params.voxel_mm)
    src_xyz = np.asarray(src_down.points, dtype=np.float64)
    tgt_xyz = np.asarray(tgt_down.points, dtype=np.float64)
    src_rgb = np.asarray(src_down.colors, dtype=np.float64)
    tgt_rgb = np.asarray(tgt_down.colors, dtype=np.float64)
    if src_rgb.shape[0] != src_xyz.shape[0]:
        src_rgb = np.zeros((src_xyz.shape[0], 3), dtype=np.float64)
    if tgt_rgb.shape[0] != tgt_xyz.shape[0]:
        tgt_rgb = np.zeros((tgt_xyz.shape[0], 3), dtype=np.float64)

    if src_fpfh.shape[0] == 0 or tgt_fpfh.shape[0] == 0:
        raise RuntimeError("FPFH 特征为空，无法建立对应。")

    tgt_feat_tree = cKDTree(tgt_fpfh)
    src_feat_tree = cKDTree(src_fpfh)
    d_st, idx_st = tgt_feat_tree.query(src_fpfh, k=2)
    _, idx_ts = src_feat_tree.query(tgt_fpfh, k=1)

    src_idx = np.arange(src_fpfh.shape[0], dtype=np.int32)
    mutual = idx_ts[idx_st[:, 0]] == src_idx
    ratio = d_st[:, 0] / np.maximum(d_st[:, 1], 1e-12)
    ratio_ok = ratio <= params.fpfh_ratio_test
    keep = mutual & ratio_ok
    if int(np.sum(keep)) < 40:
        keep = mutual
    if int(np.sum(keep)) < 40:
        keep = np.ones_like(keep, dtype=bool)

    src_sel = src_idx[keep]
    tgt_sel = idx_st[keep, 0]
    geo = np.linalg.norm(src_xyz[src_sel] - tgt_xyz[tgt_sel], axis=1)
    col = np.linalg.norm(src_rgb[src_sel] - tgt_rgb[tgt_sel], axis=1)
    valid = (geo <= params.max_corr_mm) & (col <= params.color_gate)
    if int(np.sum(valid)) < 40:
        valid = geo <= params.max_corr_mm
    src_pts = src_xyz[src_sel[valid]]
    tgt_pts = tgt_xyz[tgt_sel[valid]]
    if src_pts.shape[0] < 20:
        raise RuntimeError(f"有效对应过少: {src_pts.shape[0]}")
    return src_pts.T, tgt_pts.T


def _compute_tims(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[1]
    tim_count = n * (n - 1) // 2
    tims = np.zeros((3, tim_count), dtype=np.float64)
    edge_map = np.zeros((2, tim_count), dtype=np.int32)
    cursor = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            tims[:, cursor] = points[:, j] - points[:, i]
            edge_map[0, cursor] = i
            edge_map[1, cursor] = j
            cursor += 1
    return tims, edge_map


def _tls_scalar_estimate(values: np.ndarray, ranges: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if x.size != r.size or x.size < 2:
        return float(np.median(x) if x.size else 0.0), np.zeros((x.size,), dtype=bool)

    events: list[tuple[float, int]] = []
    for i in range(x.size):
        events.append((x[i] - r[i], i + 1))
        events.append((x[i] + r[i], -i - 1))
    events.sort(key=lambda it: it[0])

    w = 1.0 / np.maximum(r * r, 1e-12)
    ranges_sum = float(np.sum(r))
    dot_xw = 0.0
    dot_w = 0.0
    consensus = 0
    sum_x = 0.0
    sum_x2 = 0.0
    best_cost = float("inf")
    best_est = float(np.median(x))
    for _, signed_idx in events:
        idx = abs(signed_idx) - 1
        eps = 1 if signed_idx > 0 else -1
        consensus += eps
        dot_w += eps * w[idx]
        dot_xw += eps * w[idx] * x[idx]
        ranges_sum -= eps * r[idx]
        sum_x += eps * x[idx]
        sum_x2 += eps * x[idx] * x[idx]
        if dot_w <= 1e-12:
            continue
        est = dot_xw / dot_w
        residual = consensus * est * est + sum_x2 - 2.0 * sum_x * est
        cost = residual + ranges_sum
        if cost < best_cost:
            best_cost = cost
            best_est = est
    return float(best_est), np.abs(x - best_est) <= r


def _estimate_noise_bound_from_tims(src_tims: np.ndarray, dst_tims: np.ndarray, cbar2: float) -> float:
    diff = np.abs(np.linalg.norm(src_tims, axis=0) - np.linalg.norm(dst_tims, axis=0))
    base = float(np.percentile(diff, 60))
    return max(1.5, base / max(2.0 * np.sqrt(cbar2), 1e-6))


def _scale_inlier_mask(src_tims: np.ndarray, dst_tims: np.ndarray, noise_bound: float, cbar2: float) -> np.ndarray:
    v1 = np.linalg.norm(src_tims, axis=0)
    v2 = np.linalg.norm(dst_tims, axis=0)
    beta = 2.0 * noise_bound * np.sqrt(cbar2)
    return np.abs(v1 - v2) <= beta


def _build_adjacency(n: int, edges: np.ndarray, valid: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    valid_idx = np.where(valid)[0]
    for k in valid_idx:
        i = int(edges[0, k])
        j = int(edges[1, k])
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _kcore_values(adj: list[set[int]]) -> np.ndarray:
    n = len(adj)
    degree = np.array([len(v) for v in adj], dtype=np.int32)
    removed = np.zeros((n,), dtype=bool)
    core = np.zeros((n,), dtype=np.int32)
    for _ in range(n):
        alive = np.where(~removed)[0]
        if alive.size == 0:
            break
        i = int(alive[np.argmin(degree[alive])])
        removed[i] = True
        core[i] = int(degree[i])
        for nb in list(adj[i]):
            if not removed[nb]:
                degree[nb] -= 1
    return core


def _select_inlier_nodes(adj: list[set[int]], params: TeaserLikeParams) -> np.ndarray:
    n = len(adj)
    if n <= params.min_nodes:
        return np.arange(n, dtype=np.int32)

    degree = np.array([len(v) for v in adj], dtype=np.int32)
    core = _kcore_values(adj)
    max_core = int(np.max(core))
    selected = set(np.where(core >= max_core)[0].tolist())
    if len(selected) < params.min_nodes:
        for idx in np.argsort(-degree):
            selected.add(int(idx))
            if len(selected) >= params.min_nodes:
                break

    selected_list = list(selected)
    remaining = [i for i in range(n) if i not in selected]
    remaining.sort(key=lambda i: degree[i], reverse=True)
    for node in remaining:
        if not selected_list:
            break
        conn = len(adj[node].intersection(selected))
        ratio = conn / max(len(selected), 1)
        if ratio >= params.kcore_expand_ratio:
            selected.add(node)
            selected_list.append(node)

    if len(selected) < params.min_nodes:
        for idx in np.argsort(-degree):
            selected.add(int(idx))
            if len(selected) >= params.min_nodes:
                break
    return np.array(sorted(selected), dtype=np.int32)


def _svd_rotation(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    h = src @ np.diag(weights) @ dst.T
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def _gnc_tls_rotation(
    src_tims: np.ndarray,
    dst_tims: np.ndarray,
    noise_bound: float,
    params: TeaserLikeParams,
) -> tuple[np.ndarray, np.ndarray]:
    n = src_tims.shape[1]
    if n < 3:
        return np.eye(3, dtype=np.float64), np.zeros((n,), dtype=bool)
    noise_sq = max(noise_bound * noise_bound, 1e-12)
    r = np.eye(3, dtype=np.float64)
    w = np.ones((n,), dtype=np.float64)
    prev_cost = float("inf")

    residual_sq = np.sum((dst_tims - r @ src_tims) ** 2, axis=0)
    max_res = float(np.max(residual_sq))
    mu = 1.0 / max(2.0 * max_res / noise_sq - 1.0, 1e-6)

    for _ in range(params.gnc_iters):
        r = _svd_rotation(src_tims, dst_tims, w)
        residual_sq = np.sum((dst_tims - r @ src_tims) ** 2, axis=0)
        th1 = (mu + 1.0) / mu * noise_sq
        th2 = mu / (mu + 1.0) * noise_sq
        cost = 0.0
        for i in range(n):
            cost += w[i] * residual_sq[i]
            if residual_sq[i] >= th1:
                w[i] = 0.0
            elif residual_sq[i] <= th2:
                w[i] = 1.0
            else:
                w[i] = np.sqrt(noise_sq * mu * (mu + 1.0) / max(residual_sq[i], 1e-12)) - mu
        if abs(cost - prev_cost) < params.gnc_cost_tol:
            break
        prev_cost = cost
        mu *= params.gnc_factor
    return r, w >= 0.5


def _tls_translation(src: np.ndarray, dst: np.ndarray, noise_bound: float, cbar2: float) -> tuple[np.ndarray, np.ndarray]:
    raw_t = dst - src
    beta = noise_bound * np.sqrt(cbar2)
    ranges = np.ones((raw_t.shape[1],), dtype=np.float64) * beta
    inlier_all = np.ones((raw_t.shape[1],), dtype=bool)
    trans = np.zeros((3,), dtype=np.float64)
    for axis in range(3):
        est, inlier = _tls_scalar_estimate(raw_t[axis, :], ranges)
        trans[axis] = est
        inlier_all &= inlier
    return trans, inlier_all


def _normal_variation_score(points: np.ndarray, normals: np.ndarray, k: int = 16) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    tree = cKDTree(points)
    _, knn = tree.query(points, k=min(k + 1, points.shape[0]))
    if knn.ndim == 1:
        knn = knn[:, None]
    score = np.zeros((points.shape[0],), dtype=np.float64)
    for i in range(points.shape[0]):
        nbr = knn[i, 1:] if knn.shape[1] > 1 else knn[i]
        dot_val = np.abs(normals[nbr] @ normals[i])
        score[i] = float(np.mean(1.0 - np.clip(dot_val, 0.0, 1.0)))
    p95 = float(np.percentile(score, 95)) if score.shape[0] > 4 else float(np.max(score))
    return np.clip(score / max(p95, 1e-6), 0.0, 1.0)


def _joint_color_geometry_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    params: TeaserLikeParams,
) -> np.ndarray:
    src = source.voxel_down_sample(5.0)
    tgt = target.voxel_down_sample(5.0)
    if len(src.points) == 0 or len(tgt.points) == 0:
        return init_transform.copy()

    src_xyz = np.asarray(src.points, dtype=np.float64)
    tgt_xyz = np.asarray(tgt.points, dtype=np.float64)
    src_rgb = np.asarray(src.colors, dtype=np.float64)
    tgt_rgb = np.asarray(tgt.colors, dtype=np.float64)
    if src_rgb.shape[0] != src_xyz.shape[0]:
        src_rgb = np.zeros((src_xyz.shape[0], 3), dtype=np.float64)
    if tgt_rgb.shape[0] != tgt_xyz.shape[0]:
        tgt_rgb = np.zeros((tgt_xyz.shape[0], 3), dtype=np.float64)

    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    src_n = np.asarray(src.normals, dtype=np.float64)
    src_curv = _normal_variation_score(src_xyz, src_n, k=12)
    src_curv_w = 0.3 + 0.7 * src_curv

    tree = cKDTree(tgt_xyz)
    transform = init_transform.copy()
    for max_corr in params.joint_schedule_mm:
        sigma_geo = max(max_corr * 0.45, 1.0)
        for _ in range(10):
            src_tf = (transform[:3, :3] @ src_xyz.T).T + transform[:3, 3]
            dist, idx = tree.query(src_tf, k=1, distance_upper_bound=max_corr)
            valid = np.isfinite(dist) & (idx < tgt_xyz.shape[0])
            if int(np.sum(valid)) < 30:
                break
            src_sel = src_tf[valid]
            tgt_sel = tgt_xyz[idx[valid]]
            geo = dist[valid]
            col = np.linalg.norm(src_rgb[valid] - tgt_rgb[idx[valid]], axis=1)
            w_geo = np.exp(-(geo * geo) / (2.0 * sigma_geo * sigma_geo + 1e-12))
            w_col = np.exp(-(col * col) / (2.0 * params.sigma_color * params.sigma_color + 1e-12))
            w_curv = src_curv_w[valid]
            delta = _weighted_kabsch(src_sel, tgt_sel, w_geo * (w_col ** params.color_weight_power) * w_curv)
            transform = delta @ transform
    return transform


def _colored_icp_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    params: TeaserLikeParams,
) -> np.ndarray:
    src = o3d.geometry.PointCloud(source)
    tgt = o3d.geometry.PointCloud(target)
    normal_radius = max(2.0 * params.colored_icp_mm, 6.0)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    res = o3d.pipelines.registration.registration_colored_icp(
        src,
        tgt,
        params.colored_icp_mm,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=params.colored_icp_iters),
    )
    return np.asarray(res.transformation, dtype=np.float64)


def _coarse_ransac_fpfh(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel_mm: float, max_corr_mm: float) -> np.ndarray:
    src_down, src_fpfh = _preprocess_for_global_registration(source, voxel_mm)
    tgt_down, tgt_fpfh = _preprocess_for_global_registration(target, voxel_mm)
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        max_corr_mm,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr_mm),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(80_000, 0.999),
    )
    return np.asarray(res.transformation, dtype=np.float64)


def _coarse_fgr_fpfh(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel_mm: float, max_corr_mm: float) -> np.ndarray:
    src_down, src_fpfh = _preprocess_for_global_registration(source, voxel_mm)
    tgt_down, tgt_fpfh = _preprocess_for_global_registration(target, voxel_mm)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr_mm,
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


def _color_residual(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray, dist_gate: float) -> float:
    src_xyz = np.asarray(source.points, dtype=np.float64)
    tgt_xyz = np.asarray(target.points, dtype=np.float64)
    src_rgb = np.asarray(source.colors, dtype=np.float64)
    tgt_rgb = np.asarray(target.colors, dtype=np.float64)
    if src_xyz.shape[0] == 0 or tgt_xyz.shape[0] == 0:
        return 1.0
    if src_rgb.shape[0] != src_xyz.shape[0] or tgt_rgb.shape[0] != tgt_xyz.shape[0]:
        return 1.0
    src_tf = (transform[:3, :3] @ src_xyz.T).T + transform[:3, 3]
    tree = cKDTree(tgt_xyz)
    dist, idx = tree.query(src_tf, k=1, distance_upper_bound=dist_gate)
    valid = np.isfinite(dist) & (idx < tgt_xyz.shape[0])
    if int(np.sum(valid)) < 30:
        return 1.0
    return float(np.mean(np.linalg.norm(src_rgb[valid] - tgt_rgb[idx[valid]], axis=1)))


def _score_candidate(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, t: np.ndarray, params: TeaserLikeParams) -> tuple[float, float, float, float, float]:
    strict_eval = o3d.pipelines.registration.evaluate_registration(source, target, params.strict_eval_mm, t)
    wide_eval = o3d.pipelines.registration.evaluate_registration(source, target, params.wide_eval_mm, t)
    color_res = _color_residual(source, target, t, dist_gate=params.wide_eval_mm)
    score = (
        1.4 * float(strict_eval.fitness)
        + 1.0 * float(wide_eval.fitness)
        - 0.08 * float(strict_eval.inlier_rmse)
        - 0.04 * float(wide_eval.inlier_rmse)
        - params.score_color_weight * color_res
    )
    return score, float(strict_eval.fitness), float(strict_eval.inlier_rmse), float(wide_eval.fitness), float(wide_eval.inlier_rmse), color_res


def teaser_like_color_coarse_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    noise_bound_mm: float = 6.0,
    cbar2: float = 1.0,
    params: TeaserLikeParams | None = None,
) -> TeaserLikeCoarseResult:
    cfg = TeaserLikeParams() if params is None else params
    if noise_bound_mm > 0:
        cfg = TeaserLikeParams(**{**cfg.__dict__, "noise_bound_mm": noise_bound_mm, "cbar2": cbar2})
    src_corr, dst_corr = _build_color_aware_correspondences(source, target, cfg)
    num_raw = src_corr.shape[1]

    src_tims, edge_map = _compute_tims(src_corr)
    dst_tims, _ = _compute_tims(dst_corr)
    used_noise = cfg.noise_bound_mm if cfg.noise_bound_mm > 0 else _estimate_noise_bound_from_tims(src_tims, dst_tims, cfg.cbar2)

    scale_mask = _scale_inlier_mask(src_tims, dst_tims, noise_bound=used_noise, cbar2=cfg.cbar2)
    adj = _build_adjacency(src_corr.shape[1], edge_map, scale_mask)
    inlier_nodes = _select_inlier_nodes(adj, cfg)
    if inlier_nodes.size < 3:
        raise RuntimeError("TEASER-like 图筛选后节点不足。")

    src_in = src_corr[:, inlier_nodes]
    dst_in = dst_corr[:, inlier_nodes]
    src_tims_in, _ = _compute_tims(src_in)
    dst_tims_in, _ = _compute_tims(dst_in)

    rot_noise = max(2.0 * used_noise, 1.0)
    r, _ = _gnc_tls_rotation(src_tims_in, dst_tims_in, noise_bound=rot_noise, params=cfg)
    trans, trans_inlier_mask = _tls_translation(r @ src_in, dst_in, noise_bound=used_noise, cbar2=cfg.cbar2)
    teaser_t = np.eye(4, dtype=np.float64)
    teaser_t[:3, :3] = r
    teaser_t[:3, 3] = trans

    candidates: list[tuple[str, np.ndarray]] = [("teaser_like", teaser_t)]
    voxel = float(np.clip(cfg.voxel_mm, 5.0, 15.0))
    try:
        candidates.append(("ransac_fpfh", _coarse_ransac_fpfh(source, target, voxel, cfg.max_corr_mm)))
    except Exception:
        pass
    try:
        candidates.append(("fgr_fpfh", _coarse_fgr_fpfh(source, target, voxel, cfg.max_corr_mm)))
    except Exception:
        pass

    best_name = ""
    best_t = np.eye(4, dtype=np.float64)
    best_metrics = None
    best_score = -1e18
    for name, t0 in candidates:
        t1 = _joint_color_geometry_refine(source, target, t0, cfg)
        t2 = _colored_icp_refine(source, target, t1, cfg)
        score, fit_s, rmse_s, fit_w, rmse_w, color_res = _score_candidate(source, target, t2, cfg)
        if score > best_score:
            best_score = score
            best_name = name
            best_t = t2
            best_metrics = (fit_s, rmse_s, fit_w, rmse_w, color_res)

    if best_metrics is None:
        raise RuntimeError("未获得有效候选结果。")
    fit_s, rmse_s, fit_w, rmse_w, color_res = best_metrics
    return TeaserLikeCoarseResult(
        transform=best_t,
        method_name=best_name,
        used_noise_bound_mm=float(used_noise),
        num_raw_matches=num_raw,
        num_filtered_matches=int(inlier_nodes.size),
        num_inlier_nodes=int(np.sum(trans_inlier_mask)),
        fitness_strict=fit_s,
        rmse_strict=rmse_s,
        fitness_wide=fit_w,
        rmse_wide=rmse_w,
        color_residual=color_res,
    )


def visualize_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray) -> None:
    raw_src = o3d.geometry.PointCloud(source)
    raw_tgt = o3d.geometry.PointCloud(target)
    reg_src = o3d.geometry.PointCloud(source)
    reg_tgt = o3d.geometry.PointCloud(target)
    reg_src.transform(transform)
    offset = np.array([0.0, 1000.0, 0.0], dtype=np.float64)
    reg_src.translate(offset)
    reg_tgt.translate(offset)

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("TEASER-like Coarse Registration", 1440, 900)
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
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
