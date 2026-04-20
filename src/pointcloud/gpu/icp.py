from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .geometry import GeometryOps

if TYPE_CHECKING:
    from .index import GPUSpatialIndex


@dataclass(frozen=True)
class ICPPointToPointResult:
    transformation: torch.Tensor
    fitness: float
    rmse: float
    iterations: int
    converged: bool
    correspondence_count: int


@dataclass(frozen=True)
class ICPAdaptiveCurvatureResult:
    transformation: torch.Tensor
    fitness: float
    rmse: float
    iterations: int
    converged: bool
    correspondence_count: int
    mean_curvature_weight: float
    mean_plane_residual: float


def icp_point_to_point(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_index: "GPUSpatialIndex",
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    max_correspondence_distance: float | None = None,
    init_transform: torch.Tensor | None = None,
) -> ICPPointToPointResult:
    device = source_xyz.device
    dtype = source_xyz.dtype

    transform = torch.eye(4, device=device, dtype=dtype)
    if init_transform is not None:
        transform = init_transform.to(device=device, dtype=dtype).clone()

    prev_rmse = float("inf")
    converged = False
    final_fitness = 0.0
    final_corr_count = 0
    rmse = float("inf")
    radius2 = None if max_correspondence_distance is None else float(max_correspondence_distance) ** 2

    for step in range(1, int(max_iterations) + 1):
        src_w = _apply_transform(source_xyz, transform)
        d2, idx = target_index.search(src_w, k=1, return_torch=True)
        d2 = d2[:, 0]
        idx = idx[:, 0]

        if radius2 is None:
            inlier_mask = torch.ones_like(d2, dtype=torch.bool)
        else:
            inlier_mask = d2 <= radius2

        corr_count = int(inlier_mask.sum().item())
        final_corr_count = corr_count
        if corr_count < 3:
            break

        src_corr = src_w[inlier_mask]
        tgt_corr = target_xyz[idx[inlier_mask]]
        delta = _estimate_weighted_rigid_transform(
            src_pts=src_corr,
            tgt_pts=tgt_corr,
            weights=torch.ones((src_corr.shape[0],), device=device, dtype=dtype),
        )
        transform = delta @ transform

        rmse = float(torch.sqrt(torch.mean(d2[inlier_mask])).item())
        final_fitness = float(corr_count / max(1, source_xyz.shape[0]))
        if abs(prev_rmse - rmse) <= float(tolerance):
            converged = True
            return ICPPointToPointResult(
                transformation=transform,
                fitness=final_fitness,
                rmse=rmse,
                iterations=step,
                converged=converged,
                correspondence_count=final_corr_count,
            )
        prev_rmse = rmse

    return ICPPointToPointResult(
        transformation=transform,
        fitness=final_fitness,
        rmse=rmse,
        iterations=int(max_iterations),
        converged=converged,
        correspondence_count=final_corr_count,
    )


def icp_adaptive_curvature(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_index: "GPUSpatialIndex",
    max_iterations: int = 40,
    tolerance: float = 1e-5,
    max_correspondence_distance: float | None = None,
    init_transform: torch.Tensor | None = None,
    normal_k: int = 20,
    curvature_k: int = 20,
    curvature_power: float = 1.0,
    curvature_min_weight: float = 0.20,
    trim_ratio: float = 0.10,
    huber_delta: float = 0.80,
    lm_lambda: float = 1e-4,
    step_scale: float = 0.35,
) -> ICPAdaptiveCurvatureResult:
    """曲率自适应 point-to-plane ICP。

    1. 先在 target 上估计法线和曲率。
    2. 迭代时依据曲率与平面残差做权重，执行加权 point-to-plane 线性化更新。
    3. 更新退化时回退到加权 point-to-point SVD。
    """
    device = source_xyz.device
    dtype = source_xyz.dtype
    transform = torch.eye(4, device=device, dtype=dtype)
    if init_transform is not None:
        transform = init_transform.to(device=device, dtype=dtype).clone()

    radius2 = None if max_correspondence_distance is None else float(max_correspondence_distance) ** 2
    prev_rmse = float("inf")
    final_fitness = 0.0
    final_rmse = float("inf")
    final_corr_count = 0
    mean_curv_w = 0.0
    mean_plane_res = 0.0
    converged = False

    normal_k_eff = int(max(6, normal_k))
    curvature_k_eff = int(max(6, curvature_k))
    target_normals, target_curv = _estimate_target_normals_curvature(
        target_xyz=target_xyz,
        target_index=target_index,
        normal_k=normal_k_eff,
        curvature_k=curvature_k_eff,
    )

    for step in range(1, int(max_iterations) + 1):
        src_w = _apply_transform(source_xyz, transform)
        d2, idx = target_index.search(src_w, k=1, return_torch=True)
        d2 = d2[:, 0]
        idx = idx[:, 0]
        tgt_corr = target_xyz[idx]
        normal_corr = target_normals[idx]
        curv_corr = target_curv[idx]

        if radius2 is None:
            mask = torch.ones_like(d2, dtype=torch.bool)
        else:
            mask = d2 <= radius2
        if int(mask.sum().item()) < 6:
            break

        src_sel = src_w[mask]
        tgt_sel = tgt_corr[mask]
        n_sel = normal_corr[mask]
        d2_sel = d2[mask]
        curv_sel = curv_corr[mask]
        plane_residual = torch.abs(torch.sum(n_sel * (src_sel - tgt_sel), dim=1))

        # 修剪最差残差，提升对离群点与错误对应的稳定性。
        if 0.0 < float(trim_ratio) < 0.8 and src_sel.shape[0] >= 16:
            keep_n = int(max(8, round(src_sel.shape[0] * (1.0 - float(trim_ratio)))))
            keep_n = min(keep_n, src_sel.shape[0])
            keep_idx = torch.topk(plane_residual, k=keep_n, largest=False, sorted=False).indices
            src_sel = src_sel[keep_idx]
            tgt_sel = tgt_sel[keep_idx]
            n_sel = n_sel[keep_idx]
            d2_sel = d2_sel[keep_idx]
            curv_sel = curv_sel[keep_idx]
            plane_residual = plane_residual[keep_idx]

        curv_w = _curvature_weights(
            curv_sel,
            power=float(curvature_power),
            min_weight=float(curvature_min_weight),
        )
        robust_w = _huber_weights(plane_residual, delta=max(1e-6, float(huber_delta)))
        weights = torch.clamp(curv_w * robust_w, min=1e-4)

        delta = _solve_point_to_plane_step(
            src_pts=src_sel,
            tgt_pts=tgt_sel,
            normals=n_sel,
            weights=weights,
            lm_lambda=float(lm_lambda),
            step_scale=float(step_scale),
        )
        if delta is None:
            # 退化时回退到加权点到点，避免直接失败。
            delta = _estimate_weighted_rigid_transform(src_pts=src_sel, tgt_pts=tgt_sel, weights=weights)
        transform = delta @ transform

        corr_count = int(src_sel.shape[0])
        final_corr_count = corr_count
        final_fitness = float(corr_count / max(1, source_xyz.shape[0]))
        final_rmse = float(torch.sqrt(torch.mean(d2_sel)).item())
        mean_curv_w = float(torch.mean(curv_w).item())
        mean_plane_res = float(torch.mean(plane_residual).item())

        if abs(prev_rmse - final_rmse) <= float(tolerance):
            converged = True
            return ICPAdaptiveCurvatureResult(
                transformation=transform,
                fitness=final_fitness,
                rmse=final_rmse,
                iterations=step,
                converged=converged,
                correspondence_count=final_corr_count,
                mean_curvature_weight=mean_curv_w,
                mean_plane_residual=mean_plane_res,
            )
        prev_rmse = final_rmse

    return ICPAdaptiveCurvatureResult(
        transformation=transform,
        fitness=final_fitness,
        rmse=final_rmse,
        iterations=int(max_iterations),
        converged=converged,
        correspondence_count=final_corr_count,
        mean_curvature_weight=mean_curv_w,
        mean_plane_residual=mean_plane_res,
    )


def _estimate_target_normals_curvature(
    target_xyz: torch.Tensor,
    target_index: "GPUSpatialIndex",
    normal_k: int,
    curvature_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_need = int(max(normal_k, curvature_k) + 1)
    _, idx = target_index.search(target_xyz, k=k_need, return_torch=True)
    idx = idx.clamp_min(0)
    if idx.shape[1] > 1:
        idx = idx[:, 1:]
    neigh = target_xyz[idx]  # (N, K, 3)
    normals = GeometryOps.compute_normals(
        query_pts=target_xyz,
        neighbor_pts=neigh[:, :normal_k, :],
        orient="centroid",
    )
    curv = GeometryOps.compute_curvature(neigh[:, :curvature_k, :])
    return normals, curv


def _curvature_weights(curvature: torch.Tensor, power: float, min_weight: float) -> torch.Tensor:
    c = torch.clamp_min(curvature, 0.0)
    c_min = torch.min(c)
    c_max = torch.max(c)
    c_norm = (c - c_min) / (c_max - c_min + 1e-6)
    w = torch.pow(torch.clamp(c_norm, min=0.0, max=1.0), max(0.1, float(power)))
    return torch.clamp(min_weight + (1.0 - min_weight) * w, min=min_weight, max=1.0)


def _huber_weights(abs_residual: torch.Tensor, delta: float) -> torch.Tensor:
    ratio = delta / torch.clamp(abs_residual, min=delta)
    return torch.clamp(ratio, min=1e-3, max=1.0)


def _solve_point_to_plane_step(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    normals: torch.Tensor,
    weights: torch.Tensor,
    lm_lambda: float,
    step_scale: float,
) -> torch.Tensor | None:
    # r = n^T (s - t), J = [s x n, n]
    cross = torch.cross(src_pts, normals, dim=1)
    j = torch.cat([cross, normals], dim=1)  # (N, 6)
    r = torch.sum(normals * (src_pts - tgt_pts), dim=1, keepdim=True)  # (N,1)
    w_sqrt = torch.sqrt(torch.clamp_min(weights, 1e-8)).unsqueeze(1)
    jw = j * w_sqrt
    rw = r * w_sqrt

    h = jw.transpose(0, 1) @ jw
    if lm_lambda > 0:
        h = h + torch.eye(6, device=h.device, dtype=h.dtype) * float(lm_lambda)
    g = jw.transpose(0, 1) @ rw
    try:
        xi = torch.linalg.solve(h, -g).reshape(6)
    except RuntimeError:
        return None
    if not torch.isfinite(xi).all():
        return None
    xi = xi * max(1e-3, float(step_scale))
    return _se3_exp(xi.to(device=src_pts.device, dtype=src_pts.dtype))


def _se3_exp(xi: torch.Tensor) -> torch.Tensor:
    w = xi[:3]
    v = xi[3:]
    theta = torch.linalg.norm(w)
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype)
    wx = _skew(w)

    if float(theta.item()) < 1e-10:
        r = eye3 + wx
        v_mat = eye3 + 0.5 * wx
    else:
        a = torch.sin(theta) / theta
        b = (1.0 - torch.cos(theta)) / (theta * theta)
        c = (theta - torch.sin(theta)) / (theta * theta * theta)
        wx2 = wx @ wx
        r = eye3 + a * wx + b * wx2
        v_mat = eye3 + b * wx + c * wx2

    t = v_mat @ v
    out = torch.eye(4, device=xi.device, dtype=xi.dtype)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def _skew(v: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, -float(v[2].item()), float(v[1].item())],
            [float(v[2].item()), 0.0, -float(v[0].item())],
            [-float(v[1].item()), float(v[0].item()), 0.0],
        ],
        device=v.device,
        dtype=v.dtype,
    )


def _apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return points @ rot.transpose(0, 1) + trans.unsqueeze(0)


def _estimate_weighted_rigid_transform(src_pts: torch.Tensor, tgt_pts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = torch.clamp_min(weights, 1e-8).reshape(-1, 1)
    w_sum = torch.sum(w).clamp_min(1e-8)
    src_centroid = torch.sum(src_pts * w, dim=0) / w_sum
    tgt_centroid = torch.sum(tgt_pts * w, dim=0) / w_sum
    src_centered = src_pts - src_centroid
    tgt_centered = tgt_pts - tgt_centroid

    h = (src_centered * w).transpose(0, 1) @ tgt_centered
    u, _, v_t = torch.linalg.svd(h)
    r = v_t.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(r) < 0:
        v_t = v_t.clone()
        v_t[-1, :] *= -1
        r = v_t.transpose(0, 1) @ u.transpose(0, 1)
    t = tgt_centroid - r @ src_centroid

    transform = torch.eye(4, device=src_pts.device, dtype=src_pts.dtype)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform
