from __future__ import annotations

import torch
import torch.nn.functional as F


class GeometryOps:
    """几何运算工具集合。"""

    @staticmethod
    def compute_normals(
        query_pts: torch.Tensor,
        neighbor_pts: torch.Tensor,
        orient: str = "origin",
        view_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_pts.ndim != 2 or query_pts.shape[1] < 3:
            raise ValueError(f"query_pts 形状必须是 (N, 3+)，实际为 {tuple(query_pts.shape)}")
        if neighbor_pts.ndim != 3 or neighbor_pts.shape[0] != query_pts.shape[0] or neighbor_pts.shape[2] < 3:
            raise ValueError(f"neighbor_pts 形状必须是 (N, K, 3+)，实际为 {tuple(neighbor_pts.shape)}")

        orient_key = orient.lower()
        if orient_key not in {"none", "origin", "centroid", "view_point"}:
            raise ValueError(f"orient 仅支持 none/origin/centroid/view_point，实际为 {orient!r}")

        query_xyz = query_pts[..., :3]
        neighbor_xyz = neighbor_pts[..., :3]
        centered = neighbor_xyz - query_xyz.unsqueeze(1)
        cov = torch.bmm(centered.transpose(1, 2), centered)
        cov.diagonal(dim1=-2, dim2=-1).add_(1e-5)

        try:
            _, eigenvecs = torch.linalg.eigh(cov)
            normals = eigenvecs[..., 0]
        except RuntimeError:
            return torch.tensor([0.0, 0.0, 1.0], device=query_pts.device).expand(query_pts.shape[0], 3)

        if orient_key == "none":
            return F.normalize(normals, dim=-1)

        if orient_key == "origin":
            ref = torch.zeros_like(query_xyz)
        elif orient_key == "centroid":
            ref = query_xyz.mean(dim=0, keepdim=True).expand_as(query_xyz)
        else:
            if view_point is None:
                raise ValueError("orient='view_point' 时必须提供 view_point")
            if view_point.ndim == 1:
                ref = view_point[:3].to(device=query_xyz.device, dtype=query_xyz.dtype).unsqueeze(0).expand_as(query_xyz)
            elif view_point.ndim == 2 and view_point.shape[0] == query_xyz.shape[0] and view_point.shape[1] >= 3:
                ref = view_point[:, :3].to(device=query_xyz.device, dtype=query_xyz.dtype)
            else:
                raise ValueError(f"view_point 形状必须是 (3,) 或 (N,3+)，实际为 {tuple(view_point.shape)}")

        to_ref = ref - query_xyz
        sign = torch.sign((normals * to_ref).sum(dim=-1, keepdim=True))
        sign[sign == 0] = 1.0
        return F.normalize(normals * sign, dim=-1)

    @staticmethod
    def compute_curvature(neighbor_pts: torch.Tensor) -> torch.Tensor:
        if neighbor_pts.ndim != 3 or neighbor_pts.shape[2] < 3:
            raise ValueError(f"neighbor_pts 形状必须是 (N,K,3+)，实际为 {tuple(neighbor_pts.shape)}")

        neighbor_xyz = neighbor_pts[..., :3]
        centered = neighbor_xyz - neighbor_xyz.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered)
        cov.diagonal(dim1=-2, dim2=-1).add_(1e-6)
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = torch.clamp_min(eigvals, 0.0)
        lam0 = eigvals[:, 0]
        lam_sum = eigvals.sum(dim=1).clamp_min(1e-12)
        return lam0 / lam_sum

