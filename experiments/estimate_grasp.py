#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心流程：
1. RGB 自动检测长方形开口中心；
2. 将 PCD 投影到图像平面；
3. 只保留开口附近、且在 RGB 上属于暗色料盘区域的局部点云；
4. 对局部点云拟合开口所在的局部平面；
5. 开口中心像素射线与局部平面求交，得到 3D 抓取点；
6. 构造相机坐标系下的 T_camera_grasp。
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


@dataclass
class PCDData:
    xyz: np.ndarray
    rgb: np.ndarray | None = None


@dataclass
class OpeningDetection:
    center_uv: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    score: float


@dataclass
class PlaneResult:
    normal: np.ndarray
    d: float
    inlier_points: np.ndarray
    inlier_colors: np.ndarray | None
    point_count: int


@dataclass
class GraspResult:
    T_camera_grasp: np.ndarray
    grasp_point: np.ndarray
    pre_grasp_point: np.ndarray
    front_normal_toward_camera: np.ndarray
    opening_center_uv: np.ndarray
    opening_bbox_xywh: tuple[int, int, int, int]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise RuntimeError(f"向量归一化失败，norm={n}")
    return v / n


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def infer_length_unit_params(xyz: np.ndarray) -> dict[str, float]:
    """根据点云 Z 的量级自动推断单位：毫米或米。"""
    z = np.asarray(xyz)[:, 2]
    z = z[np.isfinite(z) & (z > 0)]
    if z.size == 0:
        return {"plane_dist": 0.003, "voxel": 0.0015, "approach_dist": 0.08, "axis_size": 0.08, "sphere_r": 0.008}

    med_z = float(np.median(z))
    if med_z > 10.0:
        return {"plane_dist": 3.0, "voxel": 1.2, "approach_dist": 80.0, "axis_size": 40.0, "sphere_r": 8.0}
    return {"plane_dist": 0.003, "voxel": 0.0012, "approach_dist": 0.08, "axis_size": 0.04, "sphere_r": 0.008}


def _decode_packed_rgb(value: float) -> tuple[int, int, int]:
    """兼容 PCL 常见 packed rgb float。"""
    try:
        i = struct.unpack("I", struct.pack("f", float(value)))[0]
    except Exception:
        i = int(value)
    r = (i >> 16) & 255
    g = (i >> 8) & 255
    b = i & 255
    return r, g, b


def read_pcd_ascii(path: Path) -> PCDData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCD 文件不存在：{path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    fields: list[str] = []
    data_idx = -1

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("FIELDS "):
            fields = s.split()[1:]
        elif up == "DATA ASCII":
            data_idx = i + 1
            break

    if data_idx < 0:
        raise RuntimeError("当前脚本只支持 DATA ASCII 格式的 PCD")
    if not fields:
        raise RuntimeError("PCD 缺少 FIELDS")

    rows: list[list[float]] = []
    for line in lines[data_idx:]:
        s = line.strip()
        if not s:
            continue
        rows.append([float(x) for x in s.split()])

    if not rows:
        return PCDData(np.empty((0, 3), dtype=np.float64), None)

    data = np.asarray(rows, dtype=np.float64)
    f2i = {name.lower(): i for i, name in enumerate(fields)}

    if not {"x", "y", "z"}.issubset(f2i):
        raise RuntimeError(f"PCD 缺少 x/y/z 字段，当前字段：{fields}")

    xyz = data[:, [f2i["x"], f2i["y"], f2i["z"]]].astype(np.float64)

    rgb = None
    if {"r", "g", "b"}.issubset(f2i):
        rgb = data[:, [f2i["r"], f2i["g"], f2i["b"]]]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    elif "rgb" in f2i:
        packed = data[:, f2i["rgb"]]
        rgb = np.asarray([_decode_packed_rgb(v) for v in packed], dtype=np.uint8)

    return PCDData(xyz=xyz, rgb=rgb)


def make_o3d_cloud(xyz: np.ndarray, rgb: np.ndarray | None = None):
    if o3d is None:
        raise RuntimeError("需要安装 open3d: pip install open3d")

    pcd = o3d.geometry.PointCloud()
    xyz = np.asarray(xyz, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if rgb is not None and len(rgb) == len(xyz):
        colors = np.asarray(rgb, dtype=np.float64)
        if colors.size > 0 and colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    return pcd


def auto_search_roi(image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    自动搜索料盘前端开口所在区域。
    只搜索图像下半部中间区域，避免上表面孔阵列和背景误检。
    """
    h, w = image_shape[:2]
    return int(0.24 * w), int(0.62 * h), int(0.76 * w), int(0.90 * h)


def _score_opening_candidate(
    *,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    dark_ratio: float,
    rectangularness: float,
) -> float:
    h, w = image_shape
    x, y, bw, bh = bbox
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    aspect = bw / max(float(bh), 1.0)
    wr = bw / float(w)
    hr = bh / float(h)

    center_score = 1.0 - min(abs(cx - 0.5 * w) / (0.5 * w), 1.0)
    y_score = float(np.exp(-(((cy / h - 0.79) / 0.11) ** 2)))
    aspect_score = float(np.exp(-abs(np.log(max(aspect, 1e-6) / 6.5))))
    width_score = float(np.exp(-abs(np.log(max(wr, 1e-6) / 0.16))))
    height_score = float(np.exp(-abs(np.log(max(hr, 1e-6) / 0.035))))

    return (
        2.4 * dark_ratio
        + 1.8 * rectangularness
        + 1.6 * aspect_score
        + 1.3 * width_score
        + 1.1 * height_score
        + 1.2 * center_score
        + 1.5 * y_score
    )


def detect_rect_opening_auto(rgb_bgr: np.ndarray, debug_dir: Path) -> OpeningDetection:
    """
    自动检测长方形开口中心。
    输出的 center_uv 直接作为抓取点像素中心，因此深蓝色球会落在开口中心。
    """
    ensure_dir(debug_dir)

    h, w = rgb_bgr.shape[:2]
    x1, y1, x2, y2 = auto_search_roi(rgb_bgr.shape)
    roi = rgb_bgr[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    percentiles = np.percentile(gray_blur, [3, 5, 8, 12, 16, 20, 25, 30, 35])
    base_thresholds = [45, 60, 75, 90, 105, 120, 140]
    thresholds = sorted(set(int(np.clip(t, 25, 180)) for t in list(percentiles) + base_thresholds))

    candidates: list[dict] = []

    for thr in thresholds:
        mask = (gray_blur <= thr).astype(np.uint8) * 255

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, roi.shape[1] // 45), 3))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, open_kernel)

        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue

            gx, gy = x1 + bx, y1 + by
            area = bw * bh
            aspect = bw / max(float(bh), 1.0)
            wr = bw / float(w)
            hr = bh / float(h)
            contour_area = float(cv2.contourArea(cnt))
            rectangularness = contour_area / max(float(area), 1.0)

            if aspect < 2.5 or aspect > 18.0:
                continue
            if wr < 0.045 or wr > 0.35:
                continue
            if hr < 0.010 or hr > 0.105:
                continue
            if area < w * h * 0.00012 or area > w * h * 0.025:
                continue

            patch = gray[by : by + bh, bx : bx + bw]
            dark_ratio = float(np.mean(patch <= thr)) if patch.size else 0.0
            score = _score_opening_candidate(
                bbox=(gx, gy, bw, bh),
                image_shape=(h, w),
                dark_ratio=dark_ratio,
                rectangularness=rectangularness,
            )
            candidates.append(
                {
                    "score": float(score),
                    "bbox": (int(gx), int(gy), int(bw), int(bh)),
                    "thr": int(thr),
                    "aspect": float(aspect),
                    "dark_ratio": float(dark_ratio),
                    "rectangularness": float(rectangularness),
                }
            )

    if not candidates:
        vis_fail = rgb_bgr.copy()
        cv2.rectangle(vis_fail, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(str(debug_dir / "rgb_opening_detection_failed.png"), vis_fail)
        raise RuntimeError("未自动检测到长方形开口。请查看 debug 图，确认相机是否正对料盘前端。")

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    gx, gy, bw, bh = best["bbox"]

    center_uv = np.array([gx + bw / 2.0, gy + bh / 2.0], dtype=np.float64)

    vis = rgb_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for cand in candidates[:8]:
        x, y, ww, hh = cand["bbox"]
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), (255, 0, 0), 1)
    cv2.rectangle(vis, (gx, gy), (gx + bw, gy + bh), (0, 0, 255), 2)
    cv2.circle(vis, (int(center_uv[0]), int(center_uv[1])), 6, (0, 255, 0), -1)
    txt = f"center=({center_uv[0]:.1f},{center_uv[1]:.1f}) bbox=({gx},{gy},{bw},{bh}) score={best['score']:.2f}"
    cv2.putText(vis, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / "rgb_opening_detection.png"), vis)

    (debug_dir / "opening_candidates.json").write_text(
        json.dumps(candidates[:20], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return OpeningDetection(center_uv=center_uv, bbox_xywh=(gx, gy, bw, bh), score=float(best["score"]))


def project_points_to_image(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    z = xyz[:, 2]
    valid = np.isfinite(z) & (z > 1e-9)

    uv = np.full((xyz.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * xyz[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * xyz[valid, 1] / z[valid] + K[1, 2]
    return uv


def _local_bbox_from_opening(
    opening: OpeningDetection, image_shape: tuple[int, int, int], scale_x: float, scale_y: float
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    gx, gy, bw, bh = opening.bbox_xywh
    cx = gx + bw / 2.0
    cy = gy + bh / 2.0
    half_w = max(scale_x * bw / 2.0, 0.055 * w)
    half_h = max(scale_y * bh / 2.0, 0.040 * h)
    x1 = int(max(0, cx - half_w))
    x2 = int(min(w - 1, cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(h - 1, cy + half_h))
    return x1, y1, x2, y2


def filter_local_points_by_opening(
    pcd: PCDData, rgb_bgr: np.ndarray, K: np.ndarray, opening: OpeningDetection, debug_dir: Path
) -> PCDData:
    """
    只保留开口周围小范围内的点云。
    红色内点会集中在开口附近，抓取点保持居中。
    """
    ensure_dir(debug_dir)
    xyz = pcd.xyz
    uv = project_points_to_image(xyz, K)
    h, w = rgb_bgr.shape[:2]

    valid = np.isfinite(uv).all(axis=1)
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] < w
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] < h
    valid &= np.isfinite(xyz).all(axis=1)
    valid &= xyz[:, 2] > 0

    chosen_mask = None
    chosen_roi = None

    for scale_x, scale_y in [(2.6, 3.0), (3.6, 4.0), (4.8, 5.0)]:
        x1, y1, x2, y2 = _local_bbox_from_opening(opening, rgb_bgr.shape, scale_x, scale_y)
        mask = valid.copy()
        mask &= uv[:, 0] >= x1
        mask &= uv[:, 0] <= x2
        mask &= uv[:, 1] >= y1
        mask &= uv[:, 1] <= y2

        if pcd.rgb is not None and int(mask.sum()) > 30:
            intensity = np.max(pcd.rgb[mask].astype(np.float32), axis=1)
            thr = float(np.clip(np.percentile(intensity, 78), 70, 185))
            idx = np.where(mask)[0]
            keep = intensity <= thr
            mask2 = np.zeros_like(mask)
            mask2[idx[keep]] = True
            mask = mask2

        chosen_mask = mask
        chosen_roi = (x1, y1, x2, y2)
        if int(mask.sum()) >= 80:
            break

    assert chosen_mask is not None and chosen_roi is not None

    xyz_f = xyz[chosen_mask]
    rgb_f = pcd.rgb[chosen_mask] if pcd.rgb is not None else None

    if len(xyz_f) < 40:
        raise RuntimeError(f"开口附近局部点云过少：{len(xyz_f)}。请确认 RGB/PCD 是否对齐，且使用的是 depth 内参。")

    vis = rgb_bgr.copy()
    x1, y1, x2, y2 = chosen_roi
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    gx, gy, bw, bh = opening.bbox_xywh
    cv2.rectangle(vis, (gx, gy), (gx + bw, gy + bh), (0, 0, 255), 2)
    cv2.circle(vis, (int(opening.center_uv[0]), int(opening.center_uv[1])), 6, (0, 255, 0), -1)

    idx = np.where(chosen_mask)[0]
    if len(idx) > 2500:
        idx = idx[:: max(1, len(idx) // 2500)]
    for k in idx:
        u, v = int(round(uv[k, 0])), int(round(uv[k, 1]))
        if 0 <= u < w and 0 <= v < h:
            vis[v, u] = (0, 0, 255)
    cv2.imwrite(str(debug_dir / "local_points_projection.png"), vis)

    print(f"[INFO] centered local opening points: {len(xyz_f)}, roi={chosen_roi}")
    return PCDData(xyz=xyz_f, rgb=rgb_f)


def estimate_local_plane(pcd_local: PCDData, params: dict[str, float]) -> PlaneResult:
    if o3d is None:
        raise RuntimeError("需要安装 open3d: pip install open3d")

    cloud = make_o3d_cloud(pcd_local.xyz, pcd_local.rgb)

    voxel = float(params["voxel"])
    if voxel > 0 and len(cloud.points) > 500:
        cloud = cloud.voxel_down_sample(voxel)

    if len(cloud.points) >= 300:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(cloud.points) < 50:
        raise RuntimeError(f"局部点云滤波后点数过少：{len(cloud.points)}")

    plane_dist = float(params["plane_dist"])
    model, inliers = cloud.segment_plane(distance_threshold=plane_dist, ransac_n=3, num_iterations=2000)

    if len(inliers) < max(30, int(0.10 * len(cloud.points))):
        raise RuntimeError(
            f"局部平面内点过少：{len(inliers)} / {len(cloud.points)}。可能是 RGB/PCD 未对齐或开口附近点云缺失。"
        )

    pts = np.asarray(cloud.points)[inliers]
    cols = None
    if cloud.has_colors():
        cols = (np.asarray(cloud.colors)[inliers] * 255).clip(0, 255).astype(np.uint8)

    n = np.asarray(model[:3], dtype=np.float64)
    d = float(model[3])
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise RuntimeError("平面法向异常")

    n = n / norm
    d = d / norm

    camera_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if np.dot(n, camera_dir) < 0:
        n = -n
        d = -d

    print(f"[INFO] local plane normal={n}, d={d:.6f}, inliers={len(inliers)}")
    return PlaneResult(normal=n, d=d, inlier_points=pts, inlier_colors=cols, point_count=len(inliers))


def pixel_ray_intersect_plane(u: float, v: float, K: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    ray = np.array([(float(u) - K[0, 2]) / K[0, 0], (float(v) - K[1, 2]) / K[1, 1], 1.0], dtype=np.float64)
    ray = normalize(ray)

    n = normalize(n)
    denom = float(np.dot(n, ray))
    if abs(denom) < 1e-9:
        raise RuntimeError("像素射线与局部平面近似平行，无法求交")

    t = -float(d) / denom
    if t <= 0:
        raise RuntimeError(f"射线和平面交点异常，t={t:.6f}。请检查内参或点云坐标系。")

    return t * ray


def compute_opening_x_axis(opening: OpeningDetection, K: np.ndarray, plane: PlaneResult) -> np.ndarray:
    x, y, w, h = opening.bbox_xywh
    v_mid = y + h / 2.0

    p_left = pixel_ray_intersect_plane(x, v_mid, K, plane.normal, plane.d)
    p_right = pixel_ray_intersect_plane(x + w, v_mid, K, plane.normal, plane.d)
    return normalize(p_right - p_left)


def compute_grasp(
    opening: OpeningDetection, K: np.ndarray, plane: PlaneResult, params: dict[str, float]
) -> GraspResult:
    u, v = opening.center_uv
    grasp_point = pixel_ray_intersect_plane(u, v, K, plane.normal, plane.d)

    z_axis = normalize(plane.normal)
    x_axis = compute_opening_x_axis(opening, K, plane)

    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = grasp_point

    pre_grasp = grasp_point + z_axis * float(params["approach_dist"])

    return GraspResult(
        T_camera_grasp=T,
        grasp_point=grasp_point,
        pre_grasp_point=pre_grasp,
        front_normal_toward_camera=z_axis,
        opening_center_uv=opening.center_uv,
        opening_bbox_xywh=opening.bbox_xywh,
    )


def save_result(result: GraspResult, plane: PlaneResult, output: Path) -> None:
    ensure_dir(output.parent)
    data = {
        "unit": "same_as_pcd",
        "T_camera_grasp": result.T_camera_grasp.tolist(),
        "grasp_point": result.grasp_point.tolist(),
        "pre_grasp_point": result.pre_grasp_point.tolist(),
        "front_normal_toward_camera": result.front_normal_toward_camera.tolist(),
        "opening_center_uv": result.opening_center_uv.tolist(),
        "opening_bbox_xywh": list(result.opening_bbox_xywh),
        "plane": {"normal": plane.normal.tolist(), "d": float(plane.d), "inlier_count": int(plane.point_count)},
    }
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def visualize(pcd_local: PCDData, plane: PlaneResult, result: GraspResult, params: dict[str, float]) -> None:
    if o3d is None:
        print("[WARN] 未安装 open3d，跳过可视化")
        return

    geoms = []

    local_cloud = make_o3d_cloud(pcd_local.xyz, pcd_local.rgb)
    geoms.append(local_cloud)

    red = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(plane.inlier_points), 1))
    plane_cloud = make_o3d_cloud(plane.inlier_points, red)
    geoms.append(plane_cloud)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(params["axis_size"]), origin=result.grasp_point.tolist()
    )
    frame.rotate(result.T_camera_grasp[:3, :3], center=result.grasp_point.tolist())
    geoms.append(frame)

    r = float(params["sphere_r"])
    grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    grasp_sphere.translate(result.grasp_point.tolist())
    grasp_sphere.paint_uniform_color([0.0, 0.3, 1.0])
    geoms.append(grasp_sphere)

    pre_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    pre_sphere.translate(result.pre_grasp_point.tolist())
    pre_sphere.paint_uniform_color([0.0, 0.8, 1.0])
    geoms.append(pre_sphere)

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(np.vstack([result.grasp_point, result.pre_grasp_point]))
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    line.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.8, 1.0]], dtype=np.float64))
    geoms.append(line)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(params["axis_size"]) * 2.0, origin=[0.0, 0.0, 0.0]
    )
    geoms.append(origin)

    print("[INFO] Open3D 可视化说明：")
    print("       红色点云：局部拟合平面内点")
    print("       蓝色球：开口中心抓取点")
    print("       青色球：pre-grasp 预抓取点")
    o3d.visualization.draw_geometries(geoms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动估计料盘长方形开口中心抓取位姿：RGB + PCD + 内参")
    parser.add_argument("--rgb", type=Path, required=True, help="RGB 图像路径")
    parser.add_argument("--pcd", type=Path, required=True, help="ASCII PCD 点云路径")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("grasp_result_centered.json"))
    parser.add_argument("--debug-dir", type=Path, default=Path("debug_grasp_centered"))
    parser.add_argument("--vis", action="store_true", help="显示 Open3D 可视化窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rgb = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"无法读取 RGB 图像：{args.rgb}")

    pcd = read_pcd_ascii(args.pcd)
    if len(pcd.xyz) == 0:
        raise RuntimeError("PCD 点云为空")

    K = intrinsic_matrix(args.fx, args.fy, args.cx, args.cy)
    params = infer_length_unit_params(pcd.xyz)

    ensure_dir(args.debug_dir)

    print(f"[INFO] RGB: {args.rgb}, shape={rgb.shape}")
    print(f"[INFO] PCD: {args.pcd}, points={len(pcd.xyz)}, has_rgb={pcd.rgb is not None}")
    print(f"[INFO] auto params: {params}")

    opening = detect_rect_opening_auto(rgb, args.debug_dir)
    print(f"[INFO] opening center_uv={opening.center_uv}, bbox={opening.bbox_xywh}, score={opening.score:.3f}")

    pcd_local = filter_local_points_by_opening(pcd=pcd, rgb_bgr=rgb, K=K, opening=opening, debug_dir=args.debug_dir)
    plane = estimate_local_plane(pcd_local, params)
    result = compute_grasp(opening, K, plane, params)

    save_result(result, plane, args.output)

    print("\n========== RESULT ==========")
    print("T_camera_grasp:")
    print(result.T_camera_grasp)
    print(f"grasp_point: {result.grasp_point}")
    print(f"pre_grasp_point: {result.pre_grasp_point}")
    print(f"front_normal_toward_camera: {result.front_normal_toward_camera}")
    print(f"opening_center_uv: {result.opening_center_uv}")
    print(f"opening_bbox_xywh: {result.opening_bbox_xywh}")
    print(f"[INFO] saved result: {args.output}")
    print(f"[INFO] debug images: {args.debug_dir}")

    if args.vis:
        visualize(pcd_local, plane, result, params)


if __name__ == "__main__":
    main()
