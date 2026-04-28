from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.pointcloud import test_orbbec_realtime_plane_segmentation_zero_shot as zs

from src.rgbd_camera import Gemini305, SessionOptions

# region 默认参数（优先在这里直接改）
DEFAULT_PRIOR_PLY = PROJECT_ROOT / "experiments" / "real_pallet_cad" / "MaterialPlate.ply"  # 先验点云路径
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "stage2_direct_registration"  # 输出目录
DEFAULT_DEVICE = zs.DEFAULT_DEVICE  # 推理设备
DEFAULT_PROXY_URL = zs.DEFAULT_PROXY_URL  # 下载代理
DEFAULT_HF_CACHE_DIR = zs.DEFAULT_HF_CACHE_DIR  # HuggingFace 本地缓存目录
DEFAULT_HF_LOCAL_FILES_ONLY = zs.DEFAULT_HF_LOCAL_FILES_ONLY  # 是否仅使用本地缓存
DEFAULT_PROMPT = zs.DEFAULT_PROMPT  # 检测提示词
DEFAULT_TARGET_KEYWORDS = zs.DEFAULT_TARGET_KEYWORDS  # 目标关键词
DEFAULT_STRICT_TARGET_FILTER = zs.DEFAULT_STRICT_TARGET_FILTER  # 严格关键词过滤
DEFAULT_MAX_TARGETS = 1  # 本测试仅保留一个主目标
DEFAULT_USE_SAM = zs.DEFAULT_USE_SAM  # 是否启用 SAM
DEFAULT_BOX_THRESHOLD = zs.DEFAULT_BOX_THRESHOLD  # 检测框阈值
DEFAULT_TEXT_THRESHOLD = zs.DEFAULT_TEXT_THRESHOLD  # 文本阈值
DEFAULT_MIN_TARGET_CONF = zs.DEFAULT_MIN_TARGET_CONF  # 最小目标置信度
DEFAULT_TOPK_OBJECTS = 2  # 检测保留上限
DEFAULT_SAM_MAX_BOXES = 1  # 进入 SAM 的框数量
DEFAULT_SAM_PRIMARY_ONLY = True  # 仅主目标默认走 SAM
DEFAULT_SAM_SECONDARY_CONF_THRESHOLD = zs.DEFAULT_SAM_SECONDARY_CONF_THRESHOLD  # 次目标 SAM 阈值
DEFAULT_MIN_MASK_PIXELS = zs.DEFAULT_MIN_MASK_PIXELS  # 最小掩码像素
DEFAULT_MASK_IOU_SUPPRESS = zs.DEFAULT_MASK_IOU_SUPPRESS  # 掩码 IoU 抑制
DEFAULT_DETECT_MAX_SIDE = zs.DEFAULT_DETECT_MAX_SIDE  # 检测缩放边长
DEFAULT_COMBINE_PROMPTS_FORWARD = zs.DEFAULT_COMBINE_PROMPTS_FORWARD  # 提示词合并前向
DEFAULT_DETECT_INTERVAL = 4  # 每 N 帧检测一次
DEFAULT_TIMEOUT_MS = 120  # 取帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_FRAMES = 99999  # 最大采样帧数
DEFAULT_MIN_OBJECT_POINTS = 800  # 分割后最小点数，单位 点
DEFAULT_SEGMENT_VOXEL_MM = 2.0  # 观测点云降采样体素，单位 mm
DEFAULT_PRIOR_VOXEL_MM = 2.0  # 先验点云降采样体素，单位 mm
DEFAULT_VIS = True  # 是否显示实时 3D 预览窗口（仅显示观测分割点云 + 先验点云）
DEFAULT_REALTIME_PREVIEW = True  # 是否显示实时 2D 预览窗口
DEFAULT_REGISTRATION_INTERVAL_SEC = 2.0  # 后台配准周期，单位 秒
DEFAULT_SAVE_RESULT = True  # 是否保存最优配准结果
DEFAULT_MIN_ACCEPT_FITNESS = 0.35  # 直接配准可接受 fitness 下限
DEFAULT_MAX_ACCEPT_RMSE_MM = 8.0  # 直接配准可接受 RMSE 上限，单位 mm
DEFAULT_REMOVE_OUTLIER = True  # 是否对观测点云做离群点剔除
DEFAULT_OUTLIER_NB_NEIGHBORS = 20  # 统计离群点邻居数量
DEFAULT_OUTLIER_STD_RATIO = 2.0  # 统计离群点标准差倍率
DEFAULT_KEEP_LARGEST_CLUSTER = True  # 是否仅保留最大连通簇
DEFAULT_CLUSTER_EPS_MM = 8.0  # DBSCAN 邻域半径，单位 mm
DEFAULT_CLUSTER_MIN_POINTS = 40  # DBSCAN 最小簇点数
DEFAULT_ENABLE_LONG_AXIS_CROP = True  # 是否启用沿长边方向的中心截取
DEFAULT_LONG_AXIS_KEEP_RATIO_SRC = 0.65  # 先验点云长边保留比例（中心段）
DEFAULT_LONG_AXIS_KEEP_RATIO_TGT = 0.65  # 观测点云长边保留比例（中心段）
DEFAULT_AUTO_SCALE_MIN = 0.2  # 自动缩放最小值
DEFAULT_AUTO_SCALE_MAX = 5.0  # 自动缩放最大值
DEFAULT_FINAL_EVAL_CORR_MM = 6.0  # 最终评估对应距离，单位 mm
DEFAULT_OBB_CORNER_WEIGHT = 0.25  # OBB 角点一致性权重（越大越强调长宽厚几何一致性）
# endregion


# region 数据结构
@dataclass
class RegistrationMetrics:
    frame_idx: int
    points: int
    fitness: float
    rmse_mm: float
    prior_scale_factor: float
    transformation: np.ndarray
    observed_cloud: o3d.geometry.PointCloud


@dataclass
class AsyncRegistrationResult:
    job_id: int
    frame_idx: int
    det_confidence: float
    points: int
    transformation: np.ndarray
    fitness: float
    rmse_mm: float
    prior_scale_factor: float
    observed_cloud: o3d.geometry.PointCloud


@dataclass
class RegistrationJob:
    job_id: int
    frame_idx: int
    det_confidence: float
    source_prior: o3d.geometry.PointCloud
    target_observed: o3d.geometry.PointCloud
    final_eval_corr_mm: float
    auto_scale_min: float
    auto_scale_max: float
    obb_corner_weight: float
    enable_long_axis_crop: bool
    long_axis_keep_ratio_src: float
    long_axis_keep_ratio_tgt: float


# endregion


# region 点云与配准
def _load_prior_pointcloud(path: Path, voxel_mm: float) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"先验点云不存在：{path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"先验点云为空：{path}")
    if voxel_mm > 0:
        down = pcd.voxel_down_sample(float(voxel_mm))
        if len(down.points) > 0:
            pcd = down
    _estimate_normals(pcd, radius_mm=max(6.0, float(voxel_mm) * 2.0))
    return pcd


def _estimate_normals(pcd: o3d.geometry.PointCloud, radius_mm: float) -> None:
    if len(pcd.points) == 0:
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(1.0, float(radius_mm)), max_nn=50),
    )


def _build_cloud(xyz: np.ndarray, rgb: np.ndarray | None = None, voxel_mm: float = 0.0) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    if rgb is not None and rgb.shape[0] == xyz.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0))
    if voxel_mm > 0:
        down = pcd.voxel_down_sample(float(voxel_mm))
        if len(down.points) > 0:
            pcd = down
    return pcd


def _refine_observed_cloud(
    pcd: o3d.geometry.PointCloud,
    remove_outlier: bool,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
    keep_largest_cluster: bool,
    cluster_eps_mm: float,
    cluster_min_points: int,
) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)
    if len(out.points) == 0:
        return out

    if remove_outlier and len(out.points) >= max(50, int(outlier_nb_neighbors) * 2):
        out, _ = out.remove_statistical_outlier(
            nb_neighbors=max(5, int(outlier_nb_neighbors)),
            std_ratio=max(0.5, float(outlier_std_ratio)),
        )
        if len(out.points) == 0:
            return out

    if keep_largest_cluster and len(out.points) >= max(80, int(cluster_min_points) * 2):
        labels = np.asarray(
            out.cluster_dbscan(
                eps=max(1.0, float(cluster_eps_mm)),
                min_points=max(3, int(cluster_min_points)),
                print_progress=False,
            ),
            dtype=np.int32,
        )
        valid = labels >= 0
        if np.any(valid):
            uniq, counts = np.unique(labels[valid], return_counts=True)
            largest = int(uniq[int(np.argmax(counts))])
            keep_idx = np.where(labels == largest)[0].astype(np.int32)
            out = out.select_by_index(keep_idx.tolist())
    return out


def _generate_proper_signed_permutations() -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    perm_indices = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    signs = [-1.0, 1.0]
    for p in perm_indices:
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    m = np.zeros((3, 3), dtype=np.float64)
                    m[0, p[0]] = sx
                    m[1, p[1]] = sy
                    m[2, p[2]] = sz
                    if float(np.linalg.det(m)) > 0.5:
                        mats.append(m)
    return mats


def _generate_planar_candidates() -> list[np.ndarray]:
    # 仅保留与平面场景相关的候选：不翻转 z 轴，且只考虑 0/180 度平面翻转。
    out: list[np.ndarray] = []
    for s in (1.0, -1.0):
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = s
        m[1, 1] = s
        m[2, 2] = 1.0
        if float(np.linalg.det(m)) > 0.5:
            out.append(m)
    return out


def _build_planar_obb_z_up(pcd: o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] < 10:
        return pcd.get_axis_aligned_bounding_box().get_oriented_bounding_box()

    xy = pts[:, :2]
    mean_xy = np.mean(xy, axis=0)
    xy0 = xy - mean_xy
    cov = (xy0.T @ xy0) / max(1, xy0.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    v0 = vecs[:, order[0]]
    yaw = float(np.arctan2(v0[1], v0[0]))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    r2 = np.array([[c, -s], [s, c]], dtype=np.float64)
    proj = xy0 @ r2
    min_xy = np.min(proj, axis=0)
    max_xy = np.max(proj, axis=0)
    extent_xy = np.maximum(max_xy - min_xy, 1e-6)
    center_xy_local = (min_xy + max_xy) * 0.5
    center_xy_world = mean_xy + (center_xy_local @ r2.T)

    z_min = float(np.min(pts[:, 2]))
    z_max = float(np.max(pts[:, 2]))
    extent_z = max(z_max - z_min, 1e-6)
    center_z = 0.5 * (z_min + z_max)

    center = np.array([center_xy_world[0], center_xy_world[1], center_z], dtype=np.float64)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    extent = np.array([extent_xy[0], extent_xy[1], extent_z], dtype=np.float64)
    return o3d.geometry.OrientedBoundingBox(center=center, R=rot, extent=extent)


def _crop_cloud_by_obb_long_axis_center(
    pcd: o3d.geometry.PointCloud,
    obb: o3d.geometry.OrientedBoundingBox,
    keep_ratio: float,
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] < 30:
        return o3d.geometry.PointCloud(pcd)
    ratio = float(np.clip(keep_ratio, 0.2, 1.0))
    ex = np.asarray(obb.extent, dtype=np.float64)
    long_axis = int(np.argmax(ex))
    center = np.asarray(obb.center, dtype=np.float64)
    rot = np.asarray(obb.R, dtype=np.float64)
    local = (pts - center) @ rot
    half = 0.5 * float(ex[long_axis]) * ratio
    mask = np.abs(local[:, long_axis]) <= max(half, 1e-6)
    idx = np.where(mask)[0]
    if idx.size < 30:
        return o3d.geometry.PointCloud(pcd)
    return pcd.select_by_index(idx.tolist())


def _estimate_scale_from_planar_obb(
    source_prior: o3d.geometry.PointCloud,
    target_observed: o3d.geometry.PointCloud,
    auto_scale_min: float,
    auto_scale_max: float,
) -> float:
    src_obb = _build_planar_obb_z_up(source_prior)
    tgt_obb = _build_planar_obb_z_up(target_observed)
    # 显式区分长/宽/厚三条边（降序），但缩放只使用宽和厚（忽略长边）。
    src_ex = np.sort(np.asarray(src_obb.extent, dtype=np.float64))[::-1]
    tgt_ex = np.sort(np.asarray(tgt_obb.extent, dtype=np.float64))[::-1]
    src_wt = src_ex[1:]
    tgt_wt = tgt_ex[1:]
    valid = (src_wt > 1e-6) & (tgt_wt > 1e-6)
    if not np.any(valid):
        return 1.0
    ratios = tgt_wt[valid] / src_wt[valid]
    scale = float(np.median(ratios))
    return float(np.clip(scale, float(auto_scale_min), float(auto_scale_max)))


def _run_obb_coarse_registration(
    source_prior: o3d.geometry.PointCloud,
    target_observed: o3d.geometry.PointCloud,
    final_eval_corr_mm: float,
    auto_scale_min: float,
    auto_scale_max: float,
    obb_corner_weight: float,
    enable_long_axis_crop: bool,
    long_axis_keep_ratio_src: float,
    long_axis_keep_ratio_tgt: float,
) -> tuple[np.ndarray, float, float, float]:
    if len(source_prior.points) < 30 or len(target_observed.points) < 30:
        return np.eye(4, dtype=np.float64), 0.0, 0.0, 1.0

    source_scaled = o3d.geometry.PointCloud(source_prior)
    target_used = o3d.geometry.PointCloud(target_observed)
    src_obb_raw = _build_planar_obb_z_up(source_scaled)
    tgt_obb_raw = _build_planar_obb_z_up(target_used)
    if enable_long_axis_crop:
        source_scaled = _crop_cloud_by_obb_long_axis_center(
            pcd=source_scaled,
            obb=src_obb_raw,
            keep_ratio=float(long_axis_keep_ratio_src),
        )
        target_used = _crop_cloud_by_obb_long_axis_center(
            pcd=target_used,
            obb=tgt_obb_raw,
            keep_ratio=float(long_axis_keep_ratio_tgt),
        )
        src_obb_raw = _build_planar_obb_z_up(source_scaled)
        tgt_obb_raw = _build_planar_obb_z_up(target_used)
        logger.info(
            f"长边中心截取：src_points {len(source_scaled.points)} 点, tgt_points {len(target_used.points)} 点, "
            f"src_keep_ratio {float(long_axis_keep_ratio_src):.2f}, tgt_keep_ratio {float(long_axis_keep_ratio_tgt):.2f}"
        )
    src_ex_raw = np.asarray(src_obb_raw.extent, dtype=np.float64)
    tgt_ex_raw = np.asarray(tgt_obb_raw.extent, dtype=np.float64)
    src_vol_raw = float(np.prod(np.maximum(src_ex_raw, 1e-6)))
    tgt_vol_raw = float(np.prod(np.maximum(tgt_ex_raw, 1e-6)))
    scale_factor = _estimate_scale_from_planar_obb(
        source_prior=source_scaled,
        target_observed=target_used,
        auto_scale_min=float(auto_scale_min),
        auto_scale_max=float(auto_scale_max),
    )
    src_ex_sorted = np.sort(src_ex_raw)[::-1]
    tgt_ex_sorted = np.sort(tgt_ex_raw)[::-1]
    edge_ratios = tgt_ex_sorted / np.maximum(src_ex_sorted, 1e-8)
    logger.info(
        f"OBB 自动缩放：src_extent(mm)={src_ex_raw.round(3).tolist()}, tgt_extent(mm)={tgt_ex_raw.round(3).tolist()}, "
        f"src_lwt={src_ex_sorted.round(3).tolist()}, tgt_lwt={tgt_ex_sorted.round(3).tolist()}, "
        f"ratio_lwt={edge_ratios.round(4).tolist()}, src_vol={src_vol_raw:.3f}, tgt_vol={tgt_vol_raw:.3f}, "
        f"scale={scale_factor:.6f}"
    )
    best_transform = np.eye(4, dtype=np.float64)
    best_fitness = -1.0
    best_rmse = 1e9
    best_corner_rmse = 1e9
    candidates = _generate_planar_candidates()
    logger.info(f"OBB 候选数量（已简化）：{len(candidates)}")
    src_scaled_once = o3d.geometry.PointCloud(source_prior)
    if abs(scale_factor - 1.0) > 1e-8:
        src_center = np.mean(np.asarray(src_scaled_once.points, dtype=np.float64), axis=0)
        src_scaled_once.scale(scale_factor, center=src_center.tolist())

    src_obb = _build_planar_obb_z_up(src_scaled_once)
    tgt_obb = _build_planar_obb_z_up(target_used)
    src_r = np.asarray(src_obb.R, dtype=np.float64)
    tgt_r = np.asarray(tgt_obb.R, dtype=np.float64)
    src_c = np.asarray(src_obb.center, dtype=np.float64)
    tgt_c = np.asarray(tgt_obb.center, dtype=np.float64)
    src_corners = np.asarray(src_obb.get_box_points(), dtype=np.float64)
    tgt_corners = np.asarray(tgt_obb.get_box_points(), dtype=np.float64)
    tgt_ex_raw = np.asarray(tgt_obb.extent, dtype=np.float64)
    long_axis_idx = int(np.argmax(tgt_ex_raw))
    keep_axes = [i for i in (0, 1, 2) if i != long_axis_idx]

    def _corner_bidirectional_rmse(tfm: np.ndarray) -> float:
        # 仅在“宽 + 厚”两个轴上比较角点一致性，忽略长边方向误差。
        src_tf = (tfm[:3, :3] @ src_corners.T).T + tfm[:3, 3]
        src_red = src_tf[:, keep_axes]
        tgt_red = tgt_corners[:, keep_axes]
        dmat = np.linalg.norm(src_red[:, None, :] - tgt_red[None, :, :], axis=2)
        src_to_tgt = np.min(dmat, axis=1)
        tgt_to_src = np.min(dmat, axis=0)
        both = np.concatenate([src_to_tgt, tgt_to_src], axis=0)
        return float(np.sqrt(np.mean(both * both)))

    for idx, p in enumerate(candidates, start=1):
        rot = tgt_r @ p @ src_r.T
        trans = tgt_c - rot @ src_c
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = trans
        final_eval = o3d.pipelines.registration.evaluate_registration(
            src_scaled_once,
            target_used,
            max_correspondence_distance=float(final_eval_corr_mm),
            transformation=transform,
        )
        fitness = float(final_eval.fitness)
        rmse = float(final_eval.inlier_rmse)
        corner_rmse = _corner_bidirectional_rmse(transform)
        combined_score = float(fitness) - float(obb_corner_weight) * float(corner_rmse) / max(
            float(final_eval_corr_mm), 1e-6
        )
        logger.info(
            f"OBB scale={scale_factor:.6f} candidate={idx} fitness {fitness:.4f}, rmse {rmse:.4f} mm, "
            f"corner_rmse {corner_rmse:.4f} mm, score {combined_score:.4f}"
        )
        if (
            fitness > best_fitness + 1e-8
            or (abs(fitness - best_fitness) <= 1e-8 and corner_rmse < best_corner_rmse - 1e-8)
            or (
                abs(fitness - best_fitness) <= 1e-8 and abs(corner_rmse - best_corner_rmse) <= 1e-8 and rmse < best_rmse
            )
        ):
            best_fitness = fitness
            best_rmse = rmse
            best_corner_rmse = corner_rmse
            best_transform = transform
    return best_transform, best_fitness, best_rmse, scale_factor


def _run_registration_job(job: RegistrationJob) -> AsyncRegistrationResult:
    transform, fitness, rmse, scale_factor = _run_obb_coarse_registration(
        source_prior=job.source_prior,
        target_observed=job.target_observed,
        final_eval_corr_mm=job.final_eval_corr_mm,
        auto_scale_min=job.auto_scale_min,
        auto_scale_max=job.auto_scale_max,
        obb_corner_weight=job.obb_corner_weight,
        enable_long_axis_crop=job.enable_long_axis_crop,
        long_axis_keep_ratio_src=job.long_axis_keep_ratio_src,
        long_axis_keep_ratio_tgt=job.long_axis_keep_ratio_tgt,
    )
    return AsyncRegistrationResult(
        job_id=int(job.job_id),
        frame_idx=int(job.frame_idx),
        det_confidence=float(job.det_confidence),
        points=int(len(job.target_observed.points)),
        transformation=np.asarray(transform, dtype=np.float64),
        fitness=float(fitness),
        rmse_mm=float(rmse),
        prior_scale_factor=float(scale_factor),
        observed_cloud=o3d.geometry.PointCloud(job.target_observed),
    )


def _draw_preview_overlay(
    base_2d: np.ndarray,
    dets: list[zs.Detection2D],
    metrics_text: str,
) -> np.ndarray:
    overlay = zs._draw_2d_overlay(rgb_img=base_2d, dets=dets, alpha=0.40)
    cv2.putText(
        overlay,
        metrics_text,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _make_obb_lineset(obb: o3d.geometry.OrientedBoundingBox, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    c = np.asarray(color, dtype=np.float64).reshape(1, 3)
    ls.colors = o3d.utility.Vector3dVector(np.repeat(c, repeats=len(ls.lines), axis=0))
    return ls


def _init_realtime_3d_viewer() -> tuple[
    o3d.visualization.VisualizerWithKeyCallback,
    dict[str, bool],
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.geometry.LineSet,
    o3d.geometry.LineSet,
]:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Stage2 Realtime 3D (observed + prior)", 1440, 900)
    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = 3.0
        render_opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)

    observed_vis = o3d.geometry.PointCloud()
    prior_vis = o3d.geometry.PointCloud()
    observed_obb_vis = o3d.geometry.LineSet()
    prior_obb_vis = o3d.geometry.LineSet()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axis)
    vis.add_geometry(observed_vis)
    vis.add_geometry(prior_vis)
    vis.add_geometry(observed_obb_vis)
    vis.add_geometry(prior_obb_vis)
    view = vis.get_view_control()
    if view is not None:
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
    return vis, stop, observed_vis, prior_vis, observed_obb_vis, prior_obb_vis


def _fit_camera_to_clouds(
    vis: o3d.visualization.VisualizerWithKeyCallback,
    observed: o3d.geometry.PointCloud,
    prior: o3d.geometry.PointCloud,
) -> None:
    pts_obs = np.asarray(observed.points, dtype=np.float64)
    pts_pri = np.asarray(prior.points, dtype=np.float64)
    if pts_obs.shape[0] == 0 and pts_pri.shape[0] == 0:
        return
    if pts_obs.shape[0] == 0:
        pts = pts_pri
    elif pts_pri.shape[0] == 0:
        pts = pts_obs
    else:
        pts = np.vstack([pts_obs, pts_pri])
    center = np.mean(pts, axis=0)
    vc = vis.get_view_control()
    if vc is None:
        return
    vc.set_lookat(center.tolist())
    vc.set_front([0.0, 0.0, -1.0])
    vc.set_up([0.0, -1.0, 0.0])
    vc.set_zoom(0.35)


def _save_result(
    output_dir: Path,
    metrics: RegistrationMetrics,
    prior_cloud: o3d.geometry.PointCloud,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tf_path = output_dir / "best_transform_material_plate_to_observed.txt"
    np.savetxt(str(tf_path), metrics.transformation, fmt="%.8f")

    observed_path = output_dir / "best_observed_segment.ply"
    o3d.io.write_point_cloud(str(observed_path), metrics.observed_cloud)

    aligned_prior = o3d.geometry.PointCloud(prior_cloud)
    if abs(float(metrics.prior_scale_factor) - 1.0) > 1e-8:
        c = np.mean(np.asarray(aligned_prior.points, dtype=np.float64), axis=0)
        aligned_prior.scale(float(metrics.prior_scale_factor), center=c.tolist())
    aligned_prior.transform(metrics.transformation)
    aligned_path = output_dir / "best_aligned_prior.ply"
    o3d.io.write_point_cloud(str(aligned_path), aligned_prior)
    logger.success(f"已保存结果：transform={tf_path}, observed={observed_path}, aligned_prior={aligned_path}")


def _visualize_registration(
    prior_cloud: o3d.geometry.PointCloud,
    observed_cloud: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> None:
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Stage2 直接配准（MaterialPlate -> 实时分割）", 1440, 900)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.set_background(np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32), None)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 1.5

    obs = o3d.geometry.PointCloud(observed_cloud)
    obs.paint_uniform_color([0.75, 0.75, 0.75])
    prior_raw = o3d.geometry.PointCloud(prior_cloud)
    prior_raw.paint_uniform_color([0.9, 0.25, 0.25])
    prior_aligned = o3d.geometry.PointCloud(prior_cloud)
    prior_aligned.transform(np.asarray(transformation, dtype=np.float64))
    prior_aligned.paint_uniform_color([0.20, 0.70, 1.00])

    vis.add_geometry("observed_target", obs, mat)
    vis.add_geometry("prior_raw", prior_raw, mat)
    vis.add_geometry("prior_aligned", prior_aligned, mat)
    vis.add_geometry("axis", o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0]))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


# endregion


# region 主流程
def main(
    prior_ply: Path = DEFAULT_PRIOR_PLY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    vis: bool = DEFAULT_VIS,
    realtime_preview: bool = DEFAULT_REALTIME_PREVIEW,
    registration_interval_sec: float = DEFAULT_REGISTRATION_INTERVAL_SEC,
    save_result: bool = DEFAULT_SAVE_RESULT,
    max_frames: int = DEFAULT_MAX_FRAMES,
    detect_interval: int = DEFAULT_DETECT_INTERVAL,
    min_object_points: int = DEFAULT_MIN_OBJECT_POINTS,
    segment_voxel_mm: float = DEFAULT_SEGMENT_VOXEL_MM,
    prior_voxel_mm: float = DEFAULT_PRIOR_VOXEL_MM,
    min_accept_fitness: float = DEFAULT_MIN_ACCEPT_FITNESS,
    max_accept_rmse_mm: float = DEFAULT_MAX_ACCEPT_RMSE_MM,
    remove_outlier: bool = DEFAULT_REMOVE_OUTLIER,
    outlier_nb_neighbors: int = DEFAULT_OUTLIER_NB_NEIGHBORS,
    outlier_std_ratio: float = DEFAULT_OUTLIER_STD_RATIO,
    keep_largest_cluster: bool = DEFAULT_KEEP_LARGEST_CLUSTER,
    cluster_eps_mm: float = DEFAULT_CLUSTER_EPS_MM,
    cluster_min_points: int = DEFAULT_CLUSTER_MIN_POINTS,
    enable_long_axis_crop: bool = DEFAULT_ENABLE_LONG_AXIS_CROP,
    long_axis_keep_ratio_src: float = DEFAULT_LONG_AXIS_KEEP_RATIO_SRC,
    long_axis_keep_ratio_tgt: float = DEFAULT_LONG_AXIS_KEEP_RATIO_TGT,
    auto_scale_min: float = DEFAULT_AUTO_SCALE_MIN,
    auto_scale_max: float = DEFAULT_AUTO_SCALE_MAX,
    obb_corner_weight: float = DEFAULT_OBB_CORNER_WEIGHT,
    final_eval_corr_mm: float = DEFAULT_FINAL_EVAL_CORR_MM,
    gd_model_id: str = zs.DEFAULT_GD_MODEL_ID,
    sam_model_id: str = zs.DEFAULT_SAM_MODEL_ID,
    hf_cache_dir: str = DEFAULT_HF_CACHE_DIR,
    hf_local_files_only: bool = DEFAULT_HF_LOCAL_FILES_ONLY,
    device: str = DEFAULT_DEVICE,
    proxy_url: str = DEFAULT_PROXY_URL,
    prompt: str = DEFAULT_PROMPT,
    target_keywords: str = DEFAULT_TARGET_KEYWORDS,
    strict_target_filter: bool = DEFAULT_STRICT_TARGET_FILTER,
    max_targets: int = DEFAULT_MAX_TARGETS,
    use_sam: bool = DEFAULT_USE_SAM,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    min_target_conf: float = DEFAULT_MIN_TARGET_CONF,
    topk_objects: int = DEFAULT_TOPK_OBJECTS,
    sam_max_boxes: int = DEFAULT_SAM_MAX_BOXES,
    sam_primary_only: bool = DEFAULT_SAM_PRIMARY_ONLY,
    sam_secondary_conf_threshold: float = DEFAULT_SAM_SECONDARY_CONF_THRESHOLD,
    min_mask_pixels: int = DEFAULT_MIN_MASK_PIXELS,
    mask_iou_suppress: float = DEFAULT_MASK_IOU_SUPPRESS,
    detect_max_side: int = DEFAULT_DETECT_MAX_SIDE,
    combine_prompts_forward: bool = DEFAULT_COMBINE_PROMPTS_FORWARD,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
) -> None:
    prior_cloud = _load_prior_pointcloud(Path(prior_ply), voxel_mm=float(prior_voxel_mm))
    detector = zs.ZeroShotObjectPartitionDetector(
        gd_model_id=gd_model_id,
        sam_model_id=sam_model_id,
        hf_cache_dir=hf_cache_dir,
        hf_local_files_only=hf_local_files_only,
        device=device,
        proxy_url=proxy_url,
        prompt=prompt,
        target_keywords=target_keywords,
        strict_target_filter=strict_target_filter,
        max_targets=max(1, int(max_targets)),
        use_sam=use_sam,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        min_target_conf=min_target_conf,
        topk_objects=max(1, int(topk_objects)),
        sam_max_boxes=max(1, int(sam_max_boxes)),
        sam_primary_only=sam_primary_only,
        sam_secondary_conf_threshold=sam_secondary_conf_threshold,
        combine_prompts_forward=combine_prompts_forward,
        min_mask_pixels=max(20, int(min_mask_pixels)),
        mask_iou_suppress=mask_iou_suppress,
        detect_max_side=max(128, int(detect_max_side)),
    )

    logger.info(
        f"启动 Stage2 直接配准：prior {Path(prior_ply)}, max_frames {int(max_frames)}, "
        f"detect_interval {int(detect_interval)}, min_object_points {int(min_object_points)} 点，"
        f"realtime_preview {bool(realtime_preview)}, registration_interval {float(registration_interval_sec):.2f} 秒，"
        f"auto_scale_by_obb True, auto_scale_min {float(auto_scale_min):.3f}, auto_scale_max {float(auto_scale_max):.3f}, "
        f"obb_corner_weight {float(obb_corner_weight):.3f}, long_axis_crop {bool(enable_long_axis_crop)}, "
        f"src_keep {float(long_axis_keep_ratio_src):.2f}, tgt_keep {float(long_axis_keep_ratio_tgt):.2f}"
    )

    best: RegistrationMetrics | None = None
    preview_win = "Stage2 MaterialPlate Direct Registration"
    if bool(realtime_preview):
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_win, 1280, 720)
    vis3d = None
    vis3d_stop = None
    observed_vis_pcd = None
    prior_vis_pcd = None
    observed_obb_ls = None
    prior_obb_ls = None
    camera_initialized = False
    if bool(vis):
        vis3d, vis3d_stop, observed_vis_pcd, prior_vis_pcd, observed_obb_ls, prior_obb_ls = _init_realtime_3d_viewer()

    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
    prior_for_view = o3d.geometry.PointCloud(prior_cloud)
    last_transform = np.eye(4, dtype=np.float64)
    last_async_result: AsyncRegistrationResult | None = None
    last_submit_ts = 0.0
    registration_running = {"flag": False}
    latest_submitted_job_id = 0
    next_job_id = 0
    job_queue: queue.Queue[RegistrationJob] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[AsyncRegistrationResult] = queue.Queue(maxsize=1)
    worker_stop = threading.Event()

    def _worker_loop() -> None:
        while not worker_stop.is_set():
            try:
                job = job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if worker_stop.is_set():
                break
            registration_running["flag"] = True
            try:
                res = _run_registration_job(job)
                while True:
                    try:
                        result_queue.put_nowait(res)
                        break
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
            finally:
                registration_running["flag"] = False
                job_queue.task_done()

    worker = threading.Thread(target=_worker_loop, name="stage2_obb_worker", daemon=True)
    worker.start()
    try:
        with Gemini305(options=options) as session:
            cam = session.get_camera_param()
            ci = cam.rgb_intrinsic if session.has_color_sensor else cam.depth_intrinsic
            img_w = int(max(32, ci.width))
            img_h = int(max(32, ci.height))
            fx = float(ci.fx)
            fy = float(ci.fy)
            cx = float(ci.cx)
            cy = float(ci.cy)
            point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())

            cached_dets: list[zs.Detection2D] = []
            detect_countdown = 0
            for frame_idx in range(1, int(max_frames) + 1):
                while True:
                    try:
                        async_res = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    if int(async_res.job_id) != int(latest_submitted_job_id):
                        continue
                    last_async_result = async_res
                    last_transform = np.asarray(async_res.transformation, dtype=np.float64)
                    logger.info(
                        f"帧 {async_res.frame_idx} 配准完成：det_conf {async_res.det_confidence:.3f}, "
                        f"segment_points {async_res.points} 点，fitness {async_res.fitness:.4f}, rmse {async_res.rmse_mm:.4f} mm, "
                        f"prior_scale {async_res.prior_scale_factor:.4f}"
                    )
                    if best is None or async_res.fitness > best.fitness:
                        best = RegistrationMetrics(
                            frame_idx=int(async_res.frame_idx),
                            points=int(async_res.points),
                            fitness=float(async_res.fitness),
                            rmse_mm=float(async_res.rmse_mm),
                            prior_scale_factor=float(async_res.prior_scale_factor),
                            transformation=np.asarray(async_res.transformation, dtype=np.float64),
                            observed_cloud=o3d.geometry.PointCloud(async_res.observed_cloud),
                        )

                points, color_bgr = zs._capture_preview_with_color_once(session=session, point_filter=point_filter)
                if points is None or len(points) == 0:
                    continue

                xyz = np.asarray(points[:, :3], dtype=np.float64)
                rgb = zs._extract_rgb(points)
                uv, valid_proj = zs._project_points_to_image(xyz=xyz, fx=fx, fy=fy, cx=cx, cy=cy, w=img_w, h=img_h)
                rgb_img = zs._rasterize_rgb(xyz=xyz, rgb=rgb, uv=uv, valid_proj=valid_proj, w=img_w, h=img_h)
                base_2d = (
                    cv2.resize(color_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    if color_bgr is not None
                    else rgb_img
                )

                if detect_countdown <= 0:
                    dets_new = detector.detect(base_2d)
                    if len(dets_new) > 0:
                        cached_dets = dets_new
                    detect_countdown = max(0, int(detect_interval) - 1)
                else:
                    detect_countdown -= 1

                det_for_overlay: list[zs.Detection2D] = cached_dets[:1] if len(cached_dets) > 0 else []
                metrics_text = f"frame {frame_idx} | waiting detection"

                if len(cached_dets) > 0:
                    det = cached_dets[0]
                    ids = zs._collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=det.mask)
                    if int(ids.size) >= int(min_object_points):
                        obs_xyz = xyz[ids]
                        obs_rgb = rgb[ids] if rgb.shape[0] == xyz.shape[0] else None
                        observed_cloud = _build_cloud(obs_xyz, obs_rgb, voxel_mm=float(segment_voxel_mm))
                        observed_cloud = _refine_observed_cloud(
                            pcd=observed_cloud,
                            remove_outlier=bool(remove_outlier),
                            outlier_nb_neighbors=int(outlier_nb_neighbors),
                            outlier_std_ratio=float(outlier_std_ratio),
                            keep_largest_cluster=bool(keep_largest_cluster),
                            cluster_eps_mm=float(cluster_eps_mm),
                            cluster_min_points=int(cluster_min_points),
                        )
                        if len(observed_cloud.points) >= int(min_object_points):
                            if (
                                bool(vis)
                                and observed_vis_pcd is not None
                                and prior_vis_pcd is not None
                                and vis3d is not None
                                and vis3d_stop is not None
                            ):
                                observed_show = o3d.geometry.PointCloud(observed_cloud)
                                if last_async_result is not None:
                                    observed_show = o3d.geometry.PointCloud(last_async_result.observed_cloud)
                                prior_show = o3d.geometry.PointCloud(prior_for_view)
                                if (
                                    last_async_result is not None
                                    and abs(float(last_async_result.prior_scale_factor) - 1.0) > 1e-8
                                ):
                                    c = np.mean(np.asarray(prior_show.points, dtype=np.float64), axis=0)
                                    prior_show.scale(float(last_async_result.prior_scale_factor), center=c.tolist())
                                prior_show.transform(last_transform)
                                preview_obs = o3d.geometry.PointCloud(observed_show)
                                preview_pri = o3d.geometry.PointCloud(prior_show)
                                obs_pts = np.asarray(preview_obs.points, dtype=np.float64)
                                if obs_pts.shape[0] > 0:
                                    anchor = np.mean(obs_pts, axis=0)
                                    preview_obs.translate(-anchor)
                                    preview_pri.translate(-anchor)
                                preview_obs.paint_uniform_color([0.10, 0.85, 0.10])  # 观测点云：绿
                                preview_pri.paint_uniform_color([0.95, 0.10, 0.10])  # 先验点云：红
                                observed_vis_pcd.points = preview_obs.points
                                observed_vis_pcd.colors = preview_obs.colors
                                prior_vis_pcd.points = preview_pri.points
                                prior_vis_pcd.colors = preview_pri.colors
                                vis3d.update_geometry(observed_vis_pcd)
                                vis3d.update_geometry(prior_vis_pcd)
                                if observed_obb_ls is not None and prior_obb_ls is not None:
                                    if len(preview_obs.points) >= 10:
                                        obs_obb = preview_obs.get_oriented_bounding_box()
                                        obs_ls = _make_obb_lineset(obs_obb, color=(1.0, 0.8, 0.1))
                                        observed_obb_ls.points = obs_ls.points
                                        observed_obb_ls.lines = obs_ls.lines
                                        observed_obb_ls.colors = obs_ls.colors
                                        vis3d.update_geometry(observed_obb_ls)
                                    if len(preview_pri.points) >= 10:
                                        pri_obb = preview_pri.get_oriented_bounding_box()
                                        pri_ls = _make_obb_lineset(pri_obb, color=(0.1, 0.8, 1.0))
                                        prior_obb_ls.points = pri_ls.points
                                        prior_obb_ls.lines = pri_ls.lines
                                        prior_obb_ls.colors = pri_ls.colors
                                        vis3d.update_geometry(prior_obb_ls)
                                alive = vis3d.poll_events()
                                vis3d.update_renderer()
                                if vis3d_stop["flag"] or (not alive):
                                    logger.warning("3D 预览窗口关闭，提前结束。")
                                    break

                            now_ts = time.monotonic()
                            if now_ts - last_submit_ts >= max(0.2, float(registration_interval_sec)):
                                next_job_id += 1
                                latest_submitted_job_id = next_job_id
                                new_job = RegistrationJob(
                                    job_id=int(next_job_id),
                                    frame_idx=int(frame_idx),
                                    det_confidence=float(det.confidence_2d),
                                    source_prior=o3d.geometry.PointCloud(prior_cloud),
                                    target_observed=o3d.geometry.PointCloud(observed_cloud),
                                    final_eval_corr_mm=float(final_eval_corr_mm),
                                    auto_scale_min=float(auto_scale_min),
                                    auto_scale_max=float(auto_scale_max),
                                    obb_corner_weight=float(obb_corner_weight),
                                    enable_long_axis_crop=bool(enable_long_axis_crop),
                                    long_axis_keep_ratio_src=float(long_axis_keep_ratio_src),
                                    long_axis_keep_ratio_tgt=float(long_axis_keep_ratio_tgt),
                                )
                                while True:
                                    try:
                                        job_queue.put_nowait(new_job)
                                        break
                                    except queue.Full:
                                        try:
                                            _ = job_queue.get_nowait()
                                            job_queue.task_done()
                                        except queue.Empty:
                                            break
                                last_submit_ts = now_ts
                            if registration_running["flag"]:
                                metrics_text = f"frame {frame_idx} | det {det.confidence_2d:.3f} | pts {len(observed_cloud.points)} | obb running"
                            elif job_queue.qsize() > 0:
                                metrics_text = f"frame {frame_idx} | det {det.confidence_2d:.3f} | pts {len(observed_cloud.points)} | obb queued"
                            else:
                                metrics_text = f"frame {frame_idx} | det {det.confidence_2d:.3f} | pts {len(observed_cloud.points)} | obb idle"
                        else:
                            metrics_text = f"frame {frame_idx} | det {det.confidence_2d:.3f} | filtered points {len(observed_cloud.points)}"
                    else:
                        metrics_text = (
                            f"frame {frame_idx} | det {det.confidence_2d:.3f} | too few points {int(ids.size)}"
                        )

                if last_async_result is not None:
                    metrics_text += (
                        f" | last fit {last_async_result.fitness:.3f} rmse {last_async_result.rmse_mm:.3f}mm"
                    )
                if best is not None:
                    metrics_text += f" | best {best.fitness:.3f}/{best.rmse_mm:.3f}mm"

                if bool(realtime_preview):
                    frame_show = _draw_preview_overlay(base_2d=base_2d, dets=det_for_overlay, metrics_text=metrics_text)
                    cv2.imshow(preview_win, frame_show)
                    if cv2.waitKey(1) == 27:
                        logger.warning("用户按下 ESC，提前结束。")
                        break
                if bool(vis) and vis3d is not None and vis3d_stop is not None:
                    alive = vis3d.poll_events()
                    vis3d.update_renderer()
                    if vis3d_stop["flag"] or (not alive):
                        logger.warning("3D 预览窗口关闭，提前结束。")
                        break
    finally:
        worker_stop.set()
        try:
            worker.join(timeout=1.5)
        except Exception:
            pass
        if bool(vis) and vis3d is not None:
            vis3d.destroy_window()
        if bool(realtime_preview):
            cv2.destroyWindow(preview_win)

    if best is None:
        logger.warning("未获得可用的目标分割点云，未执行到有效配准。")
        return

    logger.success(
        f"最佳配准：frame {best.frame_idx}, points {best.points} 点，"
        f"fitness {best.fitness:.4f}, rmse {best.rmse_mm:.4f} mm"
    )
    logger.info(f"最佳变换矩阵（MaterialPlate -> Observed）:\n{best.transformation}")

    direct_ok = best.fitness >= float(min_accept_fitness) and best.rmse_mm <= float(max_accept_rmse_mm)
    if direct_ok:
        logger.success(
            f"直接配准满足阈值：fitness >= {float(min_accept_fitness):.2f}, rmse <= {float(max_accept_rmse_mm):.2f} mm。"
        )
        logger.success("当前结果可作为“先不做特征识别”的依据。")
    else:
        logger.warning(
            f"直接配准未达到阈值：fitness {best.fitness:.4f}, rmse {best.rmse_mm:.4f} mm。"
            "建议下一步再引入特征粗配准。"
        )

    if bool(save_result):
        _save_result(
            output_dir=Path(output_dir),
            metrics=best,
            prior_cloud=prior_cloud,
        )


# endregion


# region CLI
def _parse_cli() -> tuple[
    Path,
    Path,
    bool,
    bool,
    float,
    bool,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    bool,
    int,
    float,
    bool,
    float,
    int,
    bool,
    float,
    float,
    float,
    float,
    float,
    str,
    str,
    str,
    bool,
    str,
]:
    parser = argparse.ArgumentParser(description="Stage2：MaterialPlate 先验点云与实时分割点云直接配准测试")
    parser.add_argument("--prior-ply", type=Path, default=DEFAULT_PRIOR_PLY, help="先验点云路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=DEFAULT_VIS, help="是否显示可视化窗口")
    parser.add_argument(
        "--realtime-preview",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REALTIME_PREVIEW,
        help="是否显示实时 2D 分割与配准指标预览",
    )
    parser.add_argument(
        "--registration-interval-sec",
        type=float,
        default=DEFAULT_REGISTRATION_INTERVAL_SEC,
        help="后台配准周期，单位 秒",
    )
    parser.add_argument(
        "--save-result", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_RESULT, help="是否保存结果"
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最大采样帧数")
    parser.add_argument("--detect-interval", type=int, default=DEFAULT_DETECT_INTERVAL, help="每 N 帧检测一次")
    parser.add_argument("--min-object-points", type=int, default=DEFAULT_MIN_OBJECT_POINTS, help="最小目标点数")
    parser.add_argument(
        "--segment-voxel-mm", type=float, default=DEFAULT_SEGMENT_VOXEL_MM, help="观测体素尺寸，单位 mm"
    )
    parser.add_argument("--prior-voxel-mm", type=float, default=DEFAULT_PRIOR_VOXEL_MM, help="先验体素尺寸，单位 mm")
    parser.add_argument(
        "--min-accept-fitness", type=float, default=DEFAULT_MIN_ACCEPT_FITNESS, help="直接配准 fitness 阈值"
    )
    parser.add_argument(
        "--max-accept-rmse-mm", type=float, default=DEFAULT_MAX_ACCEPT_RMSE_MM, help="直接配准 RMSE 阈值，单位 mm"
    )
    parser.add_argument(
        "--remove-outlier",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REMOVE_OUTLIER,
        help="是否做离群点剔除",
    )
    parser.add_argument("--outlier-nb-neighbors", type=int, default=DEFAULT_OUTLIER_NB_NEIGHBORS, help="离群点邻居数量")
    parser.add_argument("--outlier-std-ratio", type=float, default=DEFAULT_OUTLIER_STD_RATIO, help="离群点标准差倍率")
    parser.add_argument(
        "--keep-largest-cluster",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_KEEP_LARGEST_CLUSTER,
        help="是否仅保留最大连通簇",
    )
    parser.add_argument("--cluster-eps-mm", type=float, default=DEFAULT_CLUSTER_EPS_MM, help="DBSCAN 邻域半径，单位 mm")
    parser.add_argument("--cluster-min-points", type=int, default=DEFAULT_CLUSTER_MIN_POINTS, help="DBSCAN 最小簇点数")
    parser.add_argument(
        "--enable-long-axis-crop",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_LONG_AXIS_CROP,
        help="是否启用沿长边方向中心截取",
    )
    parser.add_argument(
        "--long-axis-keep-ratio-src",
        type=float,
        default=DEFAULT_LONG_AXIS_KEEP_RATIO_SRC,
        help="先验点云长边中心保留比例",
    )
    parser.add_argument(
        "--long-axis-keep-ratio-tgt",
        type=float,
        default=DEFAULT_LONG_AXIS_KEEP_RATIO_TGT,
        help="观测点云长边中心保留比例",
    )
    parser.add_argument("--auto-scale-min", type=float, default=DEFAULT_AUTO_SCALE_MIN, help="自动缩放最小值")
    parser.add_argument("--auto-scale-max", type=float, default=DEFAULT_AUTO_SCALE_MAX, help="自动缩放最大值")
    parser.add_argument("--obb-corner-weight", type=float, default=DEFAULT_OBB_CORNER_WEIGHT, help="OBB 角点一致性权重")
    parser.add_argument(
        "--final-eval-corr-mm", type=float, default=DEFAULT_FINAL_EVAL_CORR_MM, help="最终评估对应距离，单位 mm"
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="推理设备，示例 cuda:0 / cpu")
    parser.add_argument("--proxy-url", type=str, default=DEFAULT_PROXY_URL, help="下载代理地址")
    parser.add_argument("--hf-cache-dir", type=str, default=DEFAULT_HF_CACHE_DIR, help="HuggingFace 本地缓存目录")
    parser.add_argument(
        "--hf-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HF_LOCAL_FILES_ONLY,
        help="是否仅使用本地缓存模型（首次下载请关闭）",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="检测提示词")
    args = parser.parse_args()
    return (
        Path(args.prior_ply),
        Path(args.output_dir),
        bool(args.vis),
        bool(args.realtime_preview),
        float(args.registration_interval_sec),
        bool(args.save_result),
        int(args.max_frames),
        int(args.detect_interval),
        int(args.min_object_points),
        float(args.segment_voxel_mm),
        float(args.prior_voxel_mm),
        float(args.min_accept_fitness),
        float(args.max_accept_rmse_mm),
        bool(args.remove_outlier),
        int(args.outlier_nb_neighbors),
        float(args.outlier_std_ratio),
        bool(args.keep_largest_cluster),
        float(args.cluster_eps_mm),
        int(args.cluster_min_points),
        bool(args.enable_long_axis_crop),
        float(args.long_axis_keep_ratio_src),
        float(args.long_axis_keep_ratio_tgt),
        float(args.auto_scale_min),
        float(args.auto_scale_max),
        float(args.obb_corner_weight),
        float(args.final_eval_corr_mm),
        str(args.device),
        str(args.proxy_url),
        str(args.hf_cache_dir),
        bool(args.hf_local_files_only),
        str(args.prompt),
    )


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            (
                prior_ply_arg,
                output_dir_arg,
                vis_arg,
                realtime_preview_arg,
                reg_interval_arg,
                save_arg,
                max_frames_arg,
                detect_interval_arg,
                min_obj_pts_arg,
                segment_voxel_arg,
                prior_voxel_arg,
                min_fit_arg,
                max_rmse_arg,
                remove_outlier_arg,
                outlier_nb_neighbors_arg,
                outlier_std_ratio_arg,
                keep_largest_cluster_arg,
                cluster_eps_mm_arg,
                cluster_min_points_arg,
                enable_long_axis_crop_arg,
                long_axis_keep_ratio_src_arg,
                long_axis_keep_ratio_tgt_arg,
                auto_scale_min_arg,
                auto_scale_max_arg,
                obb_corner_weight_arg,
                final_eval_corr_mm_arg,
                device_arg,
                proxy_arg,
                hf_cache_arg,
                hf_local_only_arg,
                prompt_arg,
            ) = _parse_cli()
            main(
                prior_ply=prior_ply_arg,
                output_dir=output_dir_arg,
                vis=vis_arg,
                realtime_preview=realtime_preview_arg,
                registration_interval_sec=reg_interval_arg,
                save_result=save_arg,
                max_frames=max_frames_arg,
                detect_interval=detect_interval_arg,
                min_object_points=min_obj_pts_arg,
                segment_voxel_mm=segment_voxel_arg,
                prior_voxel_mm=prior_voxel_arg,
                min_accept_fitness=min_fit_arg,
                max_accept_rmse_mm=max_rmse_arg,
                remove_outlier=remove_outlier_arg,
                outlier_nb_neighbors=outlier_nb_neighbors_arg,
                outlier_std_ratio=outlier_std_ratio_arg,
                keep_largest_cluster=keep_largest_cluster_arg,
                cluster_eps_mm=cluster_eps_mm_arg,
                cluster_min_points=cluster_min_points_arg,
                enable_long_axis_crop=enable_long_axis_crop_arg,
                long_axis_keep_ratio_src=long_axis_keep_ratio_src_arg,
                long_axis_keep_ratio_tgt=long_axis_keep_ratio_tgt_arg,
                auto_scale_min=auto_scale_min_arg,
                auto_scale_max=auto_scale_max_arg,
                obb_corner_weight=obb_corner_weight_arg,
                final_eval_corr_mm=final_eval_corr_mm_arg,
                device=device_arg,
                proxy_url=proxy_arg,
                hf_cache_dir=hf_cache_arg,
                hf_local_files_only=hf_local_only_arg,
                prompt=prompt_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
