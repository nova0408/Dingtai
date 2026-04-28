from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import sys
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

from src.rgbd_camera import Gemini305, SessionOptions
from test.pointcloud import test_orbbec_realtime_plane_segmentation_zero_shot as zs


def _load_colorize_by_cycle():
    module_path = PROJECT_ROOT / "src" / "pointcloud" / "pointcloud_visual.py"
    spec = importlib.util.spec_from_file_location("pointcloud_visual_runtime", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 pointcloud_visual.py：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "colorize_by_cycle", None)
    if fn is None:
        raise RuntimeError("pointcloud_visual.py 中不存在 colorize_by_cycle")
    return fn


COLORIZE_BY_CYCLE = _load_colorize_by_cycle()

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
DEFAULT_MAX_FRAMES = 240  # 最大采样帧数
DEFAULT_MIN_OBJECT_POINTS = 800  # 分割后最小点数，单位 点
DEFAULT_SEGMENT_VOXEL_MM = 2.0  # 观测点云降采样体素，单位 mm
DEFAULT_PRIOR_VOXEL_MM = 2.0  # 先验点云降采样体素，单位 mm
DEFAULT_VIS = True  # 是否显示实时 3D 预览窗口（仅显示观测分割点云+先验点云）
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
DEFAULT_MULTI_INIT_YAW_DEG = "0,90,180,270"  # 多初值 yaw 候选（度）
DEFAULT_FINAL_EVAL_CORR_MM = 6.0  # 最终评估对应距离，单位 mm
DEFAULT_ICP_LEVELS = "6.0:18.0:50,3.0:9.0:35"  # ICP 金字塔层级：voxel_mm:corr_mm:iters
# endregion


# region 数据结构
@dataclass
class RegistrationMetrics:
    frame_idx: int
    points: int
    fitness: float
    rmse_mm: float
    transformation: np.ndarray
    observed_cloud: o3d.geometry.PointCloud


@dataclass
class AsyncRegistrationResult:
    frame_idx: int
    det_confidence: float
    points: int
    transformation: np.ndarray
    fitness: float
    rmse_mm: float
    observed_cloud: o3d.geometry.PointCloud


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


def _parse_float_list_csv(text: str) -> list[float]:
    out: list[float] = []
    for s in str(text).replace("，", ",").split(","):
        v = s.strip()
        if len(v) == 0:
            continue
        out.append(float(v))
    return out


def _parse_icp_levels(text: str) -> list[tuple[float, float, int]]:
    levels: list[tuple[float, float, int]] = []
    for token in str(text).replace("，", ",").split(","):
        item = token.strip()
        if len(item) == 0:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"ICP 层级格式错误：{item}，应为 voxel:corr:iters")
        voxel = float(parts[0])
        corr = float(parts[1])
        iters = int(parts[2])
        levels.append((voxel, corr, iters))
    if len(levels) == 0:
        raise ValueError("ICP 层级为空，请至少提供一层。")
    return levels


def _rotation_z_deg(deg: float) -> np.ndarray:
    rad = float(np.deg2rad(float(deg)))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    r = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = r
    return t


def _centroid_init(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
    src_xyz = np.asarray(source.points, dtype=np.float64)
    tgt_xyz = np.asarray(target.points, dtype=np.float64)
    if src_xyz.shape[0] == 0 or tgt_xyz.shape[0] == 0:
        return np.eye(4, dtype=np.float64)
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = np.mean(tgt_xyz, axis=0) - np.mean(src_xyz, axis=0)
    return init


def _build_multi_yaw_inits(
    source_prior: o3d.geometry.PointCloud,
    target_observed: o3d.geometry.PointCloud,
    yaw_candidates_deg: list[float],
) -> list[np.ndarray]:
    src_xyz = np.asarray(source_prior.points, dtype=np.float64)
    tgt_xyz = np.asarray(target_observed.points, dtype=np.float64)
    if src_xyz.shape[0] == 0 or tgt_xyz.shape[0] == 0:
        return [np.eye(4, dtype=np.float64)]
    src_center = np.mean(src_xyz, axis=0)
    tgt_center = np.mean(tgt_xyz, axis=0)
    out: list[np.ndarray] = []
    for yaw in yaw_candidates_deg:
        rz = _rotation_z_deg(yaw)
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = rz[:3, :3]
        t[:3, 3] = tgt_center - (t[:3, :3] @ src_center)
        out.append(t)
    if len(out) == 0:
        out.append(_centroid_init(source_prior, target_observed))
    return out


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


def _run_direct_multiscale_icp(
    source_prior: o3d.geometry.PointCloud,
    target_observed: o3d.geometry.PointCloud,
    init_transforms: list[np.ndarray],
    final_eval_corr_mm: float,
    icp_levels: list[tuple[float, float, int]],
) -> tuple[np.ndarray, float, float]:
    best_transform = np.eye(4, dtype=np.float64)
    best_fitness = -1.0
    best_rmse = 1e9
    for init_idx, init_transform in enumerate(init_transforms, start=1):
        transform = np.asarray(init_transform, dtype=np.float64).copy()
        for level, (voxel_mm, corr_mm, iters) in enumerate(icp_levels, start=1):
            src = source_prior.voxel_down_sample(voxel_mm)
            tgt = target_observed.voxel_down_sample(voxel_mm)
            if len(src.points) < 30 or len(tgt.points) < 30:
                logger.warning(f"ICP level={level} 点数不足，跳过：src {len(src.points)} 点, tgt {len(tgt.points)} 点")
                continue
            _estimate_normals(src, radius_mm=max(3.0, voxel_mm * 2.0))
            _estimate_normals(tgt, radius_mm=max(3.0, voxel_mm * 2.0))
            loss = o3d.pipelines.registration.TukeyLoss(k=max(corr_mm * 0.5, 1.0))
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            reg = o3d.pipelines.registration.registration_icp(
                src,
                tgt,
                max_correspondence_distance=float(corr_mm),
                init=transform,
                estimation_method=estimation,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)),
            )
            transform = np.asarray(reg.transformation, dtype=np.float64)
            corr_cnt = len(reg.correspondence_set) if hasattr(reg, "correspondence_set") else -1
            logger.info(
                f"ICP init={init_idx} level={level} voxel {voxel_mm:.2f} mm, corr {corr_mm:.2f} mm, "
                f"fitness {reg.fitness:.4f}, rmse {reg.inlier_rmse:.4f} mm, corr_count {corr_cnt}"
            )
            if corr_cnt == 0 or float(reg.fitness) <= 1e-8:
                logger.info(f"ICP init={init_idx} level={level} 无有效对应点，提前结束该 init 的后续层级。")
                break

        final_eval = o3d.pipelines.registration.evaluate_registration(
            source_prior,
            target_observed,
            max_correspondence_distance=float(final_eval_corr_mm),
            transformation=transform,
        )
        fitness = float(final_eval.fitness)
        rmse = float(final_eval.inlier_rmse)
        if fitness > best_fitness or (abs(fitness - best_fitness) <= 1e-8 and rmse < best_rmse):
            best_fitness = fitness
            best_rmse = rmse
            best_transform = transform
    return best_transform, best_fitness, best_rmse


def _run_registration_job(
    frame_idx: int,
    det_confidence: float,
    source_prior: o3d.geometry.PointCloud,
    target_observed: o3d.geometry.PointCloud,
    yaw_candidates_deg: list[float],
    final_eval_corr_mm: float,
    icp_levels: list[tuple[float, float, int]],
) -> AsyncRegistrationResult:
    init_transforms = _build_multi_yaw_inits(
        source_prior=source_prior,
        target_observed=target_observed,
        yaw_candidates_deg=yaw_candidates_deg,
    )
    transform, fitness, rmse = _run_direct_multiscale_icp(
        source_prior=source_prior,
        target_observed=target_observed,
        init_transforms=init_transforms,
        final_eval_corr_mm=final_eval_corr_mm,
        icp_levels=icp_levels,
    )
    return AsyncRegistrationResult(
        frame_idx=int(frame_idx),
        det_confidence=float(det_confidence),
        points=int(len(target_observed.points)),
        transformation=np.asarray(transform, dtype=np.float64),
        fitness=float(fitness),
        rmse_mm=float(rmse),
        observed_cloud=o3d.geometry.PointCloud(target_observed),
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


def _init_realtime_3d_viewer() -> tuple[o3d.visualization.VisualizerWithKeyCallback, dict[str, bool], o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Stage2 Realtime 3D (observed + prior)", 1440, 900)
    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = 1.8
        render_opt.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)

    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)

    observed_vis = o3d.geometry.PointCloud()
    prior_vis = o3d.geometry.PointCloud()
    vis.add_geometry(observed_vis)
    vis.add_geometry(prior_vis)
    return vis, stop, observed_vis, prior_vis


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
    vis.reset_view_point(True)
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
    multi_init_yaw_deg: str = DEFAULT_MULTI_INIT_YAW_DEG,
    final_eval_corr_mm: float = DEFAULT_FINAL_EVAL_CORR_MM,
    icp_levels: str = DEFAULT_ICP_LEVELS,
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
        f"detect_interval {int(detect_interval)}, min_object_points {int(min_object_points)} 点, "
        f"realtime_preview {bool(realtime_preview)}, registration_interval {float(registration_interval_sec):.2f} 秒"
    )

    best: RegistrationMetrics | None = None
    yaw_candidates = _parse_float_list_csv(multi_init_yaw_deg)
    if len(yaw_candidates) == 0:
        yaw_candidates = [0.0]
    parsed_icp_levels = _parse_icp_levels(icp_levels)

    preview_win = "Stage2 MaterialPlate Direct Registration"
    if bool(realtime_preview):
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_win, 1280, 720)
    vis3d = None
    vis3d_stop = None
    observed_vis_pcd = None
    prior_vis_pcd = None
    camera_initialized = False
    if bool(vis):
        vis3d, vis3d_stop, observed_vis_pcd, prior_vis_pcd = _init_realtime_3d_viewer()

    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
    prior_for_view = o3d.geometry.PointCloud(prior_cloud)
    last_transform = np.eye(4, dtype=np.float64)
    last_async_result: AsyncRegistrationResult | None = None
    pending_future: concurrent.futures.Future[AsyncRegistrationResult] | None = None
    last_submit_ts = 0.0
    need_reset_camera = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage2_icp_worker") as executor:
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
                    if pending_future is not None and pending_future.done():
                        async_res = pending_future.result()
                        pending_future = None
                        last_async_result = async_res
                        last_transform = np.asarray(async_res.transformation, dtype=np.float64)
                        need_reset_camera = True
                        logger.info(
                            f"帧 {async_res.frame_idx} 配准完成：det_conf {async_res.det_confidence:.3f}, "
                            f"segment_points {async_res.points} 点, fitness {async_res.fitness:.4f}, rmse {async_res.rmse_mm:.4f} mm"
                        )
                        if best is None or async_res.fitness > best.fitness:
                            best = RegistrationMetrics(
                                frame_idx=int(async_res.frame_idx),
                                points=int(async_res.points),
                                fitness=float(async_res.fitness),
                                rmse_mm=float(async_res.rmse_mm),
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
                                if bool(vis) and observed_vis_pcd is not None and prior_vis_pcd is not None and vis3d is not None and vis3d_stop is not None:
                                    observed_show = o3d.geometry.PointCloud(observed_cloud)
                                    if last_async_result is not None:
                                        observed_show = o3d.geometry.PointCloud(last_async_result.observed_cloud)
                                    prior_show = o3d.geometry.PointCloud(prior_for_view)
                                    prior_show.transform(last_transform)
                                    COLORIZE_BY_CYCLE(observed_show, cycle=2.0, axis=2, color_map="hsv")
                                    COLORIZE_BY_CYCLE(prior_show, cycle=2.0, axis=2, color_map="hsv")
                                    observed_vis_pcd.points = observed_show.points
                                    observed_vis_pcd.colors = observed_show.colors
                                    prior_vis_pcd.points = prior_show.points
                                    prior_vis_pcd.colors = prior_show.colors
                                    vis3d.update_geometry(observed_vis_pcd)
                                    vis3d.update_geometry(prior_vis_pcd)
                                    if not camera_initialized or need_reset_camera:
                                        _fit_camera_to_clouds(vis=vis3d, observed=observed_show, prior=prior_show)
                                        camera_initialized = True
                                        need_reset_camera = False
                                    alive = vis3d.poll_events()
                                    vis3d.update_renderer()
                                    if vis3d_stop["flag"] or (not alive):
                                        logger.warning("3D 预览窗口关闭，提前结束。")
                                        break

                                now_ts = time.monotonic()
                                if (pending_future is None) and (
                                    now_ts - last_submit_ts >= max(0.2, float(registration_interval_sec))
                                ):
                                    source_prior_job = o3d.geometry.PointCloud(prior_cloud)
                                    target_observed_job = o3d.geometry.PointCloud(observed_cloud)
                                    pending_future = executor.submit(
                                        _run_registration_job,
                                        int(frame_idx),
                                        float(det.confidence_2d),
                                        source_prior_job,
                                        target_observed_job,
                                        yaw_candidates,
                                        float(final_eval_corr_mm),
                                        parsed_icp_levels,
                                    )
                                    last_submit_ts = now_ts
                                if pending_future is None:
                                    metrics_text = (
                                        f"frame {frame_idx} | det {det.confidence_2d:.3f} | pts {len(observed_cloud.points)} | idle"
                                    )
                                else:
                                    metrics_text = (
                                        f"frame {frame_idx} | det {det.confidence_2d:.3f} | pts {len(observed_cloud.points)} | icp running"
                                    )
                            else:
                                metrics_text = (
                                    f"frame {frame_idx} | det {det.confidence_2d:.3f} | filtered points {len(observed_cloud.points)}"
                                )
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
            if pending_future is not None and (not pending_future.done()):
                pending_future.cancel()
            if bool(vis) and vis3d is not None:
                vis3d.destroy_window()
            if bool(realtime_preview):
                cv2.destroyWindow(preview_win)

    if best is None:
        logger.warning("未获得可用的目标分割点云，未执行到有效配准。")
        return

    logger.success(
        f"最佳配准：frame {best.frame_idx}, points {best.points} 点, "
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
    bool,
    int,
    float,
    bool,
    float,
    int,
    str,
    float,
    str,
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
    parser.add_argument("--save-result", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_RESULT, help="是否保存结果")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最大采样帧数")
    parser.add_argument("--detect-interval", type=int, default=DEFAULT_DETECT_INTERVAL, help="每 N 帧检测一次")
    parser.add_argument("--min-object-points", type=int, default=DEFAULT_MIN_OBJECT_POINTS, help="最小目标点数")
    parser.add_argument("--segment-voxel-mm", type=float, default=DEFAULT_SEGMENT_VOXEL_MM, help="观测体素尺寸，单位 mm")
    parser.add_argument("--prior-voxel-mm", type=float, default=DEFAULT_PRIOR_VOXEL_MM, help="先验体素尺寸，单位 mm")
    parser.add_argument("--min-accept-fitness", type=float, default=DEFAULT_MIN_ACCEPT_FITNESS, help="直接配准 fitness 阈值")
    parser.add_argument("--max-accept-rmse-mm", type=float, default=DEFAULT_MAX_ACCEPT_RMSE_MM, help="直接配准 RMSE 阈值，单位 mm")
    parser.add_argument("--remove-outlier", action=argparse.BooleanOptionalAction, default=DEFAULT_REMOVE_OUTLIER, help="是否做离群点剔除")
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
    parser.add_argument("--multi-init-yaw-deg", type=str, default=DEFAULT_MULTI_INIT_YAW_DEG, help="多初值 yaw 候选，逗号分隔（度）")
    parser.add_argument("--final-eval-corr-mm", type=float, default=DEFAULT_FINAL_EVAL_CORR_MM, help="最终评估对应距离，单位 mm")
    parser.add_argument("--icp-levels", type=str, default=DEFAULT_ICP_LEVELS, help="ICP 层级，格式 voxel:corr:iters,逗号分隔")
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
        str(args.multi_init_yaw_deg),
        float(args.final_eval_corr_mm),
        str(args.icp_levels),
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
                multi_init_yaw_deg_arg,
                final_eval_corr_mm_arg,
                icp_levels_arg,
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
                multi_init_yaw_deg=multi_init_yaw_deg_arg,
                final_eval_corr_mm=final_eval_corr_mm_arg,
                icp_levels=icp_levels_arg,
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

