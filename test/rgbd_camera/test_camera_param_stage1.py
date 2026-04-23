from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import (
    CameraParamPatch,
    DistortionPatch,
    IntrinsicPatch,
    OrbbecSession,
    SessionOptions,
    apply_camera_param_patch,
    camera_param_summary,
    clone_camera_param,
    filter_valid_points,
    normalize_points,
    set_point_cloud_filter_format,
)

DEFAULT_FRAMES = 20
DEFAULT_WARMUP_FRAMES = 10
DEFAULT_TIMEOUT_MS = 120
DEFAULT_MAX_DEPTH_MM = 4500.0
DEFAULT_SAVE_PLY = False
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "camera_param_stage1"
DEFAULT_PROXY_VOXEL_MM = 8.0
DEFAULT_PROXY_MAX_CORR_MM = 25.0
DEFAULT_PROXY_MAX_ITER = 20
DEFAULT_PROXY_MAX_POINTS = 30000


CAMERA_PARAM_PRESETS: dict[str, CameraParamPatch] = {
    "baseline": CameraParamPatch(),
    "depth_fx_fy_plus5": CameraParamPatch(depth=IntrinsicPatch(fx_scale=1.05, fy_scale=1.05)),
    "depth_fx_fy_minus5": CameraParamPatch(depth=IntrinsicPatch(fx_scale=0.95, fy_scale=0.95)),
    "depth_cx_plus10": CameraParamPatch(depth=IntrinsicPatch(cx_offset=10.0)),
    "depth_cx_minus10": CameraParamPatch(depth=IntrinsicPatch(cx_offset=-10.0)),
    "depth_cy_plus10": CameraParamPatch(depth=IntrinsicPatch(cy_offset=10.0)),
    "depth_k1_plus0p01": CameraParamPatch(depth_dist=DistortionPatch(k1_offset=0.01)),
    "d2c_tx_plus5mm": CameraParamPatch(d2c_translation_offset_mm=(5.0, 0.0, 0.0)),
}


@dataclass
class PresetStats:
    sampled_frames: int = 0
    temporal_mean_mm_sum: float = 0.0
    temporal_p95_mm_sum: float = 0.0
    temporal_frames: int = 0
    proxy_fitness_sum: float = 0.0
    proxy_rmse_sum: float = 0.0
    proxy_frames: int = 0

    def update(
        self,
        temporal_mean_mm: float | None = None,
        temporal_p95_mm: float | None = None,
        proxy_fitness: float | None = None,
        proxy_rmse: float | None = None,
    ) -> None:
        self.sampled_frames += 1
        if temporal_mean_mm is not None and temporal_p95_mm is not None:
            self.temporal_mean_mm_sum += float(temporal_mean_mm)
            self.temporal_p95_mm_sum += float(temporal_p95_mm)
            self.temporal_frames += 1
        if proxy_fitness is not None and proxy_rmse is not None:
            self.proxy_fitness_sum += float(proxy_fitness)
            self.proxy_rmse_sum += float(proxy_rmse)
            self.proxy_frames += 1

    def avg_temporal_mean_mm(self) -> float:
        return self.temporal_mean_mm_sum / self.temporal_frames if self.temporal_frames else float("inf")

    def avg_temporal_p95_mm(self) -> float:
        return self.temporal_p95_mm_sum / self.temporal_frames if self.temporal_frames else float("inf")

    def avg_proxy_fitness(self) -> float:
        return self.proxy_fitness_sum / self.proxy_frames if self.proxy_frames else 0.0

    def avg_proxy_rmse_mm(self) -> float:
        return self.proxy_rmse_sum / self.proxy_frames if self.proxy_frames else float("inf")

    def quality_score(self) -> float:
        if self.temporal_frames == 0 or self.proxy_frames == 0:
            return float("-inf")
        return (
            2.0 * self.avg_proxy_fitness()
            - 0.02 * self.avg_proxy_rmse_mm()
            - 0.02 * self.avg_temporal_mean_mm()
            - 0.01 * self.avg_temporal_p95_mm()
        )


@dataclass
class PresetEvaluator:
    name: str
    patch: CameraParamPatch
    stats: PresetStats = field(default_factory=PresetStats)
    last_valid_points: np.ndarray | None = None
    prev_xyz: np.ndarray | None = None
    prev_valid_mask: np.ndarray | None = None
    skipped_empty_frames: int = 0


def main(
    frames: int = DEFAULT_FRAMES,
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    save_ply: bool = DEFAULT_SAVE_PLY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    if frames <= 0:
        raise ValueError("frames must be > 0")

    session_options = SessionOptions(timeout_ms=timeout_ms)
    evaluators: list[PresetEvaluator] = [
        PresetEvaluator(name=name, patch=patch) for name, patch in CAMERA_PARAM_PRESETS.items()
    ]

    with OrbbecSession(options=session_options) as session:
        base_camera_param = session.get_camera_param()
        for idx, evaluator in enumerate(evaluators, start=1):
            evaluator.last_valid_points = None
            evaluator.prev_xyz = None
            evaluator.prev_valid_mask = None
            evaluator.skipped_empty_frames = 0

            param = clone_camera_param(base_camera_param)
            apply_camera_param_patch(param, evaluator.patch)
            point_filter = session.create_point_cloud_filter(camera_param=param)
            logger.info(f"[preset {idx}/{len(evaluators)}] {camera_param_summary(evaluator.name, param)}")

            for _ in range(max(0, warmup_frames)):
                session.wait_for_frames()

            sampled = 0
            while sampled < frames:
                frameset = session.wait_for_frames()
                if frameset is None:
                    evaluator.skipped_empty_frames += 1
                    continue

                depth_frame = frameset.get_depth_frame()
                if depth_frame is None:
                    evaluator.skipped_empty_frames += 1
                    continue

                point_frames, use_color = session.prepare_frame_for_point_cloud(frameset)
                depth_scale = float(depth_frame.get_depth_scale())

                set_point_cloud_filter_format(point_filter, depth_scale=depth_scale, use_color=use_color)
                cloud_frame = point_filter.process(point_frames)
                if cloud_frame is None:
                    evaluator.stats.update()
                    sampled += 1
                    continue

                raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
                normalized = normalize_points(raw_points)
                valid_points, _ = filter_valid_points(normalized, max_depth_mm=max_depth_mm)
                if len(valid_points) == 0:
                    evaluator.stats.update()
                    sampled += 1
                    continue

                xyz = normalized[:, :3]
                valid_mask = np.isfinite(xyz).all(axis=1)
                valid_mask &= xyz[:, 2] > 0.0
                valid_mask &= xyz[:, 2] <= max_depth_mm

                temporal_mean: float | None = None
                temporal_p95: float | None = None
                if (
                    evaluator.prev_xyz is not None
                    and evaluator.prev_valid_mask is not None
                    and evaluator.prev_xyz.shape == xyz.shape
                ):
                    overlap = valid_mask & evaluator.prev_valid_mask
                    if np.any(overlap):
                        delta = np.linalg.norm(
                            (xyz[overlap] - evaluator.prev_xyz[overlap]).astype(np.float64),
                            axis=1,
                        )
                        temporal_mean = float(np.mean(delta))
                        temporal_p95 = float(np.percentile(delta, 95.0))

                proxy_fitness, proxy_rmse = _estimate_registration_proxy(
                    source_points=valid_points,
                    target_points=evaluator.last_valid_points,
                    voxel_mm=DEFAULT_PROXY_VOXEL_MM,
                    max_corr_mm=DEFAULT_PROXY_MAX_CORR_MM,
                    max_iter=DEFAULT_PROXY_MAX_ITER,
                    max_points=DEFAULT_PROXY_MAX_POINTS,
                )

                evaluator.stats.update(
                    temporal_mean_mm=temporal_mean,
                    temporal_p95_mm=temporal_p95,
                    proxy_fitness=proxy_fitness,
                    proxy_rmse=proxy_rmse,
                )
                evaluator.last_valid_points = valid_points
                evaluator.prev_xyz = xyz.copy()
                evaluator.prev_valid_mask = valid_mask.copy()
                sampled += 1
                # logger.info(f"[{evaluator.name}] sampled frame {sampled}/{frames}")

            logger.success(
                f"[{evaluator.name}] capture done: sampled={sampled}, skipped_empty={evaluator.skipped_empty_frames}"
            )

    ranked = sorted(
        evaluators,
        key=lambda it: it.stats.quality_score(),
        reverse=True,
    )

    logger.info("==== stage-1 camera param ranking ====")
    for idx, item in enumerate(ranked, start=1):
        s = item.stats
        logger.info(
            f"#{idx} {item.name}: "
            f"frames={s.sampled_frames}, "
            f"skipped_empty={item.skipped_empty_frames}, "
            f"temporal_frames={s.temporal_frames}, "
            f"avg_temporal_mean_mm={s.avg_temporal_mean_mm():.4f}, "
            f"avg_temporal_p95_mm={s.avg_temporal_p95_mm():.4f}, "
            f"proxy_frames={s.proxy_frames}, "
            f"avg_proxy_fitness={s.avg_proxy_fitness():.5f}, "
            f"avg_proxy_rmse_mm={s.avg_proxy_rmse_mm():.4f}, "
            f"quality_score={s.quality_score():.5f}"
        )

    if save_ply:
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in ranked:
            if item.last_valid_points is None or len(item.last_valid_points) == 0:
                continue
            xyz = np.ascontiguousarray(item.last_valid_points[:, :3], dtype=np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            if item.last_valid_points.shape[1] >= 6:
                rgb = np.ascontiguousarray(item.last_valid_points[:, 3:6], dtype=np.float32)
                if float(np.max(rgb)) > 1.0:
                    rgb = rgb / 255.0
                pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0).astype(np.float64))
            output_path = output_dir / f"{item.name}.ply"
            o3d.io.write_point_cloud(str(output_path), pcd)
            logger.info(f"saved: {output_path}")


def _parse_cli() -> tuple[int, int, int, float, bool, Path]:
    parser = argparse.ArgumentParser(description="Stage-1 camera parameter tuning test")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="sampled frames for each preset")
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES, help="warmup frames before stats")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM, help="max depth for valid points")
    parser.add_argument(
        "--save-ply",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_PLY,
        help="save per-preset point cloud",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="output dir when --save-ply")
    args = parser.parse_args()
    return (
        int(args.frames),
        int(args.warmup_frames),
        int(args.timeout_ms),
        float(args.max_depth_mm),
        bool(args.save_ply),
        Path(args.output_dir),
    )


def _estimate_registration_proxy(
    source_points: np.ndarray,
    target_points: np.ndarray | None,
    voxel_mm: float,
    max_corr_mm: float,
    max_iter: int,
    max_points: int,
) -> tuple[float | None, float | None]:
    if target_points is None or len(source_points) < 200 or len(target_points) < 200:
        return None, None

    src = _downsample_for_proxy(source_points[:, :3], max_points=max_points)
    tgt = _downsample_for_proxy(target_points[:, :3], max_points=max_points)
    if len(src) < 100 or len(tgt) < 100:
        return None, None

    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(src, dtype=np.float64))
    tgt_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(tgt, dtype=np.float64))

    if voxel_mm > 0:
        src_pcd = src_pcd.voxel_down_sample(voxel_size=float(voxel_mm))
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=float(voxel_mm))
        if len(src_pcd.points) < 50 or len(tgt_pcd.points) < 50:
            return None, None

    reg = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        max_correspondence_distance=float(max_corr_mm),
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
    )
    return float(reg.fitness), float(reg.inlier_rmse)


def _downsample_for_proxy(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if len(points_xyz) <= max_points:
        return points_xyz
    step = max(1, len(points_xyz) // max_points)
    return points_xyz[::step]


if __name__ == "__main__":
    f_arg, warmup_arg, timeout_arg, max_depth_arg, save_arg, output_dir_arg = _parse_cli()
    main(
        frames=f_arg,
        warmup_frames=warmup_arg,
        timeout_ms=timeout_arg,
        max_depth_mm=max_depth_arg,
        save_ply=save_arg,
        output_dir=output_dir_arg,
    )
