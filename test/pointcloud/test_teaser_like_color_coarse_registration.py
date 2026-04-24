from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.practical_color_coarse_registration import (
    TeaserLikeParams,
    teaser_like_color_coarse_registration,
)
from src.utils.datas.kinematics.se3 import SE3_string

DEFAULT_SRC = PROJECT_ROOT / "experiments" / "pcd2.pcd"
DEFAULT_TGT = PROJECT_ROOT / "experiments" / "pcd1.pcd"
DEFAULT_VIS = True
DEFAULT_ENABLE_SWEEP = True


def visualize_before_after(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform_source_to_target: np.ndarray,
) -> None:
    raw_src = o3d.geometry.PointCloud(source)
    raw_tgt = o3d.geometry.PointCloud(target)

    reg_src = o3d.geometry.PointCloud(source)
    reg_tgt = o3d.geometry.PointCloud(target)
    reg_src.transform(transform_source_to_target)
    offset_reg = np.array([0.0, 1000.0, 0.0], dtype=np.float64)
    reg_src.translate(offset_reg)
    reg_tgt.translate(offset_reg)

    axis_raw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    axis_reg = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=offset_reg.tolist())

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        "TEASER-like 配准对比：原始 / 配准后 (+Y 1000mm)",
        1440,
        900,
    )
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


def _score_result(r, p: TeaserLikeParams) -> float:
    return (
        1.4 * r.fitness_strict
        + 1.0 * r.fitness_wide
        - 0.08 * r.rmse_strict
        - 0.04 * r.rmse_wide
        - p.score_color_weight * r.color_residual
    )


def _is_better(candidate, best, p: TeaserLikeParams, best_p: TeaserLikeParams | None) -> bool:
    if best is None:
        return True
    cand_key = (
        candidate.fitness_strict,
        candidate.fitness_wide,
        -candidate.rmse_strict,
        -candidate.rmse_wide,
        -candidate.color_residual,
        _score_result(candidate, p),
    )
    best_key = (
        best.fitness_strict,
        best.fitness_wide,
        -best.rmse_strict,
        -best.rmse_wide,
        -best.color_residual,
        _score_result(best, best_p if best_p is not None else p),
    )
    return cand_key > best_key


def main(
    src_path: Path = DEFAULT_SRC,
    tgt_path: Path = DEFAULT_TGT,
    vis: bool = DEFAULT_VIS,
    enable_sweep: bool = DEFAULT_ENABLE_SWEEP,
) -> None:
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(f"点云不存在：src={src_path}, tgt={tgt_path}")

    source = o3d.io.read_point_cloud(str(src_path))
    target = o3d.io.read_point_cloud(str(tgt_path))
    if len(source.points) == 0 or len(target.points) == 0:
        raise RuntimeError("存在空点云。")

    result = None
    if enable_sweep:
        base = TeaserLikeParams(
            noise_bound_mm=6.0,
            color_gate=0.60,
            fpfh_ratio_test=0.92,
            voxel_mm=8.0,
            color_weight_power=1.8,
            score_color_weight=1.10,
        )

        best_params: TeaserLikeParams | None = None

        # Stage A: 仅调整 score_color_weight（控制其余参数不变）
        stage_a = [1.10, 1.30, 1.50, 1.70, 2.00]
        logger.info(f"[stage-A] 扫描 score_color_weight: {stage_a}")
        for idx, val in enumerate(stage_a):
            p = replace(base, score_color_weight=val)
            try:
                r = teaser_like_color_coarse_registration(source, target, params=p)
            except Exception as exc:
                logger.warning(f"[stage-A-{idx}] failed: {exc}")
                continue
            score = _score_result(r, p)
            logger.info(
                f"[stage-A-{idx}] score={score:.5f} method={r.method_name} "
                f"score_color_weight={p.score_color_weight:.2f} "
                f"noise={r.used_noise_bound_mm:.3f} strict={r.fitness_strict:.5f} "
                f"wide={r.fitness_wide:.5f} color={r.color_residual:.5f}"
            )
            if _is_better(r, result, p, best_params):
                result = r
                best_params = p
        if result is None or best_params is None:
            raise RuntimeError("stage-A 参数搜索未得到有效结果。")
        logger.success(
            f"[stage-A] best: score_color_weight={best_params.score_color_weight:.2f}, "
            f"strict={result.fitness_strict:.5f}, wide={result.fitness_wide:.5f}, color={result.color_residual:.5f}"
        )

        # Stage B: 仅调整 color_weight_power（控制其余参数不变）
        stage_b = [1.8, 2.0, 2.2, 2.4, 2.6]
        logger.info(f"[stage-B] 扫描 color_weight_power: {stage_b}")
        for idx, val in enumerate(stage_b):
            p = replace(best_params, color_weight_power=val)
            try:
                r = teaser_like_color_coarse_registration(source, target, params=p)
            except Exception as exc:
                logger.warning(f"[stage-B-{idx}] failed: {exc}")
                continue
            score = _score_result(r, p)
            logger.info(
                f"[stage-B-{idx}] score={score:.5f} method={r.method_name} "
                f"color_weight_power={p.color_weight_power:.2f} "
                f"noise={r.used_noise_bound_mm:.3f} strict={r.fitness_strict:.5f} "
                f"wide={r.fitness_wide:.5f} color={r.color_residual:.5f}"
            )
            if _is_better(r, result, p, best_params):
                result = r
                best_params = p
        logger.success(
            f"[stage-B] best: color_weight_power={best_params.color_weight_power:.2f}, "
            f"strict={result.fitness_strict:.5f}, wide={result.fitness_wide:.5f}, color={result.color_residual:.5f}"
        )

        # Stage C: 收紧 color_gate（控制其余参数不变）
        stage_c = [0.60, 0.55, 0.50, 0.45]
        logger.info(f"[stage-C] 收紧 color_gate: {stage_c}")
        for idx, val in enumerate(stage_c):
            p = replace(best_params, color_gate=val)
            try:
                r = teaser_like_color_coarse_registration(source, target, params=p)
            except Exception as exc:
                logger.warning(f"[stage-C-{idx}] failed: {exc}")
                continue
            score = _score_result(r, p)
            logger.info(
                f"[stage-C-{idx}] score={score:.5f} method={r.method_name} "
                f"color_gate={p.color_gate:.2f} "
                f"noise={r.used_noise_bound_mm:.3f} strict={r.fitness_strict:.5f} "
                f"wide={r.fitness_wide:.5f} color={r.color_residual:.5f}"
            )
            if _is_better(r, result, p, best_params):
                result = r
                best_params = p

        logger.success(
            f"[stage-C] best params: score_color_weight={best_params.score_color_weight:.2f}, "
            f"color_weight_power={best_params.color_weight_power:.2f}, color_gate={best_params.color_gate:.2f}"
        )
        if result is None:
            raise RuntimeError("参数搜索未得到有效结果。")
    else:
        result = teaser_like_color_coarse_registration(source, target)

    logger.success(f"raw_matches={result.num_raw_matches}, filtered_nodes={result.num_filtered_matches}")
    logger.success(f"translation_inliers={result.num_inlier_nodes}")
    logger.success(f"chosen_method={result.method_name}, used_noise_bound_mm={result.used_noise_bound_mm:.3f}")
    logger.success(f"T(src->tgt): {SE3_string(result.transform)}")
    logger.success(
        f"strict(fitness={result.fitness_strict:.5f}, rmse={result.rmse_strict:.5f}), "
        f"wide(fitness={result.fitness_wide:.5f}, rmse={result.rmse_wide:.5f}), "
        f"color_residual={result.color_residual:.5f}"
    )

    if vis:
        visualize_before_after(source, target, result.transform)


def _parse_cli() -> tuple[Path, Path, bool, bool]:
    parser = argparse.ArgumentParser(description="TEASER-like 粗配准测试（支持 CLI + IDE 默认运行）")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="source pcd path")
    parser.add_argument("--tgt", type=Path, default=DEFAULT_TGT, help="target pcd path")
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=DEFAULT_VIS, help="show visualizer")
    parser.add_argument(
        "--sweep",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_SWEEP,
        help="enable parameter sweep",
    )
    args = parser.parse_args()
    return args.src, args.tgt, bool(args.vis), bool(args.sweep)


if __name__ == "__main__":
    src_arg, tgt_arg, vis_arg, sweep_arg = _parse_cli()
    main(src_path=src_arg, tgt_path=tgt_arg, vis=vis_arg, enable_sweep=sweep_arg)
