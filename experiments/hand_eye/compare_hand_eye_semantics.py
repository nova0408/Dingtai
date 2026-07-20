from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rotation3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.hand_eye import HandEyeMethodName, calibrate_hand_eye_multi_method

PairMode = Literal["adjacent", "all"]


@dataclass(frozen=True, slots=True)
class SampleEntry:
    sample_index: int
    robot_pose_base_end: np.ndarray
    board_pose_camera_board: np.ndarray


@dataclass(frozen=True, slots=True)
class SemanticVariantResult:
    robot_semantics: str
    board_semantics: str
    method_name: str
    rotation_rmse_deg: float
    translation_rmse_mm: float
    cv_rotation_rmse_deg: float | None
    cv_translation_rmse_mm: float | None
    score: float | None
    error_message: str | None
    transform_matrix: np.ndarray | None


@dataclass(frozen=True, slots=True)
class SubsetResult:
    sample_indices: tuple[int, ...]
    method_name: str
    rotation_rmse_deg: float
    translation_rmse_mm: float
    cv_rotation_rmse_deg: float | None
    cv_translation_rmse_mm: float | None
    score: float | None
    error_message: str | None


def main() -> int:
    args = _parse_cli()
    samples = _load_samples(Path(args.samples_csv))
    variants = _build_variants(samples)
    results = _evaluate_variants(
        variants=variants,
        pair_mode=args.pair_mode,
        methods=None,
        cv_folds=args.cv_folds,
    )
    subset_results = _evaluate_subsets(
        samples=samples,
        pair_mode=args.pair_mode,
        cv_folds=args.cv_folds,
        board_semantics="T_board_camera",
        robot_semantics="T_ref_end",
        subset_size=args.subset_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hand_eye_semantics_comparison.txt"
    _write_report(report_path, results, len(samples))
    _write_subset_report(output_dir / "hand_eye_subset_comparison.txt", subset_results, len(samples))
    _print_summary(results)
    _print_subset_summary(subset_results)
    _print_sample_participation_summary(subset_results, len(samples))
    logger.success("结果已写入: {}", report_path)
    return 0


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 samples.csv 的手眼语义对比脚本")
    parser.add_argument(
        "--samples-csv",
        type=str,
        default=str(Path("experiments/hand_eye/runs/20260708_111018/samples.csv")),
        help="采样 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("experiments/hand_eye/runs/20260708_111018/semantics_compare")),
        help="结果输出目录",
    )
    parser.add_argument(
        "--pair-mode",
        type=str,
        default="all",
        choices=["adjacent", "all"],
        help="相对运动构造方式",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="交叉验证折数；不传则沿用默认自动策略",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=3,
        help="穷举子集大小，默认 3",
    )
    return parser.parse_args()


def _load_samples(samples_csv: Path) -> list[SampleEntry]:
    if not samples_csv.exists():
        raise FileNotFoundError(f"找不到 samples.csv: {samples_csv}")
    entries: list[SampleEntry] = []
    with samples_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("board_visible", "0") != "1":
                continue
            camera_board = _pose_from_csv(row, prefix="camera_board")
            board_camera = _pose_from_csv(row, prefix="board_camera")
            if camera_board is None or board_camera is None:
                continue
            robot_pose = _robot_pose_from_csv(row)
            entries.append(
                SampleEntry(
                    sample_index=int(row["sample_index"]),
                    robot_pose_base_end=robot_pose,
                    board_pose_camera_board=camera_board,
                )
            )
    if len(entries) < 3:
        raise ValueError(f"有效样本太少，当前只有 {len(entries)} 个")
    return entries


def _pose_from_csv(row: dict[str, str], prefix: str) -> np.ndarray | None:
    x_key = f"{prefix}_x_mm"
    y_key = f"{prefix}_y_mm"
    z_key = f"{prefix}_z_mm"
    qw_key = f"{prefix}_qw"
    qx_key = f"{prefix}_qx"
    qy_key = f"{prefix}_qy"
    qz_key = f"{prefix}_qz"
    if row.get(x_key, "") == "":
        return None
    translation = np.array([float(row[x_key]), float(row[y_key]), float(row[z_key])], dtype=np.float64)
    quat_wxyz = np.array(
        [float(row[qw_key]), float(row[qx_key]), float(row[qy_key]), float(row[qz_key])], dtype=np.float64
    )
    rotation = Rotation3D.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _robot_pose_from_csv(row: dict[str, str]) -> np.ndarray:
    translation = np.array(
        [float(row["robot_tcp_x_mm"]), float(row["robot_tcp_y_mm"]), float(row["robot_tcp_z_mm"])],
        dtype=np.float64,
    )
    rpy_deg = np.array(
        [float(row["robot_tcp_roll_deg"]), float(row["robot_tcp_pitch_deg"]), float(row["robot_tcp_yaw_deg"])],
        dtype=np.float64,
    )
    rotation = Rotation3D.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _build_variants(samples: list[SampleEntry]) -> list[tuple[str, str, list[np.ndarray], list[np.ndarray]]]:
    robot_seq = [sample.robot_pose_base_end for sample in samples]
    board_seq = [sample.board_pose_camera_board for sample in samples]
    robot_seq_inv = [_invert_se3(pose) for pose in robot_seq]
    board_seq_inv = [_invert_se3(pose) for pose in board_seq]
    return [
        ("T_ref_end", "T_board_camera", robot_seq, board_seq),
        ("T_ref_end", "T_camera_board", robot_seq, board_seq_inv),
        ("T_end_ref", "T_board_camera", robot_seq_inv, board_seq),
        ("T_end_ref", "T_camera_board", robot_seq_inv, board_seq_inv),
    ]


def _evaluate_variants(
    variants: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]],
    pair_mode: PairMode,
    methods: list[HandEyeMethodName] | None,
    cv_folds: int | None,
) -> list[SemanticVariantResult]:
    results: list[SemanticVariantResult] = []
    for robot_semantics, board_semantics, robot_seq, board_seq in variants:
        try:
            multi = calibrate_hand_eye_multi_method(
                group_a_poses=robot_seq,
                group_b_poses=board_seq,
                pair_mode=pair_mode,
                methods=methods,
                cv_folds=cv_folds,
            )
            best = multi.best_result
            if best is None or best.transform is None or best.residual is None:
                raise RuntimeError("没有可用的候选解")
            results.append(
                SemanticVariantResult(
                    robot_semantics=robot_semantics,
                    board_semantics=board_semantics,
                    method_name=best.method_name,
                    rotation_rmse_deg=best.residual.rotation_rmse_deg,
                    translation_rmse_mm=best.residual.translation_rmse,
                    cv_rotation_rmse_deg=(
                        None if best.cv_residual is None else best.cv_residual.val_rotation_rmse_deg_mean
                    ),
                    cv_translation_rmse_mm=(
                        None if best.cv_residual is None else best.cv_residual.val_translation_rmse_mean
                    ),
                    score=best.score,
                    error_message=None,
                    transform_matrix=np.asarray(best.transform.as_SE3(), dtype=np.float64).reshape(4, 4),
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                SemanticVariantResult(
                    robot_semantics=robot_semantics,
                    board_semantics=board_semantics,
                    method_name="failed",
                    rotation_rmse_deg=float("inf"),
                    translation_rmse_mm=float("inf"),
                    cv_rotation_rmse_deg=None,
                    cv_translation_rmse_mm=None,
                    score=float("inf"),
                    error_message=str(exc),
                    transform_matrix=None,
                )
            )
    return sorted(results, key=lambda item: float(item.score or float("inf")))


def _evaluate_subsets(
    samples: list[SampleEntry],
    pair_mode: PairMode,
    cv_folds: int | None,
    robot_semantics: str,
    board_semantics: str,
    subset_size: int,
) -> list[SubsetResult]:
    if subset_size < 3:
        raise ValueError("subset_size 至少为 3")
    indices = list(range(len(samples)))
    robot_seq = [sample.robot_pose_base_end for sample in samples]
    board_seq = [sample.board_pose_camera_board for sample in samples]
    if robot_semantics == "T_end_ref":
        robot_seq = [_invert_se3(pose) for pose in robot_seq]
    if board_semantics == "T_camera_board":
        board_seq = [_invert_se3(pose) for pose in board_seq]

    subset_results: list[SubsetResult] = []
    for subset in itertools.combinations(indices, subset_size):
        sub_robot = [robot_seq[index] for index in subset]
        sub_board = [board_seq[index] for index in subset]
        try:
            multi = calibrate_hand_eye_multi_method(
                group_a_poses=sub_robot,
                group_b_poses=sub_board,
                pair_mode=pair_mode,
                methods=None,
                cv_folds=cv_folds,
            )
            best = multi.best_result
            if best is None or best.transform is None or best.residual is None:
                raise RuntimeError("没有可用候选解")
            subset_results.append(
                SubsetResult(
                    sample_indices=tuple(samples[index].sample_index for index in subset),
                    method_name=best.method_name,
                    rotation_rmse_deg=best.residual.rotation_rmse_deg,
                    translation_rmse_mm=best.residual.translation_rmse,
                    cv_rotation_rmse_deg=(
                        None if best.cv_residual is None else best.cv_residual.val_rotation_rmse_deg_mean
                    ),
                    cv_translation_rmse_mm=(
                        None if best.cv_residual is None else best.cv_residual.val_translation_rmse_mean
                    ),
                    score=best.score,
                    error_message=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            subset_results.append(
                SubsetResult(
                    sample_indices=tuple(samples[index].sample_index for index in subset),
                    method_name="failed",
                    rotation_rmse_deg=float("inf"),
                    translation_rmse_mm=float("inf"),
                    cv_rotation_rmse_deg=None,
                    cv_translation_rmse_mm=None,
                    score=float("inf"),
                    error_message=str(exc),
                )
            )
    return sorted(subset_results, key=lambda item: float(item.score or float("inf")))


def _write_report(report_path: Path, results: list[SemanticVariantResult], sample_count: int) -> None:
    lines = [
        "hand_eye semantics comparison",
        f"sample_count={sample_count}",
        "",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"[{index}] robot={result.robot_semantics} board={result.board_semantics} method={result.method_name}",
                f"rotation_rmse_deg={result.rotation_rmse_deg:.6f}",
                f"translation_rmse_mm={result.translation_rmse_mm:.6f}",
                f"cv_rotation_rmse_deg={_fmt_optional(result.cv_rotation_rmse_deg)}",
                f"cv_translation_rmse_mm={_fmt_optional(result.cv_translation_rmse_mm)}",
                f"score={_fmt_optional(result.score)}",
                f"error_message={result.error_message or 'None'}",
            ]
        )
        if result.transform_matrix is not None:
            lines.append("transform_matrix_se3=")
            for row in result.transform_matrix:
                lines.append("  " + ", ".join(f"{float(value): .8f}" for value in row))
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subset_report(report_path: Path, results: list[SubsetResult], sample_count: int) -> None:
    lines = [
        "hand_eye subset comparison",
        f"sample_count={sample_count}",
        f"subset_count={len(results)}",
        "",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"[{index}] samples={list(result.sample_indices)} method={result.method_name}",
                f"rotation_rmse_deg={result.rotation_rmse_deg:.6f}",
                f"translation_rmse_mm={result.translation_rmse_mm:.6f}",
                f"cv_rotation_rmse_deg={_fmt_optional(result.cv_rotation_rmse_deg)}",
                f"cv_translation_rmse_mm={_fmt_optional(result.cv_translation_rmse_mm)}",
                f"score={_fmt_optional(result.score)}",
                f"error_message={result.error_message or 'None'}",
            ]
        )
        lines.append("")
    participation = _build_participation_summary(results, sample_count)
    lines.append("[sample_participation]")
    for sample_index, stats in participation.items():
        lines.append(
            f"sample_{sample_index}: count={stats['count']} mean_score={_fmt_optional(stats['mean_score'])} "
            f"mean_rot_rmse={_fmt_optional(stats['mean_rot_rmse'])} mean_trans_rmse={_fmt_optional(stats['mean_trans_rmse'])}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(results: list[SemanticVariantResult]) -> None:
    logger.info("语义对比结果按 score 排序:")
    for result in results:
        logger.info(
            "robot={} board={} method={} rot_rmse={}deg trans_rmse={}mm score={}",
            result.robot_semantics,
            result.board_semantics,
            result.method_name,
            _fmt_optional(result.rotation_rmse_deg),
            _fmt_optional(result.translation_rmse_mm),
            _fmt_optional(result.score),
        )


def _print_subset_summary(results: list[SubsetResult]) -> None:
    logger.info("3样本穷举结果按 score 排序:")
    for result in results[:10]:
        logger.info(
            "samples={} method={} rot_rmse={}deg trans_rmse={}mm score={}",
            list(result.sample_indices),
            result.method_name,
            _fmt_optional(result.rotation_rmse_deg),
            _fmt_optional(result.translation_rmse_mm),
            _fmt_optional(result.score),
        )


def _print_sample_participation_summary(results: list[SubsetResult], sample_count: int) -> None:
    participation = _build_participation_summary(results, sample_count)
    logger.info("样本参与统计:")
    for sample_index, stats in participation.items():
        logger.info(
            "sample={} count={} mean_score={} mean_rot_rmse={}deg mean_trans_rmse={}mm",
            sample_index,
            stats["count"],
            _fmt_optional(stats["mean_score"]),
            _fmt_optional(stats["mean_rot_rmse"]),
            _fmt_optional(stats["mean_trans_rmse"]),
        )


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    if not np.isfinite(value):
        return "inf"
    return f"{float(value):.6f}"


def _build_participation_summary(
    results: list[SubsetResult],
    sample_count: int,
) -> dict[int, dict[str, float]]:
    summary: dict[int, dict[str, float]] = {
        sample_index + 1: {"count": 0.0, "score_sum": 0.0, "rot_sum": 0.0, "trans_sum": 0.0}
        for sample_index in range(sample_count)
    }
    for result in results:
        for sample_index in result.sample_indices:
            stats = summary[sample_index]
            stats["count"] += 1.0
            stats["score_sum"] += float(result.score or float("inf"))
            stats["rot_sum"] += float(result.rotation_rmse_deg)
            stats["trans_sum"] += float(result.translation_rmse_mm)
    output: dict[int, dict[str, float]] = {}
    for sample_index, stats in summary.items():
        count = stats["count"]
        if count <= 0.0:
            continue
        output[sample_index] = {
            "count": count,
            "mean_score": stats["score_sum"] / count,
            "mean_rot_rmse": stats["rot_sum"] / count,
            "mean_trans_rmse": stats["trans_sum"] / count,
        }
    return dict(sorted(output.items(), key=lambda item: item[0]))


def _invert_se3(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -(rotation.T @ translation)
    return inverted


if __name__ == "__main__":
    raise SystemExit(main())
