from __future__ import annotations

import argparse
import csv
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

from src.calibration.hand_eye import calibrate_hand_eye_multi_method  # noqa: E402

PairMode = Literal["adjacent", "all"]


@dataclass(frozen=True, slots=True)
class SampleEntry:
    sample_index: int
    robot_pose_base_end: np.ndarray
    board_pose_camera_board: np.ndarray


@dataclass(frozen=True, slots=True)
class LeaveOneOutResult:
    removed_sample: int
    method_name: str
    rotation_rmse_deg: float
    translation_rmse_mm: float
    score: float | None
    error_message: str | None


def main() -> int:
    args = _parse_cli()
    samples = _load_samples(Path(args.samples_csv))
    results = _evaluate_leave_one_out(
        samples=samples,
        pair_mode=args.pair_mode,
        cv_folds=args.cv_folds,
        robot_semantics=args.robot_semantics,
        board_semantics=args.board_semantics,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hand_eye_leave_one_out.txt"
    _write_report(report_path, results, len(samples))
    _print_summary(results)
    logger.success("结果已写入: {}", report_path)
    return 0


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手眼标定 leave-one-out 离群点分析")
    parser.add_argument(
        "--samples-csv",
        type=str,
        default=str(Path("experiments/hand_eye/runs/20260708_111018/samples.csv")),
        help="采样 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("experiments/hand_eye/runs/20260708_111018/leave_one_out")),
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
        help="交叉验证折数；不传则自动策略",
    )
    parser.add_argument(
        "--robot-semantics",
        type=str,
        default="T_ref_end",
        choices=["T_ref_end", "T_end_ref"],
        help="机器人位姿语义",
    )
    parser.add_argument(
        "--board-semantics",
        type=str,
        default="T_board_camera",
        choices=["T_board_camera", "T_camera_board"],
        help="标定板位姿语义",
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
            board_pose = _pose_from_csv(row, prefix="camera_board")
            if board_pose is None:
                continue
            robot_pose = _robot_pose_from_csv(row)
            entries.append(
                SampleEntry(
                    sample_index=int(row["sample_index"]),
                    robot_pose_base_end=robot_pose,
                    board_pose_camera_board=board_pose,
                )
            )
    if len(entries) < 4:
        raise ValueError(f"有效样本太少，当前只有 {len(entries)} 个")
    return entries


def _pose_from_csv(row: dict[str, str], prefix: str) -> np.ndarray | None:
    x_key = f"{prefix}_x_mm"
    if row.get(x_key, "") == "":
        return None
    translation = np.array(
        [float(row[f"{prefix}_x_mm"]), float(row[f"{prefix}_y_mm"]), float(row[f"{prefix}_z_mm"])],
        dtype=np.float64,
    )
    quat_wxyz = np.array(
        [
            float(row[f"{prefix}_qw"]),
            float(row[f"{prefix}_qx"]),
            float(row[f"{prefix}_qy"]),
            float(row[f"{prefix}_qz"]),
        ],
        dtype=np.float64,
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
    rotation = Rotation3D.from_euler("XYZ", rpy_deg, degrees=True).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _evaluate_leave_one_out(
    samples: list[SampleEntry],
    pair_mode: PairMode,
    cv_folds: int | None,
    robot_semantics: str,
    board_semantics: str,
) -> list[LeaveOneOutResult]:
    robot_seq = [sample.robot_pose_base_end for sample in samples]
    board_seq = [sample.board_pose_camera_board for sample in samples]
    if robot_semantics == "T_end_ref":
        robot_seq = [_invert_se3(pose) for pose in robot_seq]
    if board_semantics == "T_camera_board":
        board_seq = [_invert_se3(pose) for pose in board_seq]

    results: list[LeaveOneOutResult] = []
    for leave_out_index in range(len(samples)):
        keep_indices = [index for index in range(len(samples)) if index != leave_out_index]
        sub_robot = [robot_seq[index] for index in keep_indices]
        sub_board = [board_seq[index] for index in keep_indices]
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
            results.append(
                LeaveOneOutResult(
                    removed_sample=samples[leave_out_index].sample_index,
                    method_name=best.method_name,
                    rotation_rmse_deg=best.residual.rotation_rmse_deg,
                    translation_rmse_mm=best.residual.translation_rmse,
                    score=best.score,
                    error_message=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                LeaveOneOutResult(
                    removed_sample=samples[leave_out_index].sample_index,
                    method_name="failed",
                    rotation_rmse_deg=float("inf"),
                    translation_rmse_mm=float("inf"),
                    score=float("inf"),
                    error_message=str(exc),
                )
            )
    return sorted(results, key=lambda item: float(item.score or float("inf")))


def _write_report(report_path: Path, results: list[LeaveOneOutResult], sample_count: int) -> None:
    lines = [
        "hand_eye leave-one-out analysis",
        f"sample_count={sample_count}",
        "",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"[{index}] removed_sample={result.removed_sample} method={result.method_name}",
                f"rotation_rmse_deg={result.rotation_rmse_deg:.6f}",
                f"translation_rmse_mm={result.translation_rmse_mm:.6f}",
                f"score={_fmt_optional(result.score)}",
                f"error_message={result.error_message or 'None'}",
            ]
        )
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(results: list[LeaveOneOutResult]) -> None:
    logger.info("leave-one-out 结果按 score 排序:")
    for result in results:
        logger.info(
            "removed_sample={} method={} rot_rmse={}deg trans_rmse={}mm score={}",
            result.removed_sample,
            result.method_name,
            _fmt_optional(result.rotation_rmse_deg),
            _fmt_optional(result.translation_rmse_mm),
            _fmt_optional(result.score),
        )


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    if not np.isfinite(value):
        return "inf"
    return f"{float(value):.6f}"


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
