from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rotation3D

RUN_DIR = Path("experiments/hand_eye/runs/20260708_152829")
SAMPLES_CSV = RUN_DIR / "samples.csv"
HAND_EYE_RESULT = RUN_DIR / "hand_eye_result.txt"
BOARD_IN_BASE_CSV = RUN_DIR / "board_in_base.csv"


@dataclass(frozen=True, slots=True)
class PoseRecord:
    sample_index: int
    robot_end: np.ndarray
    cam_board: np.ndarray
    expected_base_board: np.ndarray


def main() -> int:
    tool_cam = _load_tool_cam(HAND_EYE_RESULT)
    samples = _load_samples(SAMPLES_CSV)
    expected_rows = _load_expected_board_in_base(BOARD_IN_BASE_CSV) if BOARD_IN_BASE_CSV.exists() else {}

    logger.info("run_dir={}", RUN_DIR)
    logger.info("samples={}", len(samples))
    logger.info("tool_cam_path={}", HAND_EYE_RESULT)
    logger.info("board_in_base_csv={}", BOARD_IN_BASE_CSV if BOARD_IN_BASE_CSV.exists() else "NA")
    logger.info("T_tool_cam(m)=\n{}", np.array2string(tool_cam, precision=10, suppress_small=False))

    errors_t: list[float] = []
    errors_r: list[float] = []

    for sample in samples:
        base_board = sample.robot_end @ tool_cam @ sample.cam_board
        expected = expected_rows.get(sample.sample_index)
        if expected is not None:
            delta = np.linalg.inv(expected) @ base_board
            t_err_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
            r_err_deg = float(np.linalg.norm(Rotation3D.from_matrix(delta[:3, :3]).as_rotvec()) * 180.0 / np.pi)
            errors_t.append(t_err_mm)
            errors_r.append(r_err_deg)
            logger.info(
                "sample_{:03d} base_board=({}, {}, {}) rpy=({}, {}, {}) err_t_mm={:.4f} err_r_deg={:.4f}",
                sample.sample_index,
                f"{base_board[0, 3] * 1000.0:.3f}",
                f"{base_board[1, 3] * 1000.0:.3f}",
                f"{base_board[2, 3] * 1000.0:.3f}",
                f"{Rotation3D.from_matrix(base_board[:3, :3]).as_euler('XYZ', degrees=True)[0]:.3f}",
                f"{Rotation3D.from_matrix(base_board[:3, :3]).as_euler('XYZ', degrees=True)[1]:.3f}",
                f"{Rotation3D.from_matrix(base_board[:3, :3]).as_euler('XYZ', degrees=True)[2]:.3f}",
                t_err_mm,
                r_err_deg,
            )
        else:
            logger.info(
                "sample_{:03d} base_board=({}, {}, {}) rpy=({}, {}, {})",
                sample.sample_index,
                f"{base_board[0, 3] * 1000.0:.3f}",
                f"{base_board[1, 3] * 1000.0:.3f}",
                f"{base_board[2, 3] * 1000.0:.3f}",
                f"{Rotation3D.from_matrix(base_board[:3, :3]).as_euler('XYZ', degrees=True)[0]:.3f}",
                f"{Rotation3D.from_matrix(base_board[:3, :3]).as_euler('XYZ', degrees=True)[1]:.3f}",
                f"{Rotation3D.from_matrix(base_board[:3, :3]).as_euler('XYZ', degrees=True)[2]:.3f}",
            )

    if errors_t:
        logger.success("mean_err_t_mm={:.4f} max_err_t_mm={:.4f}", float(np.mean(errors_t)), float(np.max(errors_t)))
        logger.success("mean_err_r_deg={:.4f} max_err_r_deg={:.4f}", float(np.mean(errors_r)), float(np.max(errors_r)))
    return 0


def _load_tool_cam(result_path: Path) -> np.ndarray:
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    lines = result_path.read_text(encoding="utf-8").splitlines()
    matrix_lines: list[str] = []
    capture = False
    for line in lines:
        if line.strip() == "T_tool_cam:":
            capture = True
            continue
        if capture:
            if line.strip().startswith("[[") or matrix_lines:
                matrix_lines.append(line)
                if line.strip().endswith("]]"):
                    break
    if not matrix_lines:
        raise ValueError(f"找不到 T_tool_cam: {result_path}")
    matrix_text = "\n".join(matrix_lines)
    numbers = [float(v) for v in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", matrix_text)]
    if len(numbers) != 16:
        raise ValueError(f"解析到的矩阵元素数量不是 16: {len(numbers)}")
    return np.asarray(numbers, dtype=np.float64).reshape(4, 4)


def _load_samples(csv_path: Path) -> list[PoseRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    samples: list[PoseRecord] = []
    for row in rows:
        if int(row["board_visible"]) != 1:
            continue
        sample_index = int(row["sample_index"])
        robot_end = _robot_end_matrix(row)
        cam_board = _camera_board_matrix(row)
        expected = _expected_base_board_matrix(row)
        samples.append(PoseRecord(sample_index=sample_index, robot_end=robot_end, cam_board=cam_board, expected_base_board=expected))
    return samples


def _load_expected_board_in_base(csv_path: Path) -> dict[int, np.ndarray]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        mapping: dict[int, np.ndarray] = {}
        for row in reader:
            sample_index = int(row["sample_index"])
            mapping[sample_index] = _expected_base_board_matrix(row)
    return mapping


def _robot_end_matrix(row: dict[str, str]) -> np.ndarray:
    rotation = Rotation3D.from_euler(
        "xyz",
        [
            float(row["robot_end_roll_deg"]),
            float(row["robot_end_pitch_deg"]),
            float(row["robot_end_yaw_deg"]),
        ],
        degrees=True,
    ).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.array(
        [
            float(row["robot_end_x_mm"]),
            float(row["robot_end_y_mm"]),
            float(row["robot_end_z_mm"]),
        ],
        dtype=np.float64,
    ) * 0.001
    return matrix


def _camera_board_matrix(row: dict[str, str]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    quat = [
        float(row["camera_board_qx"]),
        float(row["camera_board_qy"]),
        float(row["camera_board_qz"]),
        float(row["camera_board_qw"]),
    ]
    matrix[:3, :3] = Rotation3D.from_quat(quat).as_matrix()
    matrix[:3, 3] = np.array(
        [
            float(row["camera_board_x_mm"]),
            float(row["camera_board_y_mm"]),
            float(row["camera_board_z_mm"]),
        ],
        dtype=np.float64,
    ) * 0.001
    return matrix


def _expected_base_board_matrix(row: dict[str, str]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    quat = [
        float(row["base_board_qx"]),
        float(row["base_board_qy"]),
        float(row["base_board_qz"]),
        float(row["base_board_qw"]),
    ]
    matrix[:3, :3] = Rotation3D.from_quat(quat).as_matrix()
    matrix[:3, 3] = np.array(
        [
            float(row["base_board_x_mm"]),
            float(row["base_board_y_mm"]),
            float(row["base_board_z_mm"]),
        ],
        dtype=np.float64,
    ) * 0.001
    return matrix


if __name__ == "__main__":
    raise SystemExit(main())
