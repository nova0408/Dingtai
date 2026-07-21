"""三球全局笛卡尔纠偏的先验与矩阵计算。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt



def validate_offset_matrix(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """校验并复制用于全局纠偏的 4x4 齐次矩阵。"""

    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"全局纠偏矩阵必须是 (4, 4)，实际为 {result.shape}")
    return result.copy()


def load_tool_camera_transform(hand_eye_result_path: Path) -> npt.NDArray[np.float64]:
    """从手眼结果文本读取以 m 为单位的 T_tool_cam。"""

    if not hand_eye_result_path.is_file():
        raise FileNotFoundError(f"手眼结果文件不存在：{hand_eye_result_path}")
    rows: list[list[float]] = []
    collecting = False
    for line in hand_eye_result_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "T_tool_cam:":
            collecting = True
            continue
        if collecting and not line.strip():
            break
        if collecting:
            values = [float(token) for token in line.replace("[", " ").replace("]", " ").split()]
            if len(values) != 4:
                raise ValueError(f"手眼矩阵行格式错误：{line}")
            rows.append(values)
            if len(rows) == 4:
                break
    return validate_offset_matrix(np.asarray(rows, dtype=np.float64))


def load_prior_base_ball_transform(prior_capture_path: Path, hand_eye_result_path: Path) -> npt.NDArray[np.float64]:
    """读取先验 JSON 并重建 T_prior_base_ball，内部单位为 m。"""

    if not prior_capture_path.is_file():
        raise FileNotFoundError(f"先验文件不存在：{prior_capture_path}")
    content = json.loads(prior_capture_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"先验文件根节点必须是 object: {prior_capture_path}")
    tcp_matrix = validate_offset_matrix(content["tcp_pose_matrix"])
    camera_ball_matrix = validate_offset_matrix(content["local_pose_transform"])
    camera_ball_matrix[:3, 3] *= 0.001
    return tcp_matrix @ load_tool_camera_transform(hand_eye_result_path) @ camera_ball_matrix


def calculate_global_offset(
    current_tcp_matrix_m: npt.ArrayLike,
    tool_camera_matrix_m: npt.ArrayLike,
    current_camera_ball_matrix_m: npt.ArrayLike,
    prior_base_ball_matrix_m: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """计算 T_off = T_tcp @ T_tool_cam @ T_cam_ball @ inv(T_prior_base_ball)。"""

    current_base_ball = (
        validate_offset_matrix(current_tcp_matrix_m)
        @ validate_offset_matrix(tool_camera_matrix_m)
        @ validate_offset_matrix(current_camera_ball_matrix_m)
    )
    return current_base_ball @ np.linalg.inv(validate_offset_matrix(prior_base_ball_matrix_m))
