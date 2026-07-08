from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ============================================================
# 基础矩阵工具
# ============================================================

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    构造 4x4 齐次变换矩阵。

    T = [R t]
        [0 1]

    R: 3x3 旋转矩阵
    t: 3维平移向量
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """
    求 4x4 齐次变换矩阵的逆。
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def quat_wxyz_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    将 qw, qx, qy, qz 四元数转换成旋转矩阵。

    scipy 的 Rotation.from_quat() 输入顺序是：
        qx, qy, qz, qw
    """
    q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q_xyzw)

    if norm < 1e-12:
        raise ValueError("四元数模长接近 0，数据异常。")

    q_xyzw = q_xyzw / norm
    return Rotation.from_quat(q_xyzw).as_matrix()


def rpy_xyz_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    将 robot_tcp_roll_deg / pitch_deg / yaw_deg 转为旋转矩阵。

    这里使用你的采集程序中的默认约定：

        DEFAULT_SDK_RPY_SEQUENCE = "xyz"

    即：

        Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)

    注意：
        这里是本次修复的关键点。
        robot_tcp 不再直接使用 CSV 里的 robot_tcp_qw/qx/qy/qz。
    """
    return Rotation.from_euler(
        "xyz",
        [float(roll_deg), float(pitch_deg), float(yaw_deg)],
        degrees=True,
    ).as_matrix()


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    计算旋转矩阵对应的旋转角度，单位 degree。
    """
    rotvec = Rotation.from_matrix(R).as_rotvec()
    return float(np.linalg.norm(rotvec) * 180.0 / np.pi)


def save_matrix_txt(path: Path, name: str, T: np.ndarray) -> None:
    """
    保存矩阵到 txt 文件。
    """
    with path.open("w", encoding="utf-8") as f:
        f.write(name + "\n")
        np.savetxt(f, T, fmt="%.10f")


# ============================================================
# 读取 samples.csv 中的位姿
# ============================================================

def read_robot_tcp_transform_from_row(row, unit_scale: float = 0.001) -> np.ndarray:
    """
    读取机器人 TCP 位姿。

    语义：

        T_base_tool = ^base T_tool

    也就是机械臂末端 / TCP / tool 坐标系到机器人基座坐标系的变换。

    关键修复：
        不使用 robot_tcp_qw/qx/qy/qz。
        而是使用 robot_tcp_roll_deg / pitch_deg / yaw_deg 重新构造旋转矩阵。

    原因：
        当前 samples.csv 中 robot_tcp_qw/qx/qy/qz 与 RPY 重建结果不一致，
        直接使用四元数会导致 10 cm 到 20 cm 级别误差。
    """
    t_base_tool = np.array(
        [
            float(row["robot_tcp_x_mm"]),
            float(row["robot_tcp_y_mm"]),
            float(row["robot_tcp_z_mm"]),
        ],
        dtype=np.float64,
    ) * unit_scale

    R_base_tool = rpy_xyz_deg_to_R(
        roll_deg=float(row["robot_tcp_roll_deg"]),
        pitch_deg=float(row["robot_tcp_pitch_deg"]),
        yaw_deg=float(row["robot_tcp_yaw_deg"]),
    )

    return make_T(R_base_tool, t_base_tool)


def read_camera_board_transform_from_row(row, unit_scale: float = 0.001) -> np.ndarray:
    """
    读取相机观测到的标定板位姿。

    语义：

        T_cam_board = ^cam T_board

    也就是标定板坐标系到相机坐标系的变换。

    这里继续使用 camera_board_qw/qx/qy/qz。
    """
    t_cam_board = np.array(
        [
            float(row["camera_board_x_mm"]),
            float(row["camera_board_y_mm"]),
            float(row["camera_board_z_mm"]),
        ],
        dtype=np.float64,
    ) * unit_scale

    R_cam_board = quat_wxyz_to_R(
        qw=float(row["camera_board_qw"]),
        qx=float(row["camera_board_qx"]),
        qy=float(row["camera_board_qy"]),
        qz=float(row["camera_board_qz"]),
    )

    return make_T(R_cam_board, t_cam_board)


def read_board_camera_transform_from_row(row, unit_scale: float = 0.001) -> np.ndarray:
    """
    读取标定板坐标系下的相机位姿。

    语义：

        T_board_cam = ^board T_cam

    如果 samples.csv 中没有 board_camera_*，主流程会自动由：

        T_board_cam = inv(T_cam_board)

    生成。
    """
    t_board_cam = np.array(
        [
            float(row["board_camera_x_mm"]),
            float(row["board_camera_y_mm"]),
            float(row["board_camera_z_mm"]),
        ],
        dtype=np.float64,
    ) * unit_scale

    R_board_cam = quat_wxyz_to_R(
        qw=float(row["board_camera_qw"]),
        qx=float(row["board_camera_qx"]),
        qy=float(row["board_camera_qy"]),
        qz=float(row["board_camera_qz"]),
    )

    return make_T(R_board_cam, t_board_cam)


def load_samples(
    csv_path: Path,
    unit_scale: float = 0.001,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    读取 samples.csv。

    返回：
        T_base_tool_list:
            robot_tcp_*，语义为 ^base T_tool

        T_cam_board_list:
            camera_board_*，语义为 ^cam T_board

        T_board_cam_list:
            board_camera_*，语义为 ^board T_cam
            如果 CSV 中没有 board_camera_*，则自动由 inv(T_cam_board) 得到。

        used_indices:
            有效样本编号
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "sample_index",
        "board_visible",

        "robot_tcp_x_mm",
        "robot_tcp_y_mm",
        "robot_tcp_z_mm",
        "robot_tcp_roll_deg",
        "robot_tcp_pitch_deg",
        "robot_tcp_yaw_deg",

        "camera_board_x_mm",
        "camera_board_y_mm",
        "camera_board_z_mm",
        "camera_board_qw",
        "camera_board_qx",
        "camera_board_qy",
        "camera_board_qz",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"samples.csv 缺少必要列: {col}\n"
                f"当前 CSV 实际列名为:\n{list(df.columns)}"
            )

    board_camera_columns = [
        "board_camera_x_mm",
        "board_camera_y_mm",
        "board_camera_z_mm",
        "board_camera_qw",
        "board_camera_qx",
        "board_camera_qy",
        "board_camera_qz",
    ]

    has_board_camera = all(col in df.columns for col in board_camera_columns)

    if has_board_camera:
        print("[信息] samples.csv 中存在 board_camera_* 列，将直接读取。")
    else:
        print("[信息] samples.csv 中不存在 board_camera_* 列，将由 camera_board_* 自动取逆得到。")

    T_base_tool_list: List[np.ndarray] = []
    T_cam_board_list: List[np.ndarray] = []
    T_board_cam_list: List[np.ndarray] = []
    used_indices: List[int] = []

    for _, row in df.iterrows():
        sample_index = int(row["sample_index"])

        if int(row["board_visible"]) != 1:
            print(f"[跳过] sample_{sample_index:03d}: board_visible != 1")
            continue

        T_base_tool = read_robot_tcp_transform_from_row(
            row=row,
            unit_scale=unit_scale,
        )

        T_cam_board = read_camera_board_transform_from_row(
            row=row,
            unit_scale=unit_scale,
        )

        if has_board_camera:
            T_board_cam = read_board_camera_transform_from_row(
                row=row,
                unit_scale=unit_scale,
            )

            # 一致性检查：
            # board_camera 应该等于 inv(camera_board)
            T_board_cam_from_inv = invert_T(T_cam_board)

            diff_t = np.linalg.norm(
                T_board_cam[:3, 3] - T_board_cam_from_inv[:3, 3]
            )

            diff_R = T_board_cam[:3, :3].T @ T_board_cam_from_inv[:3, :3]
            diff_angle = rotation_angle_deg(diff_R)

            if diff_t > 1e-4 or diff_angle > 1e-3:
                print(
                    f"[警告] sample_{sample_index:03d}: "
                    f"board_camera 与 inv(camera_board) 不完全一致，"
                    f"dt={diff_t:.6e} m, dR={diff_angle:.6e} deg"
                )
        else:
            T_board_cam = invert_T(T_cam_board)

        T_base_tool_list.append(T_base_tool)
        T_cam_board_list.append(T_cam_board)
        T_board_cam_list.append(T_board_cam)
        used_indices.append(sample_index)

        print(f"[有效] sample_{sample_index:03d}")

    if len(used_indices) < 6:
        raise RuntimeError(f"有效样本太少: {len(used_indices)}")

    return T_base_tool_list, T_cam_board_list, T_board_cam_list, used_indices


# ============================================================
# OpenCV Eye-in-Hand 手眼标定
# ============================================================

def solve_opencv_handeye(
    T_base_tool_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    method_name: str,
) -> np.ndarray:
    """
    使用 OpenCV calibrateHandEye 求解 eye-in-hand 手眼标定。

    输入：
        T_base_tool_list:
            ^base T_tool

        T_cam_board_list:
            ^cam T_board

    OpenCV 参数对应：
        R_gripper2base:
            ^base R_tool

        t_gripper2base:
            ^base t_tool

        R_target2cam:
            ^cam R_board

        t_target2cam:
            ^cam t_board

    输出：
        T_tool_cam:
            ^tool T_cam

    即：
        相机坐标系到机械臂末端 / TCP / tool 坐标系的齐次变换。
    """
    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    if method_name not in method_map:
        raise ValueError(f"未知手眼标定方法: {method_name}")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for T_base_tool, T_cam_board in zip(T_base_tool_list, T_cam_board_list):
        R_gripper2base.append(T_base_tool[:3, :3])
        t_gripper2base.append(T_base_tool[:3, 3].reshape(3, 1))

        R_target2cam.append(T_cam_board[:3, :3])
        t_target2cam.append(T_cam_board[:3, 3].reshape(3, 1))

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method_map[method_name],
    )

    T_tool_cam = make_T(R_cam2gripper, t_cam2gripper)

    return T_tool_cam


# ============================================================
# 验证结果
# ============================================================

def compute_pose_stability(T_list: List[np.ndarray]) -> Dict[str, object]:
    """
    计算一组位姿的稳定性。

    对手眼标定验证来说：

        T_base_board_i
        =
        T_base_tool_i @ T_tool_cam @ T_cam_board_i

    如果标定板固定，则所有 T_base_board_i 应该基本一致。
    """
    translations = np.array([T[:3, 3] for T in T_list], dtype=np.float64)

    t_mean_vec = translations.mean(axis=0)
    t_error_vec = translations - t_mean_vec
    t_errors = np.linalg.norm(t_error_vec, axis=1)

    R_ref = T_list[0][:3, :3]
    r_errors = []

    for T in T_list:
        R_i = T[:3, :3]
        R_err = R_ref.T @ R_i
        r_errors.append(rotation_angle_deg(R_err))

    r_errors = np.array(r_errors, dtype=np.float64)

    return {
        "t_mean_vec": t_mean_vec,
        "t_errors": t_errors,
        "r_errors": r_errors,
        "t_mean": float(np.mean(t_errors)),
        "t_std": float(np.std(t_errors)),
        "t_max": float(np.max(t_errors)),
        "r_mean": float(np.mean(r_errors)),
        "r_std": float(np.std(r_errors)),
        "r_max": float(np.max(r_errors)),
    }


def validate_with_cam_board(
    T_base_tool_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    T_tool_cam: np.ndarray,
) -> Dict[str, object]:
    """
    使用固定标定板约束验证手眼结果。

    对每一帧计算：

        T_base_board_i
        =
        T_base_tool_i @ T_tool_cam @ T_cam_board_i

    如果标定板固定不动，则所有 T_base_board_i 应该基本一致。
    """
    T_base_board_list = []

    for T_base_tool, T_cam_board in zip(T_base_tool_list, T_cam_board_list):
        T_base_board = T_base_tool @ T_tool_cam @ T_cam_board
        T_base_board_list.append(T_base_board)

    return compute_pose_stability(T_base_board_list)


def print_validation(name: str, stats: Dict[str, object]) -> None:
    print(f"\n================ {name} 验证 ================")

    print("标定板在 base 下的位置均值，单位 m:")
    print(stats["t_mean_vec"])

    print("\n每帧标定板位置误差，单位 m:")
    print(stats["t_errors"])

    print("\n平移误差统计:")
    print(f"mean = {stats['t_mean']:.6f} m")
    print(f"std  = {stats['t_std']:.6f} m")
    print(f"max  = {stats['t_max']:.6f} m")

    print("\n每帧标定板旋转误差，相对于第 1 帧，单位 deg:")
    print(stats["r_errors"])

    print("\n旋转误差统计:")
    print(f"mean = {stats['r_mean']:.6f} deg")
    print(f"std  = {stats['r_std']:.6f} deg")
    print(f"max  = {stats['r_max']:.6f} deg")


# ============================================================
# 多方法求解与排序
# ============================================================

def run_all_methods(
    T_base_tool_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    T_board_cam_list: List[np.ndarray],
) -> Dict[str, object]:
    """
    运行多种手眼标定方法，并按验证误差排序。

    实际上 T_board_cam_list 只是保留用于兼容和对照。
    最终 OpenCV 需要的是：

        T_cam_board = ^cam T_board

    如果只有 T_board_cam，可以先取逆得到 T_cam_board。
    """
    candidates: List[Dict[str, object]] = []

    methods = [
        "TSAI",
        "PARK",
        "HORAUD",
        "ANDREFF",
        "DANIILIDIS",
    ]

    print("\n================ OpenCV 标准输入：camera_board = ^cam T_board ================")

    for method in methods:
        try:
            T_tool_cam = solve_opencv_handeye(
                T_base_tool_list=T_base_tool_list,
                T_cam_board_list=T_cam_board_list,
                method_name=method,
            )

            stats = validate_with_cam_board(
                T_base_tool_list=T_base_tool_list,
                T_cam_board_list=T_cam_board_list,
                T_tool_cam=T_tool_cam,
            )

            item = {
                "source": "camera_board",
                "method": method,
                "T_tool_cam": T_tool_cam,
                "T_cam_tool": invert_T(T_tool_cam),
                **stats,
            }

            candidates.append(item)

            print(f"\n方法: {method}")
            print("T_tool_cam = ^tool T_cam:")
            print(T_tool_cam)
            print(
                f"t_mean={stats['t_mean']:.6f} m, "
                f"t_max={stats['t_max']:.6f} m, "
                f"r_mean={stats['r_mean']:.6f} deg, "
                f"r_max={stats['r_max']:.6f} deg"
            )

        except Exception as e:
            print(f"\n方法 {method} 失败: {e}")

    print("\n================ 使用 board_camera，先取逆为 camera_board ================")

    T_cam_board_from_board_camera = [invert_T(T) for T in T_board_cam_list]

    for method in methods:
        try:
            T_tool_cam = solve_opencv_handeye(
                T_base_tool_list=T_base_tool_list,
                T_cam_board_list=T_cam_board_from_board_camera,
                method_name=method,
            )

            stats = validate_with_cam_board(
                T_base_tool_list=T_base_tool_list,
                T_cam_board_list=T_cam_board_from_board_camera,
                T_tool_cam=T_tool_cam,
            )

            item = {
                "source": "board_camera_inverse",
                "method": method,
                "T_tool_cam": T_tool_cam,
                "T_cam_tool": invert_T(T_tool_cam),
                **stats,
            }

            candidates.append(item)

            print(f"\n方法: {method}")
            print("T_tool_cam = ^tool T_cam:")
            print(T_tool_cam)
            print(
                f"t_mean={stats['t_mean']:.6f} m, "
                f"t_max={stats['t_max']:.6f} m, "
                f"r_mean={stats['r_mean']:.6f} deg, "
                f"r_max={stats['r_max']:.6f} deg"
            )

        except Exception as e:
            print(f"\n方法 {method} 失败: {e}")

    if len(candidates) == 0:
        raise RuntimeError("所有手眼标定方法均失败。")

    # 推荐排序逻辑：
    # 1. 优先平移误差小
    # 2. 再看旋转误差
    candidates = sorted(
        candidates,
        key=lambda x: (float(x["t_mean"]), float(x["r_mean"])),
    )

    print("\n================ 排序后的候选结果 ================")

    for i, item in enumerate(candidates):
        print(f"\n候选 {i + 1}")
        print(f"source = {item['source']}")
        print(f"method = {item['method']}")
        print(f"t_mean = {item['t_mean']:.6f} m")
        print(f"t_max  = {item['t_max']:.6f} m")
        print(f"r_mean = {item['r_mean']:.6f} deg")
        print(f"r_max  = {item['r_max']:.6f} deg")
        print("T_tool_cam = ^tool T_cam:")
        print(item["T_tool_cam"])

    # PARK 通常比较稳定；如果 PARK 和最佳误差接近，优先选 PARK。
    best = candidates[0]

    park_candidates = [
        item for item in candidates
        if item["method"] == "PARK"
    ]

    if park_candidates:
        best_park = park_candidates[0]

        # 如果 PARK 与最优结果差异很小，优先采用 PARK。
        if (
            best_park["t_mean"] <= best["t_mean"] + 0.001
            and best_park["r_mean"] <= best["r_mean"] + 0.5
        ):
            best = best_park

    return best


# ============================================================
# 保存结果
# ============================================================

def save_result(
    output_dir: Path,
    best: Dict[str, object],
    used_indices: List[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    T_tool_cam = np.asarray(best["T_tool_cam"], dtype=np.float64)
    T_cam_tool = np.asarray(best["T_cam_tool"], dtype=np.float64)

    np.savez(
        output_dir / "handeye_result_best.npz",
        T_tool_cam=T_tool_cam,
        T_ee_cam=T_tool_cam,
        T_cam_tool=T_cam_tool,
        T_cam_ee=T_cam_tool,
        source=str(best["source"]),
        method=str(best["method"]),
        t_mean_m=float(best["t_mean"]),
        t_std_m=float(best["t_std"]),
        t_max_m=float(best["t_max"]),
        r_mean_deg=float(best["r_mean"]),
        r_std_deg=float(best["r_std"]),
        r_max_deg=float(best["r_max"]),
        used_indices=np.array(used_indices, dtype=np.int32),
    )

    save_matrix_txt(
        output_dir / "T_tool_cam_best.txt",
        "T_tool_cam = ^tool T_cam",
        T_tool_cam,
    )

    save_matrix_txt(
        output_dir / "T_ee_cam_best.txt",
        "T_ee_cam = ^ee T_cam",
        T_tool_cam,
    )

    save_matrix_txt(
        output_dir / "T_cam_tool_best.txt",
        "T_cam_tool = ^cam T_tool",
        T_cam_tool,
    )

    save_matrix_txt(
        output_dir / "T_cam_ee_best.txt",
        "T_cam_ee = ^cam T_ee",
        T_cam_tool,
    )

    with (output_dir / "diagnosis_best.txt").open("w", encoding="utf-8") as f:
        f.write("================ best hand-eye calibration result ================\n")
        f.write(f"source = {best['source']}\n")
        f.write(f"method = {best['method']}\n")
        f.write(f"t_mean_m = {best['t_mean']:.10f}\n")
        f.write(f"t_std_m = {best['t_std']:.10f}\n")
        f.write(f"t_max_m = {best['t_max']:.10f}\n")
        f.write(f"r_mean_deg = {best['r_mean']:.10f}\n")
        f.write(f"r_std_deg = {best['r_std']:.10f}\n")
        f.write(f"r_max_deg = {best['r_max']:.10f}\n")

        f.write("\nT_tool_cam = ^tool T_cam\n")
        np.savetxt(f, T_tool_cam, fmt="%.10f")

        f.write("\nT_cam_tool = ^cam T_tool\n")
        np.savetxt(f, T_cam_tool, fmt="%.10f")

        f.write("\nused samples:\n")
        for idx in used_indices:
            f.write(f"sample_{idx:03d}\n")

    print("\n================ 保存完成 ================")
    print(f"npz:          {output_dir / 'handeye_result_best.npz'}")
    print(f"T_tool_cam:   {output_dir / 'T_tool_cam_best.txt'}")
    print(f"T_ee_cam:     {output_dir / 'T_ee_cam_best.txt'}")
    print(f"T_cam_tool:   {output_dir / 'T_cam_tool_best.txt'}")
    print(f"diagnosis:    {output_dir / 'diagnosis_best.txt'}")

def save_board_pose_in_base(
    output_dir: Path,
    T_base_tool_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    T_tool_cam: np.ndarray,
    used_indices: List[int],
) -> None:
    """
    计算并保存每一帧标定板在机械臂 base 坐标系下的位姿。

    公式：

        T_base_board_i
        =
        T_base_tool_i @ T_tool_cam @ T_cam_board_i

    其中：
        T_base_tool_i:
            ^base T_tool

        T_tool_cam:
            ^tool T_cam

        T_cam_board_i:
            ^cam T_board

    输出：
        board_in_base.csv
        board_in_base_mean.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    T_base_board_list = []

    print("\n================ Board 在机械臂 Base 下的位姿 ================")

    for sample_index, T_base_tool, T_cam_board in zip(
        used_indices,
        T_base_tool_list,
        T_cam_board_list,
    ):
        T_base_board = T_base_tool @ T_tool_cam @ T_cam_board
        T_base_board_list.append(T_base_board)

        R_base_board = T_base_board[:3, :3]
        t_base_board = T_base_board[:3, 3]

        quat_xyzw = Rotation.from_matrix(R_base_board).as_quat()
        qx, qy, qz, qw = quat_xyzw

        rpy_xyz_deg = Rotation.from_matrix(R_base_board).as_euler(
            "xyz",
            degrees=True,
        )

        x_mm = t_base_board[0] * 1000.0
        y_mm = t_base_board[1] * 1000.0
        z_mm = t_base_board[2] * 1000.0

        print(f"\nsample_{sample_index:03d}")
        print("T_base_board = ^base T_board:")
        print(T_base_board)
        print(
            f"board position in base: "
            f"x={x_mm:.3f} mm, "
            f"y={y_mm:.3f} mm, "
            f"z={z_mm:.3f} mm"
        )
        print(
            f"board rpy xyz in base: "
            f"roll={rpy_xyz_deg[0]:.3f} deg, "
            f"pitch={rpy_xyz_deg[1]:.3f} deg, "
            f"yaw={rpy_xyz_deg[2]:.3f} deg"
        )

        records.append(
            {
                "sample_index": sample_index,

                "base_board_x_m": t_base_board[0],
                "base_board_y_m": t_base_board[1],
                "base_board_z_m": t_base_board[2],

                "base_board_x_mm": x_mm,
                "base_board_y_mm": y_mm,
                "base_board_z_mm": z_mm,

                "base_board_qw": qw,
                "base_board_qx": qx,
                "base_board_qy": qy,
                "base_board_qz": qz,

                "base_board_roll_deg": rpy_xyz_deg[0],
                "base_board_pitch_deg": rpy_xyz_deg[1],
                "base_board_yaw_deg": rpy_xyz_deg[2],

                "T00": T_base_board[0, 0],
                "T01": T_base_board[0, 1],
                "T02": T_base_board[0, 2],
                "T03": T_base_board[0, 3],
                "T10": T_base_board[1, 0],
                "T11": T_base_board[1, 1],
                "T12": T_base_board[1, 2],
                "T13": T_base_board[1, 3],
                "T20": T_base_board[2, 0],
                "T21": T_base_board[2, 1],
                "T22": T_base_board[2, 2],
                "T23": T_base_board[2, 3],
                "T30": T_base_board[3, 0],
                "T31": T_base_board[3, 1],
                "T32": T_base_board[3, 2],
                "T33": T_base_board[3, 3],
            }
        )

    df = pd.DataFrame(records)
    csv_path = output_dir / "board_in_base.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 计算 board 在 base 下的平均位置
    # ------------------------------------------------------------

    translations = np.array(
        [T[:3, 3] for T in T_base_board_list],
        dtype=np.float64,
    )

    t_mean = translations.mean(axis=0)
    t_std = translations.std(axis=0)

    print("\n================ Board 在 Base 下的位置统计 ================")
    print(
        f"mean position: "
        f"x={t_mean[0] * 1000.0:.3f} mm, "
        f"y={t_mean[1] * 1000.0:.3f} mm, "
        f"z={t_mean[2] * 1000.0:.3f} mm"
    )
    print(
        f"std position: "
        f"x={t_std[0] * 1000.0:.3f} mm, "
        f"y={t_std[1] * 1000.0:.3f} mm, "
        f"z={t_std[2] * 1000.0:.3f} mm"
    )

    mean_txt_path = output_dir / "board_in_base_mean.txt"

    with mean_txt_path.open("w", encoding="utf-8") as f:
        f.write("Board pose in robot base frame\n")
        f.write("Formula:\n")
        f.write("T_base_board = T_base_tool @ T_tool_cam @ T_cam_board\n\n")

        f.write("Mean board position in base:\n")
        f.write(f"x_m = {t_mean[0]:.10f}\n")
        f.write(f"y_m = {t_mean[1]:.10f}\n")
        f.write(f"z_m = {t_mean[2]:.10f}\n\n")

        f.write("Mean board position in base, unit mm:\n")
        f.write(f"x_mm = {t_mean[0] * 1000.0:.6f}\n")
        f.write(f"y_mm = {t_mean[1] * 1000.0:.6f}\n")
        f.write(f"z_mm = {t_mean[2] * 1000.0:.6f}\n\n")

        f.write("Std board position in base, unit mm:\n")
        f.write(f"x_std_mm = {t_std[0] * 1000.0:.6f}\n")
        f.write(f"y_std_mm = {t_std[1] * 1000.0:.6f}\n")
        f.write(f"z_std_mm = {t_std[2] * 1000.0:.6f}\n\n")

        f.write("Per-sample results are saved in board_in_base.csv\n")

    print("\nBoard 在 Base 下的结果已保存：")
    print(f"CSV:  {csv_path}")
    print(f"Mean: {mean_txt_path}")

# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/handeyeCali",
        help="数据目录，默认 ./data/handeyeCali",
    )

    parser.add_argument(
        "--csv_name",
        type=str,
        default="samples.csv",
        help="CSV 文件名，默认 samples.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/handeyeCali/output",
        help="输出目录，默认 ./data/handeyeCali/output",
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="mm",
        choices=["mm", "m"],
        help="CSV 中平移单位，默认 mm",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / args.csv_name
    output_dir = Path(args.output_dir)

    if args.unit == "mm":
        unit_scale = 0.001
    elif args.unit == "m":
        unit_scale = 1.0
    else:
        raise ValueError("unit 只能是 mm 或 m")

    print("================ 手眼标定开始 ================")
    print(f"data_dir   = {data_dir}")
    print(f"csv_path   = {csv_path}")
    print(f"output_dir = {output_dir}")
    print(f"unit       = {args.unit}")
    print(f"unit_scale = {unit_scale}")

    T_base_tool_list, T_cam_board_list, T_board_cam_list, used_indices = load_samples(
        csv_path=csv_path,
        unit_scale=unit_scale,
    )

    print("\n有效样本数量:", len(used_indices))
    print("有效样本编号:", used_indices)

    best = run_all_methods(
        T_base_tool_list=T_base_tool_list,
        T_cam_board_list=T_cam_board_list,
        T_board_cam_list=T_board_cam_list,
    )

    T_tool_cam = np.asarray(best["T_tool_cam"], dtype=np.float64)
    T_cam_tool = np.asarray(best["T_cam_tool"], dtype=np.float64)

    print("\n================ 最终采用结果 ================")
    print(f"source = {best['source']}")
    print(f"method = {best['method']}")
    print(f"t_mean = {best['t_mean']:.6f} m")
    print(f"t_max  = {best['t_max']:.6f} m")
    print(f"r_mean = {best['r_mean']:.6f} deg")
    print(f"r_max  = {best['r_max']:.6f} deg")

    print("\nT_tool_cam = ^tool T_cam")
    print("含义：相机坐标系到机械臂末端 / TCP / tool 坐标系的齐次变换")
    print(T_tool_cam)

    print("\nT_cam_tool = ^cam T_tool")
    print("含义：机械臂末端 / TCP / tool 坐标系到相机坐标系的齐次变换")
    print(T_cam_tool)

    print_validation("最终结果", best)

    save_result(
        output_dir=output_dir,
        best=best,
        used_indices=used_indices,
    )

    save_board_pose_in_base(
        output_dir=output_dir,
        T_base_tool_list=T_base_tool_list,
        T_cam_board_list=T_cam_board_list,
        T_tool_cam=T_tool_cam,
        used_indices=used_indices,
    )

    print("\n================ 使用方式 ================")
    print("最终使用：")
    print("    T_tool_cam = ^tool T_cam")
    print("也就是：")
    print("    T_ee_cam = ^ee T_cam")
    print("\n如果相机坐标系下有点 p_cam，则：")
    print("    p_tool = T_tool_cam @ p_cam")
    print("\n如果机器人当前末端位姿为 T_base_tool，则：")
    print("    T_base_cam = T_base_tool @ T_tool_cam")
    print("    p_base = T_base_cam @ p_cam")

    if float(best["t_mean"]) > 0.01 or float(best["r_mean"]) > 2.0:
        print("\n[警告]")
        print("当前最佳结果误差仍然偏大。")
        print("建议继续检查：")
        print("1. robot_tcp_roll_deg / pitch_deg / yaw_deg 的 Euler 顺序是否确实为 xyz。")
        print("2. cartPosture(endInRef) 是否为 ^base T_tool。")
        print("3. 图像帧与机器人位姿是否严格同步。")
        print("4. 机械臂采样时是否完全静止。")
        print("5. 标定板是否在采集过程中完全固定。")
    else:
        print("\n[结果判断]")
        print("当前验证误差处于可接受范围，可以用于后续相机点到机器人基座的坐标转换。")


if __name__ == "__main__":
    main()