from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from scipy.spatial.transform import Rotation as Rotation3D

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from src.calibration import CHARUCO_200_12_9, CharucoPoseEstimator  # noqa: E402
from test.wuji.charuco_detect import (  # noqa: E402
    DEFAULT_CAMERA_NAME,
    DEFAULT_MIN_CHARUCO_CORNERS,
    DEFAULT_ORIN_SERVICE_ADDR,
    _read_camera_calibration,
    _validate_runtime_requirements,
)
from test.wuji.charuco_pose_offset_interactive import DEFAULT_ARM_IP  # noqa: E402
from test.wuji.xcoresdk_arm_cli_test import _print_sdk_result, _shutdown_robot  # noqa: E402

# region 默认参数
DEFAULT_WINDOW_NAME = "Charuco Pose Drag"
DEFAULT_CAMERA_TIMEOUT_MS = 30_000
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_FONT_SIZE = 20
DEFAULT_REFRESH_SLEEP_S = 0.03
DEFAULT_HAND_EYE_RESULT_PATH = Path("experiments/hand_eye/runs/20260708_152829/hand_eye_result.txt")
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    """单个位姿快照。

    约定：
    - ``pose_matrix`` 保存 4x4 齐次矩阵的原始数值
    - 该矩阵内部统一使用 ``m`` 作为长度单位
    - ``translation_mm`` 仅用于界面展示与日志输出
    - ``rpy_deg`` 仅用于界面展示与日志输出

    这次排查过的关键问题也记录在这里，避免后面再踩一遍：
    - xCoreSDK `cartPosture(...).trans` 的原始单位是 ``m``
    - GUI 里更适合给人看的平移单位通常是 ``mm``
    - 如果先把 `trans` 转成 ``mm``，再写回 `pose_matrix`
      那么后续所有 ``T_base_tool @ T_tool_cam @ T_cam_board`` 计算
      都会把本该是米制的矩阵当成米继续连乘，最终让 `base_board`
      看起来与采样记录明显不一致
    - 因此源头修正规则是：`pose_matrix` 只保存计算链路原始单位，
      当前页内部统一为 ``m``；任何 ``mm/deg`` 都只能在展示层生成
    """

    pose_matrix: np.ndarray
    translation_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    recorded_at_iso: str


@dataclass(slots=True)
class RuntimeState:
    reference_base_board: PoseSnapshot | None = None
    last_action_text: str = "等待有效板位姿，按空格记录基准帧"
# endregion


# region 主流程
def main(
    service_addr: str = DEFAULT_ORIN_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    arm_ip: str = DEFAULT_ARM_IP,
    min_charuco_corners: int = DEFAULT_MIN_CHARUCO_CORNERS,
) -> int:
    _validate_runtime_requirements()
    tool_cam_m = _load_tool_cam_from_result(DEFAULT_HAND_EYE_RESULT_PATH)
    state = RuntimeState()
    estimator = CharucoPoseEstimator(CHARUCO_200_12_9)

    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(str(arm_ip))
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=DEFAULT_CAMERA_TIMEOUT_MS)
    try:
        calibration_response = client.get_camera_intrinsics(timeout_s=3.0)
        calibration = _read_camera_calibration(calibration_response)

        robot.robotInfo(ec)
        _print_sdk_result("robotInfo", ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"连接机械臂失败: ip={arm_ip}")

        frame_stream = client.subscribe_camera_color_frames(camera_name)
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame_packet = next(frame_stream)
            frame_bgr = np.asarray(frame_packet.color_bgr, dtype=np.uint8).copy()

            charuco_result = estimator.estimate_pose(
                image_bgr=frame_bgr,
                camera_matrix=calibration.camera_matrix,
                dist_coeffs=calibration.dist_coeffs,
                min_charuco_corners=int(min_charuco_corners),
            )
            end_pose_in_ref = _read_end_pose_in_ref(robot, ec)
            board_pose_camera_board = _read_board_pose_camera_board(charuco_result)
            base_board = _compute_base_board(end_pose_in_ref, tool_cam_m, board_pose_camera_board)

            preview_bgr = _draw_preview(
                frame_bgr=frame_bgr,
                charuco_result=charuco_result,
                end_pose_in_ref=end_pose_in_ref,
                board_pose_camera_board=board_pose_camera_board,
                base_board=base_board,
                state=state,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                if end_pose_in_ref is None or board_pose_camera_board is None or base_board is None:
                    state.last_action_text = "空格记录失败：当前没有有效的 TCP 或板位姿"
                else:
                    state.reference_base_board = base_board
                    state.last_action_text = f"已记录基准帧 {datetime.now().isoformat(timespec='seconds')}"
            time.sleep(DEFAULT_REFRESH_SLEEP_S)
        return 0
    finally:
        client.close()
        cv2.destroyAllWindows()
        _shutdown_robot(robot, ec)
# endregion


# region 机器人与检测
def _read_end_pose_in_ref(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> PoseSnapshot | None:
    end_pose_in_ref = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        return None
    return _pose_snapshot_from_sdk_pose(end_pose_in_ref)


def _read_board_pose_camera_board(charuco_result: Any) -> np.ndarray | None:
    if charuco_result is None or not charuco_result.board_visible or charuco_result.transform_se3 is None:
        return None
    return np.asarray(charuco_result.transform_se3, dtype=np.float64).reshape(4, 4)


def _load_tool_cam_from_result(result_path: Path) -> np.ndarray:
    if not result_path.exists():
        raise FileNotFoundError(f"找不到手眼结果文件: {result_path}")
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
        raise ValueError(f"无法在 {result_path} 中找到 T_tool_cam")
    matrix_text = "\n".join(matrix_lines)
    numbers = [float(value) for value in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", matrix_text)]
    if len(numbers) != 16:
        raise ValueError(f"无法从 {result_path} 解析出 16 个矩阵元素，实际解析到 {len(numbers)} 个")
    return np.asarray(numbers, dtype=np.float64).reshape(4, 4)
# endregion


# region 预览绘制
def _draw_preview(
    frame_bgr: np.ndarray,
    charuco_result: Any,
    end_pose_in_ref: PoseSnapshot | None,
    board_pose_camera_board: np.ndarray | None,
    base_board: PoseSnapshot | None,
    state: RuntimeState,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if charuco_result is not None and charuco_result.marker_corners_px:
        cv2.aruco.drawDetectedMarkers(canvas, charuco_result.marker_corners_px, charuco_result.marker_ids, borderColor=(255, 180, 0))
    if charuco_result is not None and charuco_result.rvec is not None and charuco_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            np.eye(3, dtype=np.float64),
            np.zeros((5, 1), dtype=np.float64),
            np.asarray(charuco_result.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(charuco_result.tvec, dtype=np.float64).reshape(3, 1),
            30.0,
            2,
        )

    lines = _build_overlay_lines(
        charuco_result=charuco_result,
        end_pose_in_ref=end_pose_in_ref,
        board_pose_camera_board=board_pose_camera_board,
        base_board=base_board,
        state=state,
    )
    return _draw_text_block(canvas, lines, (18, 24))


def _build_overlay_lines(
    charuco_result: Any,
    end_pose_in_ref: PoseSnapshot | None,
    board_pose_camera_board: np.ndarray | None,
    base_board: PoseSnapshot | None,
    state: RuntimeState,
) -> list[str]:
    marker_count = 0 if charuco_result is None else int(charuco_result.marker_count)
    charuco_count = 0 if charuco_result is None else int(charuco_result.charuco_count)
    reproj = "NA" if charuco_result is None or charuco_result.reprojection_error_px is None else f"{float(charuco_result.reprojection_error_px):.4f}"
    lines = [
        f"board_visible={bool(board_pose_camera_board is not None)} marker={marker_count} charuco={charuco_count} reproj={reproj}",
        f"action={state.last_action_text}",
        "end_pose_in_ref=" + _format_pose_line(end_pose_in_ref),
        (
            "board_pose_camera_board="
            + _format_pose_line(_pose_snapshot_from_matrix_m(_camera_board_matrix_m(board_pose_camera_board)))
            if board_pose_camera_board is not None
            else "board_pose_camera_board=NA"
        ),
        "base_board=" + _format_pose_line(base_board),
        "T_base_cam @ T_cam_board = base_board",
    ]
    if state.reference_base_board is not None and base_board is not None:
        delta = _compute_delta_pose(state.reference_base_board, base_board)
        lines.append("base_board_delta=" + _format_pose_line(delta))
    lines.append("space=record reference   q/esc=quit")
    return lines


def _draw_text_block(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(DEFAULT_FONT_SIZE)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=2)
        y += 26
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
# endregion


# region 位姿工具
def _pose_snapshot_from_sdk_pose(cartesian_pose: xCoreSDK_python.CartesianPosition) -> PoseSnapshot:
    """把 xCoreSDK 原始 CartesianPosition 转成内部统一位姿快照。

    这里必须保留原始数据语义：
    - ``cartesian_pose.trans`` 的原始单位是 ``m``
    - ``cartesian_pose.rpy`` 的原始单位是 ``rad``
    - ``pose_matrix`` 也必须继续保存 ``m`` 语义，不能先转成 ``mm``
    """

    # 问题源头就在这里：如果把 SDK 原始 `trans(m)` 提前改写成 `mm`
    # 再塞进 pose_matrix，那么后面所有矩阵乘法都会发生单位串链错误。
    rotation = Rotation3D.from_euler(
        "xyz",
        np.asarray(cartesian_pose.rpy, dtype=np.float64).reshape(3),
        degrees=False,
    ).as_matrix()
    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = np.asarray(cartesian_pose.trans, dtype=np.float64).reshape(3)
    return _pose_snapshot_from_matrix_m(pose_matrix)


def _pose_snapshot_from_matrix_m(pose_matrix: np.ndarray) -> PoseSnapshot:
    """从内部统一使用 ``m`` 的矩阵生成展示快照。

    这里专门负责把内部矩阵拆成适合人看的 ``mm/deg``。
    也就是说：
    - `pose_matrix` 继续保持计算真值
    - `translation_mm / rpy_deg` 只是展示投影
    """

    matrix = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    rpy_deg = Rotation3D.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True)
    return PoseSnapshot(
        pose_matrix=matrix,
        translation_mm=(float(matrix[0, 3] * 1000.0), float(matrix[1, 3] * 1000.0), float(matrix[2, 3] * 1000.0)),
        rpy_deg=(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])),
        recorded_at_iso=datetime.now().isoformat(timespec="milliseconds"),
    )


def _compute_base_board(
    end_pose_in_ref: PoseSnapshot | None,
    tool_cam_m: np.ndarray,
    board_pose_camera_board: np.ndarray | None,
) -> PoseSnapshot | None:
    if end_pose_in_ref is None or board_pose_camera_board is None:
        return None
    base_board = (
        np.asarray(end_pose_in_ref.pose_matrix, dtype=np.float64).reshape(4, 4)
        @ np.asarray(tool_cam_m, dtype=np.float64).reshape(4, 4)
        @ _camera_board_matrix_m(board_pose_camera_board)
    )
    return _pose_snapshot_from_matrix_m(base_board)


def _compute_delta_pose(reference_pose: PoseSnapshot | None, current_pose: PoseSnapshot | None) -> PoseSnapshot | None:
    if reference_pose is None or current_pose is None:
        return None
    delta = _inverse_transform(reference_pose.pose_matrix) @ current_pose.pose_matrix
    return _pose_snapshot_from_matrix_m(delta)


def _inverse_transform(pose_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def _camera_board_matrix_m(camera_board_mm: np.ndarray | None) -> np.ndarray:
    if camera_board_mm is None:
        raise ValueError("缺少 T_cam_board")
    matrix = np.asarray(camera_board_mm, dtype=np.float64).reshape(4, 4).copy()
    matrix[:3, 3] *= 0.001
    return matrix
# endregion


# region 通用工具
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(DEFAULT_FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def _format_pose_line(pose: PoseSnapshot | None) -> str:
    if pose is None:
        return "NA"
    return (
        f"xyz_mm=({pose.translation_mm[0]:.1f}, {pose.translation_mm[1]:.1f}, {pose.translation_mm[2]:.1f}) "
        f"rpy_deg=({pose.rpy_deg[0]:.1f}, {pose.rpy_deg[1]:.1f}, {pose.rpy_deg[2]:.1f})"
    )


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="charuco pose drag")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_ORIN_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP)
    parser.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS)
    return parser.parse_args(argv)
# endregion


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            arm_ip=str(args.arm_ip),
            min_charuco_corners=int(args.min_charuco_corners),
        )
    )
