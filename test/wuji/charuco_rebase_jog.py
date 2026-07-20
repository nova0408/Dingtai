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
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rotation3D

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.wuji.charuco_detect import (
    DEFAULT_CAMERA_NAME,
    DEFAULT_MIN_CHARUCO_CORNERS,
    DEFAULT_ORIN_SERVICE_ADDR,
    _read_camera_calibration,
    _validate_runtime_requirements,
)
from test.wuji.charuco_pose_offset_interactive import DEFAULT_ARM_IP
from test.wuji.xcoresdk_arm_cli_test import (
    DEFAULT_TOOL_NAME,
    DEFAULT_WOBJ_NAME,
    _apply_named_toolset,
    _print_sdk_result,
    _shutdown_robot,
)

from camera_pipeline.client import CameraPipelineClient
from sdk.xcoresdk import xCoreSDK_python
from src.calibration import CHARUCO_200_12_9, CharucoPoseEstimator

# T_prior_base_board=T_tcp@T_tool_cam@T_cam_board
# T_new_base_board=T_off@T_prior_base_board =T_tcp@T_tool_cam@T_cam_board
# T_off=T_tcp@T_tool_cam@T_cam_board@inv(T_prior_base_board)
# T_new_tcp=inv(T_off)@T_tcp

# region 默认参数
DEFAULT_WINDOW_NAME = "Charuco Rebase Jog"
DEFAULT_CAMERA_TIMEOUT_MS = 30_000
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_FONT_SIZE = 20
DEFAULT_REFRESH_SLEEP_S = 0.03
DEFAULT_HAND_EYE_RESULT_PATH = Path("experiments/hand_eye/runs/20260708_152829/hand_eye_result.txt")
DEFAULT_TCP_TRANSLATION_TOLERANCE_MM = 0.5
DEFAULT_TCP_ROTATION_TOLERANCE_DEG = 0.2
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    """单个位姿快照。

    `pose_matrix` 内部统一保存 m 单位的 4x4 齐次矩阵。
    `translation_mm` / `rpy_deg` 只用于显示与日志。
    """

    pose_matrix: np.ndarray
    translation_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    recorded_at_iso: str


@dataclass(frozen=True, slots=True)
class PriorKnowledge:
    tool_cam_m: np.ndarray
    calibration_base_board_m: np.ndarray
    source_path: Path


@dataclass(frozen=True, slots=True)
class RebaseSnapshot:
    toolset_text: str
    current_end_pose: PoseSnapshot
    current_cam_board_pose: PoseSnapshot
    current_base_board_pose: PoseSnapshot
    offset_pose: PoseSnapshot
    delta_offset_pose: PoseSnapshot | None
    prior_to_current_base_board_pose: PoseSnapshot


@dataclass(slots=True)
class RuntimeState:
    reference_offset_pose: PoseSnapshot | None = None
    latest_snapshot: RebaseSnapshot | None = None
    last_action_text: str = "等待有效 ChArUco 位姿，启动后已自动进入拖动模式，空格记录 offset 基准"


# endregion


# region 主流程
def main(
    service_addr: str = DEFAULT_ORIN_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    arm_ip: str = DEFAULT_ARM_IP,
    min_charuco_corners: int = DEFAULT_MIN_CHARUCO_CORNERS,
    hand_eye_result_path: Path = DEFAULT_HAND_EYE_RESULT_PATH,
) -> int:
    _validate_runtime_requirements()
    prior = _load_prior_knowledge(hand_eye_result_path)
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
        _ensure_fixed_toolset(robot, ec)
        _enable_drag_mode(robot, ec)

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

            current_end_pose = _read_end_pose_in_ref(robot, ec)
            current_cam_board_m = _read_camera_board_matrix_m(charuco_result)
            if current_end_pose is not None and current_cam_board_m is not None:
                toolset_text = _read_current_toolset_text(robot, ec)
                state.latest_snapshot = _build_rebase_snapshot(
                    prior=prior,
                    toolset_text=toolset_text,
                    current_end_pose=current_end_pose,
                    current_cam_board_m=current_cam_board_m,
                    reference_offset_pose=state.reference_offset_pose,
                )
                state.last_action_text = "拖动中实时验算 offset 是否恒定"
            else:
                state.latest_snapshot = None
                state.last_action_text = "等待有效 ChArUco 位姿"

            preview_bgr = _draw_preview(
                frame_bgr=frame_bgr,
                charuco_result=charuco_result,
                calibration=calibration,
                prior=prior,
                state=state,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                if state.latest_snapshot is None:
                    state.last_action_text = "记录失败：当前没有有效 offset"
                else:
                    state.reference_offset_pose = state.latest_snapshot.offset_pose
                    state.last_action_text = f"已记录 offset 基准 {datetime.now().isoformat(timespec='seconds')}"
            time.sleep(DEFAULT_REFRESH_SLEEP_S)
        return 0
    finally:
        client.close()
        cv2.destroyAllWindows()
        _shutdown_robot(robot, ec)


# endregion


# region 先验加载
def _load_prior_knowledge(result_path: Path) -> PriorKnowledge:
    if not result_path.exists():
        raise FileNotFoundError(f"找不到手眼结果文件: {result_path}")
    lines = result_path.read_text(encoding="utf-8").splitlines()
    tool_cam_m = _parse_matrix_after_header(lines, "T_tool_cam:")
    calibration_base_board_m = _parse_base_board_mean_from_result(lines)
    return PriorKnowledge(
        tool_cam_m=tool_cam_m, calibration_base_board_m=calibration_base_board_m, source_path=result_path
    )


def _parse_matrix_after_header(lines: list[str], header: str) -> np.ndarray:
    matrix_lines: list[str] = []
    capture = False
    for line in lines:
        if line.strip() == header:
            capture = True
            continue
        if capture:
            if line.strip().startswith("[[") or matrix_lines:
                matrix_lines.append(line)
                if line.strip().endswith("]]"):
                    break
    if not matrix_lines:
        raise ValueError(f"无法找到矩阵头: {header}")
    numbers = [
        float(value) for value in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", "\n".join(matrix_lines))
    ]
    if len(numbers) != 16:
        raise ValueError(f"{header} 解析出的矩阵元素数量不是 16，而是 {len(numbers)}")
    return np.asarray(numbers, dtype=np.float64).reshape(4, 4)


def _parse_base_board_mean_from_result(lines: list[str]) -> np.ndarray:
    mean_translation_m = _parse_mean_base_board_translation_m(lines)
    sample_rotations: list[np.ndarray] = []
    collecting_samples = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[per_sample_base_board]":
            collecting_samples = True
            continue
        if not collecting_samples or not stripped.startswith("sample_"):
            continue
        match = re.search(
            r"t_mm=\(([^,]+), ([^,]+), ([^)]+)\)\s+rpy_deg=\(([^,]+), ([^,]+), ([^)]+)\)",
            stripped,
        )
        if match is None:
            continue
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = [float(value) for value in match.groups()]
        sample_rotations.append(Rotation3D.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix())
    if mean_translation_m is None or not sample_rotations:
        raise ValueError("hand_eye_result.txt 中缺少 [per_sample_base_board] 段，无法恢复 T_base_board")
    mean_rotation = _mean_rotation_matrix(sample_rotations)
    base_board_m = np.eye(4, dtype=np.float64)
    base_board_m[:3, :3] = mean_rotation
    base_board_m[:3, 3] = mean_translation_m
    return base_board_m


def _parse_mean_base_board_translation_m(lines: list[str]) -> np.ndarray | None:
    x_value: float | None = None
    y_value: float | None = None
    z_value: float | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("x_m ="):
            x_value = float(stripped.split("=", maxsplit=1)[1].strip())
        elif stripped.startswith("y_m ="):
            y_value = float(stripped.split("=", maxsplit=1)[1].strip())
        elif stripped.startswith("z_m ="):
            z_value = float(stripped.split("=", maxsplit=1)[1].strip())
    if x_value is None or y_value is None or z_value is None:
        return None
    return np.array([x_value, y_value, z_value], dtype=np.float64)


def _mean_rotation_matrix(rotations: list[np.ndarray]) -> np.ndarray:
    quaternions_xyzw = [Rotation3D.from_matrix(rotation).as_quat() for rotation in rotations]
    reference = quaternions_xyzw[0]
    accumulator = np.zeros(4, dtype=np.float64)
    for quaternion in quaternions_xyzw:
        aligned = quaternion.copy()
        if float(np.dot(reference, aligned)) < 0.0:
            aligned *= -1.0
        accumulator += aligned
    accumulator /= np.linalg.norm(accumulator)
    return Rotation3D.from_quat(accumulator).as_matrix()


# endregion


# region 读取当前状态
def _read_end_pose_in_ref(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> PoseSnapshot | None:
    _ensure_fixed_toolset(robot, ec)
    end_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        return None
    return _pose_snapshot_from_sdk_pose(end_pose)


def _read_camera_board_matrix_m(charuco_result: Any) -> np.ndarray | None:
    if charuco_result is None or not charuco_result.board_visible or charuco_result.transform_se3 is None:
        return None
    source = np.asarray(charuco_result.transform_se3, dtype=np.float64).reshape(4, 4)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = source[:3, :3]
    matrix[:3, 3] = source[:3, 3] * 0.001
    return matrix


# endregion


# region 重定位求解
def _build_rebase_snapshot(
    prior: PriorKnowledge,
    toolset_text: str,
    current_end_pose: PoseSnapshot,
    current_cam_board_m: np.ndarray,
    reference_offset_pose: PoseSnapshot | None,
) -> RebaseSnapshot:
    current_tcp_m = np.asarray(current_end_pose.pose_matrix, dtype=np.float64).reshape(4, 4)
    tool_cam_m = np.asarray(prior.tool_cam_m, dtype=np.float64).reshape(4, 4)
    prior_base_board_m = np.asarray(prior.calibration_base_board_m, dtype=np.float64).reshape(4, 4)
    current_base_board_m = _compute_base_board_matrix_m(
        end_pose_in_ref=current_end_pose,
        tool_cam_m=tool_cam_m,
        board_pose_camera_board_m=current_cam_board_m,
    )
    # 用户最终确认的公式：
    #   T_new_base_board = T_off @ T_prior_base_board
    #   T_new_base_board = T_tcp @ T_tool_cam @ T_cam_board
    # 因此：
    #   T_off = T_new_base_board @ inv(T_prior_base_board)
    # 当前验证页只关注：任意拖动位置下，这个 T_off 是否保持稳定。
    offset_m = current_tcp_m @ tool_cam_m @ current_cam_board_m @ np.linalg.inv(prior_base_board_m)
    prior_to_current_base_board_m = np.linalg.inv(prior_base_board_m) @ current_base_board_m
    delta_offset_pose = None
    if reference_offset_pose is not None:
        delta_offset_m = np.linalg.inv(reference_offset_pose.pose_matrix) @ offset_m
        delta_offset_pose = _pose_snapshot_from_matrix_m(delta_offset_m)
    return RebaseSnapshot(
        toolset_text=toolset_text,
        current_end_pose=current_end_pose,
        current_cam_board_pose=_pose_snapshot_from_matrix_m(current_cam_board_m),
        current_base_board_pose=_pose_snapshot_from_matrix_m(current_base_board_m),
        offset_pose=_pose_snapshot_from_matrix_m(offset_m),
        delta_offset_pose=delta_offset_pose,
        prior_to_current_base_board_pose=_pose_snapshot_from_matrix_m(prior_to_current_base_board_m),
    )


def _compute_base_board_matrix_m(
    end_pose_in_ref: PoseSnapshot,
    tool_cam_m: np.ndarray,
    board_pose_camera_board_m: np.ndarray,
) -> np.ndarray:
    """严格复用 charuco_pose_drag 的 base_board 计算链路。"""

    return (
        np.asarray(end_pose_in_ref.pose_matrix, dtype=np.float64).reshape(4, 4)
        @ np.asarray(tool_cam_m, dtype=np.float64).reshape(4, 4)
        @ np.asarray(board_pose_camera_board_m, dtype=np.float64).reshape(4, 4)
    )


def _enable_drag_mode(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> None:
    _ensure_fixed_toolset(robot, ec)
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("切换非实时运动模式失败")
    robot.setPowerState(False, ec)
    _print_sdk_result("setPowerState(False)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前下电失败")
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前切换手动模式失败")
    robot.moveReset(ec)
    _print_sdk_result("moveReset(drag)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前 moveReset 失败")
    robot.enableDrag(
        int(xCoreSDK_python.DragParameterSpace.cartesianSpace),
        int(xCoreSDK_python.DragParameterType.freely),
        ec,
        enable_drag_button=False,
    )
    _print_sdk_result("enableDrag(cartesianSpace, freely)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("开启拖动失败")


def _ensure_fixed_toolset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> xCoreSDK_python.Toolset:
    toolset = _apply_named_toolset(robot, ec)
    if toolset is None:
        raise RuntimeError(f"设置默认工具/工件失败: tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    return toolset


def _read_current_toolset_text(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> str:
    toolset = _ensure_fixed_toolset(robot, ec)
    end_snapshot = _pose_snapshot_from_frame(toolset.end)
    ref_snapshot = _pose_snapshot_from_frame(toolset.ref)
    return (
        f"tool={DEFAULT_TOOL_NAME} end={_format_pose_line(end_snapshot)} "
        f"wobj={DEFAULT_WOBJ_NAME} ref={_format_pose_line(ref_snapshot)}"
    )


# endregion


# region 预览绘制
def _draw_preview(
    frame_bgr: np.ndarray,
    charuco_result: Any,
    calibration: Any,
    prior: PriorKnowledge,
    state: RuntimeState,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if charuco_result is not None and charuco_result.marker_corners_px:
        cv2.aruco.drawDetectedMarkers(
            canvas, charuco_result.marker_corners_px, charuco_result.marker_ids, borderColor=(255, 180, 0)
        )
    if charuco_result is not None and charuco_result.rvec is not None and charuco_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            np.asarray(charuco_result.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(charuco_result.tvec, dtype=np.float64).reshape(3, 1),
            30.0,
            2,
        )
    lines = _build_overlay_lines(prior, state, charuco_result)
    return _draw_text_block(canvas, lines, (18, 24))


def _build_overlay_lines(
    prior: PriorKnowledge,
    state: RuntimeState,
    charuco_result: Any,
) -> list[str]:
    marker_count = 0 if charuco_result is None else int(charuco_result.marker_count)
    charuco_count = 0 if charuco_result is None else int(charuco_result.charuco_count)
    reproj = (
        "NA"
        if charuco_result is None or charuco_result.reprojection_error_px is None
        else f"{float(charuco_result.reprojection_error_px):.4f}"
    )
    lines = [
        f"board_visible={bool(state.latest_snapshot is not None)} marker={marker_count} charuco={charuco_count} reproj={reproj}",
        f"prior={prior.source_path.name}",
        f"action={state.last_action_text}",
        "calib_base_board=" + _format_pose_line(_pose_snapshot_from_matrix_m(prior.calibration_base_board_m)),
        "reference_offset=" + _format_pose_line(state.reference_offset_pose),
    ]
    if state.latest_snapshot is None:
        lines.append(f"toolset=tool={DEFAULT_TOOL_NAME} wobj={DEFAULT_WOBJ_NAME}")
        lines.append("current_tcp=NA")
        lines.append("current_cam_board=NA")
        lines.append("current_base_board=NA")
        lines.append("offset=NA")
        lines.append("delta_offset=NA")
        lines.append("prior_to_current_base_board=NA")
    else:
        lines.extend(
            [
                "toolset=" + state.latest_snapshot.toolset_text,
                "current_tcp=" + _format_pose_line(state.latest_snapshot.current_end_pose),
                "current_cam_board=" + _format_pose_line(state.latest_snapshot.current_cam_board_pose),
                "current_base_board=" + _format_pose_line(state.latest_snapshot.current_base_board_pose),
                "offset=" + _format_pose_line(state.latest_snapshot.offset_pose),
                "delta_offset=" + _format_pose_line(state.latest_snapshot.delta_offset_pose),
                "prior_to_current_base_board="
                + _format_pose_line(state.latest_snapshot.prior_to_current_base_board_pose),
            ]
        )
    lines.append("启动即拖动模式   Space=记录 offset 基准   观察 delta_offset 是否接近 0   Q/Esc=quit")
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
    rotation = Rotation3D.from_euler(
        "xyz",
        np.asarray(cartesian_pose.rpy, dtype=np.float64).reshape(3),
        degrees=False,
    ).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(cartesian_pose.trans, dtype=np.float64).reshape(3)
    return _pose_snapshot_from_matrix_m(matrix)


def _pose_snapshot_from_matrix_m(pose_matrix: np.ndarray) -> PoseSnapshot:
    matrix = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    rpy_deg = Rotation3D.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return PoseSnapshot(
        pose_matrix=matrix,
        translation_mm=(float(matrix[0, 3] * 1000.0), float(matrix[1, 3] * 1000.0), float(matrix[2, 3] * 1000.0)),
        rpy_deg=(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])),
        recorded_at_iso=datetime.now().isoformat(timespec="milliseconds"),
    )


def _pose_snapshot_from_frame(frame: xCoreSDK_python.Frame) -> PoseSnapshot:
    rotation = Rotation3D.from_euler(
        "xyz",
        np.asarray(frame.rpy, dtype=np.float64).reshape(3),
        degrees=False,
    ).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(frame.trans, dtype=np.float64).reshape(3)
    return _pose_snapshot_from_matrix_m(matrix)


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
    parser = argparse.ArgumentParser(description="charuco rebase jog")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_ORIN_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP)
    parser.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS)
    parser.add_argument("--hand-eye-result-path", type=Path, default=DEFAULT_HAND_EYE_RESULT_PATH)
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
            hand_eye_result_path=Path(args.hand_eye_result_path),
        )
    )
