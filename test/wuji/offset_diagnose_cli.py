from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.wuji.ball_pose_detection import (
    DEFAULT_CAMERA_NAME as DEFAULT_BALL_CAMERA_NAME,
)
from test.wuji.ball_pose_detection import (
    DEFAULT_SERVICE_ADDR as DEFAULT_BALL_SERVICE_ADDR,
)
from test.wuji.ball_pose_detection import (
    _build_priors_from_capture,
    _build_three_ball_basis_transform,
    _load_prior_capture,
)
from test.wuji.record_left_replay_cli import (
    DEFAULT_HAND_EYE_RESULT_PATH,
    _format_sequence,
    _homogeneous_matrix_to_cartesian_position,
    _homogeneous_matrix_to_rpy,
    _load_tool_camera_transform_m,
)
from test.wuji.xcoresdk_arm_cli_test import (
    DEFAULT_CARTESIAN_ZONE,
    DEFAULT_JOINT_SPEED,
    DEFAULT_JOINT_ZONE,
    DEFAULT_TOOL_NAME,
    DEFAULT_WOBJ_NAME,
    LEFT_ARM_IP,
    _apply_named_toolset,
    _deg_to_rad,
    _ensure_nrt_motion_ready,
    _m_to_mm,
    _mm_to_m,
    _print_sdk_result,
    _shutdown_robot,
    _validate_cartesian_target,
)

from sdk.xcoresdk import xCoreSDK_python

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "offset_diagnose_cli"
DEFAULT_ARM_IP = LEFT_ARM_IP
DEFAULT_PRIOR_CAPTURE_PATH = (
    PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture" / "summary.json"
)


@dataclass(frozen=True, slots=True)
class DiagnoseResult:
    prior_base_ball_m: np.ndarray
    current_base_ball_m: np.ndarray
    offset_m: np.ndarray
    new_tcp_m: np.ndarray
    target_tcp_m: np.ndarray


@dataclass(slots=True)
class RuntimeState:
    latest_result: DiagnoseResult | None = None
    current_target_joint_deg: list[float] | None = None
    current_target_tcp_m: np.ndarray | None = None
    current_target_pose: xCoreSDK_python.CartesianPosition | None = None
    current_target_offset_m: np.ndarray | None = None
    offset_applied: bool = False


def main(
    service_addr: str = DEFAULT_BALL_SERVICE_ADDR,
    camera_name: str = DEFAULT_BALL_CAMERA_NAME,
    arm_ip: str = DEFAULT_ARM_IP,
    prior_capture_path: Path = DEFAULT_PRIOR_CAPTURE_PATH,
    hand_eye_result_path: Path = DEFAULT_HAND_EYE_RESULT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    logger.info("offset 诊断页启动")
    logger.info("服务地址={} 相机={} 机械臂={}", service_addr, camera_name, arm_ip)
    logger.info("先验 summary 路径={} (必须包含 tcp_pose_matrix / local_pose_transform)", prior_capture_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_capture = _load_prior_capture(prior_capture_path)
    _build_priors_from_capture(prior_capture)

    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(str(arm_ip))
    state = RuntimeState()
    try:
        _apply_fixed_toolset(robot, ec)
        if not _ensure_nrt_motion_ready(robot, ec):
            raise RuntimeError("机械臂未准备好")

        while True:
            command = input(
                "输入 f=打开相机预览, 7个关节角(deg)=直接MoveAbsJ, "
                "6个笛卡尔xyzrpy(mm/deg)=应用offset后运动, q=退出: "
            ).strip()
            if command.lower() == "q":
                if input("确认退出并关闭窗口? y/N: ").strip().lower() == "y":
                    cv2.destroyAllWindows()
                    return 0
                continue
            if command.lower() == "f":
                _run_camera_preview(
                    service_addr=service_addr,
                    camera_name=camera_name,
                    prior_capture_path=prior_capture_path,
                    hand_eye_result_path=hand_eye_result_path,
                    output_dir=output_dir,
                    robot=robot,
                    ec=ec,
                    state=state,
                )
                continue
            try:
                target_values = _parse_numeric_values(command)
            except ValueError as exc:
                logger.warning("输入格式错误: {}", exc)
                continue
            if len(target_values) == 7:
                state.current_target_joint_deg = target_values
                _execute_joint_target(robot, ec, target_values)
                continue
            if len(target_values) == 6:
                _execute_cartesian_target_with_offset(
                    robot=robot,
                    ec=ec,
                    state=state,
                    target_xyzrpy_mm_deg=target_values,
                )
                continue
            logger.warning("输入数量无效: {}，需要 7 个关节角或 6 个笛卡尔 xyzrpy", len(target_values))
    finally:
        cv2.destroyAllWindows()
        _shutdown_robot(robot, ec)


def _parse_numeric_values(raw_text: str) -> list[float]:
    values = [token for token in raw_text.replace(",", " ").replace("，", " ").split() if token]
    if not values:
        raise ValueError("请输入数字列表")
    return [float(value) for value in values]


def _execute_joint_target(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_joint_deg: list[float],
) -> None:
    """输入 7 个关节角时只执行关节运动，不应用 offset。"""

    target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(target_joint_deg))
    cmd_id = xCoreSDK_python.PyString()
    robot.moveReset(ec)
    _print_sdk_result("moveReset(joint)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("关节运动 moveReset 失败")
    robot.moveAppend(
        [xCoreSDK_python.MoveAbsJCommand(target_joint, DEFAULT_JOINT_SPEED, DEFAULT_JOINT_ZONE)], cmd_id, ec
    )
    _print_sdk_result("moveAppend(MoveAbsJ-joint-no-offset)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("关节运动 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(joint)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("关节运动 moveStart 失败")
    logger.info("已下发关节运动，不应用 offset: joint(deg)=[{}]", _format_sequence(target_joint_deg))


def _execute_cartesian_target_with_offset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    state: RuntimeState,
    target_xyzrpy_mm_deg: list[float],
) -> None:
    """输入 6 个 xyzrpy(mm/deg) 时应用 offset 后执行，优先 MoveL。"""

    if state.latest_result is None:
        raise RuntimeError("尚未计算 offset，先输入 f 打开预览并按空格计算 offset")
    _apply_fixed_toolset(robot, ec)
    source_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前 TCP 失败")
    target_pose = xCoreSDK_python.CartesianPosition(
        _mm_to_m(target_xyzrpy_mm_deg[:3]) + _deg_to_rad(target_xyzrpy_mm_deg[3:])
    )
    target_pose.hasElbow = source_pose.hasElbow
    target_pose.elbow = source_pose.elbow
    target_pose.confData = list(source_pose.confData)

    target_tcp_m = _cartesian_to_matrix_m(target_pose)
    # 诊断页与 replay 保持一致：T_new_tcp = inv(T_off) @ T_tcp。
    new_tcp_m = np.linalg.inv(state.latest_result.offset_m) @ target_tcp_m
    new_tcp_pose = _homogeneous_matrix_to_cartesian_position(target_pose, new_tcp_m)
    state.current_target_tcp_m = target_tcp_m
    state.current_target_pose = new_tcp_pose
    state.current_target_offset_m = new_tcp_m
    logger.info("输入笛卡尔目标 xyzrpy(mm/deg)=[{}]", _format_sequence(target_xyzrpy_mm_deg))
    logger.info("应用 offset 后 T_new_tcp xyz(mm)=[{}]", _format_sequence(_m_to_mm(list(new_tcp_m[:3, 3]))))

    target_joint = _solve_ik_for_cartesian_target(robot, ec, new_tcp_pose)
    if target_joint is None:
        logger.warning("offset 后 T_new_tcp 逆解失败，取消本次笛卡尔运动")
        return
    should_fallback_to_move_abs_j = not _validate_cartesian_target(robot, ec, new_tcp_pose)
    if should_fallback_to_move_abs_j:
        logger.warning("MoveL 路径检查失败，回退 MoveAbsJ")
    _execute_cartesian_motion(
        robot=robot,
        ec=ec,
        target_pose=new_tcp_pose,
        target_joint=target_joint,
        should_fallback_to_move_abs_j=should_fallback_to_move_abs_j,
    )
    _print_offset_application_candidates(target_tcp_m, state.latest_result.offset_m)
    state.offset_applied = True


def _run_camera_preview(
    service_addr: str,
    camera_name: str,
    prior_capture_path: Path,
    hand_eye_result_path: Path,
    output_dir: Path,
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    state: RuntimeState,
) -> None:
    from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest
    from camera_pipeline.client import CameraPipelineClient

    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=30_000)
    try:
        cv2.namedWindow("offset_diagnose_cli", cv2.WINDOW_NORMAL)
        frame_stream = client.subscribe_camera_color_frames(camera_name)
        while True:
            frame = next(frame_stream)
            frame_bgr = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
            detection_response = client.request_ball_pose_detection(
                BallPoseDetectionRequest(
                    request_id=1,
                    camera_name=str(camera_name),
                    frame_id=int(frame.frame_id),
                    enable_debug=True,
                    priors=tuple(priors),
                )
            )
            if detection_response.error is not None or detection_response.matched_count < 3:
                cv2.imshow("offset_diagnose_cli", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    logger.warning("本帧检测失败，无法计算 offset")
                continue
            cam_ball_m = _build_three_ball_basis_transform(detection_response.detections)
            if cam_ball_m is None:
                logger.warning("三球坐标系构造失败")
                continue
            cam_ball_m = np.asarray(cam_ball_m, dtype=np.float64).copy()
            cam_ball_m[:3, 3] *= 0.001
            _apply_fixed_toolset(robot, ec)
            tcp_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
            if ec.get("ec", 0) != 0:
                raise RuntimeError("读取当前 TCP 失败")
            tcp_m = _cartesian_to_matrix_m(tcp_pose)
            tool_cam_m = _load_tool_camera_transform_m(hand_eye_result_path)
            prior_base_ball_m = _load_prior_base_ball_transform(prior_capture_path, hand_eye_result_path)
            current_base_ball_m = tcp_m @ tool_cam_m @ cam_ball_m
            offset_m = current_base_ball_m @ np.linalg.inv(prior_base_ball_m)
            new_tcp_m = np.linalg.inv(offset_m) @ tcp_m
            state.latest_result = DiagnoseResult(
                prior_base_ball_m=prior_base_ball_m,
                current_base_ball_m=current_base_ball_m,
                offset_m=offset_m,
                new_tcp_m=new_tcp_m,
                target_tcp_m=tcp_m,
            )
            preview = _build_preview_image(
                frame_bgr=frame_bgr,
                detection_response=detection_response,
                result=state.latest_result,
                tool_cam_m=tool_cam_m,
                cam_ball_m=cam_ball_m,
                tcp_m=tcp_m,
            )
            cv2.imshow("offset_diagnose_cli", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                _print_diagnose_result(state.latest_result)
                _save_result(output_dir, state.latest_result, state.current_target_joint_deg or [])
                state.offset_applied = False
                logger.success("已计算 offset；后续输入 6 个笛卡尔 xyzrpy 时才会应用")
                break
    finally:
        client.close()
        cv2.destroyWindow("offset_diagnose_cli")


def _apply_latest_offset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    state: RuntimeState,
    output_dir: Path,
) -> None:
    if state.latest_result is None:
        logger.warning("没有可应用的 offset")
        return
    payload = {
        "T_off": state.latest_result.offset_m.tolist(),
        "T_new_tcp": state.latest_result.new_tcp_m.tolist(),
    }
    (output_dir / "latest_offset.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("已在退出前记录最新 offset 到 {}", output_dir / "latest_offset.json")


def _build_preview_image(
    frame_bgr: np.ndarray,
    detection_response: object,
    result: DiagnoseResult | None,
    tool_cam_m: np.ndarray | None = None,
    cam_ball_m: np.ndarray | None = None,
    tcp_m: np.ndarray | None = None,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    _draw_detected_balls(overlay, detection_response)
    if cam_ball_m is not None:
        _draw_cam_ball_axes(overlay, cam_ball_m, detection_response)
    lines = [
        "SPACE=compute offset  Q=quit  输入关节值=执行",
        f"toolset={DEFAULT_TOOL_NAME}/{DEFAULT_WOBJ_NAME} fixed before every cartPosture(endInRef)",
    ]
    if result is None:
        lines.append("offset: NA")
    else:
        lines.extend(
            [
                "T_prior_base_ball = T_tcp @ T_tool_cam @ T_cam_ball",
                "T_off = T_tcp @ T_tool_cam @ T_cam_ball @ inv(T_prior_base_ball)",
                "T_new_tcp = inv(T_off) @ T_tcp",
                f"offset_xyz_mm={_format_xyz_mm(result.offset_m)}",
                f"offset_rpy_deg={_format_rpy_deg(result.offset_m)}",
                f"prior_base_ball_xyz_mm={_format_xyz_mm(result.prior_base_ball_m)}",
                f"prior_base_ball_rpy_deg={_format_rpy_deg(result.prior_base_ball_m)}",
                f"current_base_ball_xyz_mm={_format_xyz_mm(result.current_base_ball_m)}",
                f"current_base_ball_rpy_deg={_format_rpy_deg(result.current_base_ball_m)}",
                f"new_tcp_xyz_mm={_format_xyz_mm(result.new_tcp_m)}",
                f"new_tcp_rpy_deg={_format_rpy_deg(result.new_tcp_m)}",
            ]
        )
    if tcp_m is not None:
        lines.append(f"tcp_xyz_mm={_format_xyz_mm(tcp_m)}")
        lines.append(f"tcp_rpy_deg={_format_rpy_deg(tcp_m)}")
    if tool_cam_m is not None:
        lines.append(f"tool_cam_xyz_mm={_format_xyz_mm(tool_cam_m)}")
        lines.append(f"tool_cam_rpy_deg={_format_rpy_deg(tool_cam_m)}")
    if cam_ball_m is not None:
        lines.append(f"cam_ball_xyz_mm={_format_xyz_mm(cam_ball_m)}")
        lines.append(f"cam_ball_rpy_deg={_format_rpy_deg(cam_ball_m)}")
    lines.extend(_build_detection_lines(detection_response))
    _draw_text_lines(overlay, lines)
    return overlay


def _build_detection_lines(detection_response: object) -> list[str]:
    lines = [f"matched_count={getattr(detection_response, 'matched_count', 'NA')}"]
    for index, item in enumerate(_iter_detection_items(detection_response), start=1):
        color_hex = str(item.get("color_hex", "unknown"))
        center_mm = _as_vector3(item.get("center_mm"))
        center_px = _as_vector2(item.get("center_px"))
        radius_px = item.get("radius_px", "NA")
        status = item.get("status", "NA")
        if center_mm is None:
            lines.append(f"ball{index} {color_hex}: center_mm=NA px={_format_optional_px(center_px)} status={status}")
            continue
        lines.append(
            f"ball{index} {color_hex}: mm=({center_mm[0]:.2f},{center_mm[1]:.2f},{center_mm[2]:.2f}) "
            f"px={_format_optional_px(center_px)} r_px={radius_px} status={status}"
        )
    return lines


def _draw_detected_balls(image_bgr: np.ndarray, detection_response: object) -> None:
    intrinsics = _get_camera_intrinsics(detection_response)
    for index, item in enumerate(_iter_detection_items(detection_response), start=1):
        center_mm = _as_vector3(item.get("center_mm"))
        center_px = _as_vector2(item.get("center_px"))
        if center_px is None and center_mm is not None and intrinsics is not None:
            center_px = _project_point_to_pixel(center_mm, intrinsics)
        if center_px is None:
            continue
        px = (int(round(float(center_px[0]))), int(round(float(center_px[1]))))
        color_hex = str(item.get("color_hex", "#ffffff"))
        color_bgr = _hex_to_bgr(color_hex)
        raw_radius_px = item.get("radius_px")
        radius_px = 10 if raw_radius_px is None else int(round(float(str(raw_radius_px))))
        radius_px = max(6, min(radius_px, 80))
        cv2.circle(image_bgr, px, radius_px, color_bgr, 2, cv2.LINE_AA)
        cv2.circle(image_bgr, px, 3, (255, 255, 255), -1, cv2.LINE_AA)
        label_lines = [f"B{index} {color_hex}", f"px=({px[0]},{px[1]})"]
        if center_mm is not None:
            label_lines.append(f"mm=({center_mm[0]:.1f},{center_mm[1]:.1f},{center_mm[2]:.1f})")
        _draw_label_block(image_bgr, label_lines, (px[0] + 8, px[1] - 8))


def _draw_cam_ball_axes(image_bgr: np.ndarray, cam_ball_m: np.ndarray, detection_response: object) -> None:
    intrinsics = _get_camera_intrinsics(detection_response)
    if intrinsics is None:
        return
    rotation = np.asarray(cam_ball_m[:3, :3], dtype=np.float64)
    origin_mm = np.asarray(cam_ball_m[:3, 3], dtype=np.float64) * 1000.0
    origin_px = _project_point_to_pixel(origin_mm, intrinsics)
    if origin_px is None:
        return
    origin_point = (int(round(float(origin_px[0]))), int(round(float(origin_px[1]))))
    axis_length_mm = 35.0
    axis_points_mm = (
        origin_mm + rotation[:, 0] * axis_length_mm,
        origin_mm + rotation[:, 1] * axis_length_mm,
        origin_mm + rotation[:, 2] * axis_length_mm,
    )
    axis_labels = ("X", "Y", "Z")
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    cv2.circle(image_bgr, origin_point, 6, (255, 255, 255), -1, cv2.LINE_AA)
    for label, point_mm, color in zip(axis_labels, axis_points_mm, axis_colors, strict=True):
        point_px = _project_point_to_pixel(point_mm, intrinsics)
        if point_px is None:
            continue
        point = (int(round(float(point_px[0]))), int(round(float(point_px[1]))))
        cv2.arrowedLine(image_bgr, origin_point, point, color, 3, cv2.LINE_AA, tipLength=0.2)
        _draw_label_block(image_bgr, [f"ball {label}"], (point[0] + 6, point[1] + 6))
    _draw_label_block(
        image_bgr,
        [
            "T_cam_ball",
            f"xyz_mm={_format_xyz_mm(cam_ball_m)}",
            f"rpy_deg={_format_rpy_deg(cam_ball_m)}",
        ],
        (origin_point[0] + 10, origin_point[1] + 10),
    )


def _draw_label_block(image_bgr: np.ndarray, lines: list[str], position: tuple[int, int]) -> None:
    x, y = position
    for index, line in enumerate(lines):
        text_position = (x, y + index * 18)
        cv2.putText(image_bgr, line, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, line, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def _iter_detection_items(detection_response: object) -> list[dict[str, object]]:
    detections = getattr(detection_response, "detections", ())
    if not isinstance(detections, list | tuple):
        return []
    return [dict(item) for item in detections if isinstance(item, dict)]


def _get_camera_intrinsics(detection_response: object) -> tuple[float, float, float, float] | None:
    debug = getattr(detection_response, "debug", None)
    intrinsics = None if debug is None else getattr(debug, "camera_intrinsics", None)
    vector = np.asarray(intrinsics, dtype=np.float64)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        return None
    return (float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3]))


def _project_point_to_pixel(
    point_mm: np.ndarray,
    camera_intrinsics: tuple[float, float, float, float],
) -> np.ndarray | None:
    point = np.asarray(point_mm, dtype=np.float64).reshape(3)
    z_mm = float(point[2])
    if z_mm <= 1e-6:
        return None
    fx, fy, cx, cy = camera_intrinsics
    x_px = fx * float(point[0]) / z_mm + cx
    y_px = fy * float(point[1]) / z_mm + cy
    if not np.isfinite(x_px) or not np.isfinite(y_px):
        return None
    return np.asarray([x_px, y_px], dtype=np.float64)


def _as_vector3(value: object) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _as_vector2(value: object) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _format_optional_px(center_px: np.ndarray | None) -> str:
    if center_px is None:
        return "NA"
    return f"({float(center_px[0]):.1f},{float(center_px[1]):.1f})"


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    color = color_hex.strip().lstrip("#")
    if len(color) != 6:
        return (255, 255, 255)
    try:
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
    except ValueError:
        return (255, 255, 255)
    return (blue, green, red)


def _draw_text_lines(image_bgr: np.ndarray, lines: list[str]) -> None:
    x0, y0 = 20, 30
    for index, line in enumerate(lines):
        position = (x0, y0 + index * 24)
        cv2.putText(image_bgr, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def _format_xyz_mm(matrix: np.ndarray) -> str:
    return f"({float(matrix[0,3]) * 1000.0:.2f}, {float(matrix[1,3]) * 1000.0:.2f}, {float(matrix[2,3]) * 1000.0:.2f})"


def _format_rpy_deg(matrix: np.ndarray) -> str:
    rpy = np.degrees(
        np.asarray(_homogeneous_matrix_to_rpy(np.asarray(matrix, dtype=np.float64).tolist()), dtype=np.float64)
    )
    return f"({float(rpy[0]):.2f}, {float(rpy[1]):.2f}, {float(rpy[2]):.2f})"


def _detect_current_cam_ball_m(service_addr: str, camera_name: str, prior_capture_path: Path) -> np.ndarray:
    from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest
    from camera_pipeline.client import CameraPipelineClient

    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=30_000)
    try:
        response = client.request_ball_pose_detection(
            BallPoseDetectionRequest(
                request_id=1,
                camera_name=str(camera_name),
                frame_id=-1,
                enable_debug=True,
                priors=tuple(priors),
            )
        )
    finally:
        client.close()
    if response.error is not None:
        raise RuntimeError(f"ball pose detection 返回错误: {response.error}")
    if response.matched_count < 3:
        raise RuntimeError("ball pose detection 未返回足够的三球检测结果")
    ball_m = _build_three_ball_basis_transform(response.detections)
    if ball_m is None:
        raise RuntimeError("当前三球坐标系构造失败")
    ball_m = np.asarray(ball_m, dtype=np.float64).copy()
    ball_m[:3, 3] *= 0.001
    return ball_m


def _load_prior_base_ball_transform(prior_capture_path: Path, hand_eye_result_path: Path) -> np.ndarray:
    prior_capture = _load_prior_capture(prior_capture_path)
    tcp_pose_matrix = prior_capture.get("tcp_pose_matrix")
    local_pose_transform = prior_capture.get("local_pose_transform")
    if tcp_pose_matrix is None or local_pose_transform is None:
        raise RuntimeError(f"先验文件缺少 tcp_pose_matrix 或 local_pose_transform: {prior_capture_path}")
    tcp_m = np.asarray(tcp_pose_matrix, dtype=np.float64).copy()
    cam_ball_m = np.asarray(local_pose_transform, dtype=np.float64).copy()
    cam_ball_m[:3, 3] *= 0.001
    tool_cam_m = _load_tool_camera_transform_m(hand_eye_result_path)
    return tcp_m @ tool_cam_m @ cam_ball_m


def _apply_fixed_toolset(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """和 hand_eye_orin_left_arm_drag.py 保持一致：每次读 endInRef 前固定 tool/wobj。"""

    if _apply_named_toolset(robot, ec) is None:
        raise RuntimeError(f"设置固定 toolset 失败: tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")


def _cartesian_to_matrix_m(cart_pose: xCoreSDK_python.CartesianPosition) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    # hand_eye_orin_left_arm_drag.py 的实际求解链路等价于使用
    # scipy Rotation.from_euler("xyz", SDK rpy(rad), degrees=False) 重建 T_tcp。
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", np.asarray(cart_pose.rpy, dtype=np.float64).reshape(3), degrees=False
    ).as_matrix()
    matrix[:3, 3] = np.asarray(cart_pose.trans, dtype=np.float64).reshape(3)
    return matrix


def _print_diagnose_result(result: DiagnoseResult) -> None:
    def _line(name: str, matrix: np.ndarray) -> str:
        return (
            f"{name}: xyz(m)=({matrix[0,3]:.6f}, {matrix[1,3]:.6f}, {matrix[2,3]:.6f}) "
            f"rpy_deg={_format_rpy_deg(matrix)}"
        )

    print(_line("T_prior_base_ball", result.prior_base_ball_m))
    print(_line("T_current_base_ball", result.current_base_ball_m))
    print(_line("T_off", result.offset_m))
    print(_line("T_new_tcp", result.new_tcp_m))
    print(f"offset_norm(m)={float(np.linalg.norm(result.offset_m[:3, 3])):.6f}")


def _print_offset_application_candidates(target_tcp_m: np.ndarray, offset_m: np.ndarray) -> None:
    inverse_offset_m = np.linalg.inv(offset_m)
    candidates = (
        ("T_new_tcp_1 = inv(T_off) @ T_tcp", inverse_offset_m @ target_tcp_m),
        ("T_new_tcp_2 = T_tcp @ T_off", target_tcp_m @ offset_m),
        ("T_new_tcp_3 = T_off @ T_tcp", offset_m @ target_tcp_m),
        ("T_new_tcp_4 = T_tcp @ inv(T_off)", target_tcp_m @ inverse_offset_m),
    )
    print("offset 应用候选结果，平移单位 mm，欧拉单位 deg：")
    for label, matrix in candidates:
        print(f"{label}: xyz_mm={_format_xyz_mm(matrix)} rpy_deg={_format_rpy_deg(matrix)}")


def _save_result(output_dir: Path, result: DiagnoseResult, target_joint_deg: list[float]) -> None:
    payload = {
        "target_joint_deg": target_joint_deg,
        "T_prior_base_ball": result.prior_base_ball_m.tolist(),
        "T_current_base_ball": result.current_base_ball_m.tolist(),
        "T_off": result.offset_m.tolist(),
        "T_new_tcp": result.new_tcp_m.tolist(),
    }
    (output_dir / "diagnose.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_ik_for_cartesian_target(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_pose: xCoreSDK_python.CartesianPosition,
) -> xCoreSDK_python.JointPosition | None:
    robot_model = robot.model()
    _apply_fixed_toolset(robot, ec)
    toolset = robot.toolset(ec)
    _print_sdk_result("toolset(cartesian-offset)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取 toolset 失败")
    ik = robot_model.calcIk(target_pose, toolset, ec)
    _print_sdk_result("calcIk(cartesian-offset)", ec)
    if ec.get("ec", 0) != 0:
        logger.warning("逆解失败，无法执行 MoveAbsJ: ec={} message={}", ec.get("ec", 0), ec.get("message", ""))
        return None
    return ik


def _execute_cartesian_motion(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_pose: xCoreSDK_python.CartesianPosition,
    target_joint: xCoreSDK_python.JointPosition,
    should_fallback_to_move_abs_j: bool,
) -> None:
    cmd_id = xCoreSDK_python.PyString()
    robot.moveReset(ec)
    _print_sdk_result("moveReset(cartesian-offset)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("moveReset 失败")
    if should_fallback_to_move_abs_j:
        robot.moveAppend(
            [xCoreSDK_python.MoveAbsJCommand(target_joint, DEFAULT_JOINT_SPEED, DEFAULT_JOINT_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveAbsJ-cartesian-offset)", ec)
    else:
        robot.moveAppend(
            [xCoreSDK_python.MoveLCommand(target_pose, 20.0, DEFAULT_CARTESIAN_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveL-cartesian-offset)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("笛卡尔 offset 运动 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(cartesian-offset)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("笛卡尔 offset 运动 moveStart 失败")
    logger.info("已下发 {}", "MoveAbsJ 回退运动" if should_fallback_to_move_abs_j else "MoveL 笛卡尔运动")


if __name__ == "__main__":
    raise SystemExit(main())
