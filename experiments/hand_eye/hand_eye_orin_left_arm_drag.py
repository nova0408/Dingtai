from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rotation3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from camera_pipeline.camera_stream import CameraColorFramePacket  # noqa: E402
from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from src.calibration import CHARUCO_200_12_9  # noqa: E402
from src.calibration import CharucoPoseEstimator  # noqa: E402
from src.calibration import HandEyeCalibrationResult  # noqa: E402
from src.calibration import calibrate_hand_eye_from_pose_sequences  # noqa: E402
from src.calibration.hand_eye import PoseLike  # noqa: E402
from src.utils.datas import Quaternion, Transform, Translation  # noqa: E402

# region 默认参数
DEFAULT_WINDOW_NAME = "Left Arm Hand-Eye Calibration"
DEFAULT_WINDOW_WIDTH = 1680
DEFAULT_WINDOW_HEIGHT = 960
DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_LEFT_ARM_IP = "192.168.1.161"
DEFAULT_TOOL_NAME = "g_tool_0"
DEFAULT_WOBJ_NAME = "g_wobj_0"
DEFAULT_OUTPUT_ROOT = Path("experiments/hand_eye/runs")
DEFAULT_CAMERA_TIMEOUT_S = 10.0
DEFAULT_MIN_CHARUCO_CORNERS = 6
DEFAULT_SDK_RPY_SEQUENCE = "xyz"
EXPECTED_LEFT_ARM_TYPE = "AR5-5_0.8L-W4C1C9-ZY2"
AXIS_DRAW_LENGTH_MM = 28.0
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_FONT_SIZE = 20
TRANSLATION_SPAN_TARGETS_MM = {
    "x": 120.0,
    "y": 120.0,
    "z": 80.0,
}
ROTATION_SPAN_TARGETS_DEG = {
    "roll": 35.0,
    "pitch": 35.0,
    "yaw": 45.0,
}
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_name: str
    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    host_timestamp_iso: str
    joint_degrees: tuple[float, ...]
    tcp_pose_base_tool: Transform
    tcp_translation_mm: tuple[float, float, float]
    tcp_rpy_degrees: tuple[float, float, float]
    sdk_pose_has_elbow: bool
    sdk_pose_elbow_deg: float
    sdk_pose_conf_data: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_index: int
    host_timestamp_iso: str
    host_timestamp_s: float
    frame_id: int
    camera_timestamp_ms: float
    board_visible: bool
    marker_count: int
    charuco_count: int
    reprojection_error_px: float | None
    robot_snapshot: RobotSnapshot
    board_pose_camera_board: Transform | None
    board_pose_board_camera: Transform | None
    raw_frame_path: Path
    preview_frame_path: Path


@dataclass(frozen=True, slots=True)
class ConnectedLeftArm:
    robot_ip: str
    robot: xCoreSDK_python.xMateErProRobot
    robot_type: str
    robot_uid: str
    ec: dict[str, object]


@dataclass(frozen=True, slots=True)
class SamplingCoverageSummary:
    sample_count: int
    span_x_mm: float
    span_y_mm: float
    span_z_mm: float
    span_roll_deg: float
    span_pitch_deg: float
    span_yaw_deg: float


# endregion


# region 主流程
def main() -> int:
    args = _parse_cli()
    _validate_runtime_requirements()
    session_dir = _create_session_dir(Path(args.output_root))
    logger.info(f"输出目录: {session_dir}")

    left_arm = _connect_left_arm(
        robot_ip=str(args.left_arm_ip),
        tool_name=str(args.tool_name),
        wobj_name=str(args.wobj_name),
    )
    try:
        _enable_left_arm_drag(left_arm)
        _run_calibration_session(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            session_dir=session_dir,
            left_arm=left_arm,
            tool_name=str(args.tool_name),
            wobj_name=str(args.wobj_name),
            min_charuco_corners=int(args.min_charuco_corners),
            sdk_rpy_sequence=str(args.sdk_rpy_sequence),
        )
        return 0
    finally:
        _shutdown_left_arm(left_arm)


def _run_calibration_session(
    service_addr: str,
    camera_name: str,
    session_dir: Path,
    left_arm: ConnectedLeftArm,
    tool_name: str,
    wobj_name: str,
    min_charuco_corners: int,
    sdk_rpy_sequence: str,
) -> None:
    client = CameraPipelineClient(
        service_addr=service_addr,
        timeout_ms=int(DEFAULT_CAMERA_TIMEOUT_S * 1000.0),
    )
    estimator = CharucoPoseEstimator(CHARUCO_200_12_9)
    calibration = _read_camera_calibration(client)
    raw_frames_dir = session_dir / "frames_raw"
    preview_frames_dir = session_dir / "frames_preview"
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    preview_frames_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_result_path = session_dir / "tool_camera_extrinsic_result.txt"
    samples: list[SampleRecord] = []
    last_result: HandEyeCalibrationResult | None = None
    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    try:
        for frame_packet in client.subscribe_camera_color_frames(camera_name):
            robot_snapshot = _read_left_arm_snapshot(left_arm, sdk_rpy_sequence=sdk_rpy_sequence)
            frame_bgr = np.asarray(frame_packet.color_bgr, dtype=np.uint8).copy()
            started = time.perf_counter()
            charuco_result = estimator.estimate_pose(
                image_bgr=frame_bgr,
                camera_matrix=calibration.camera_matrix,
                dist_coeffs=calibration.dist_coeffs,
                min_charuco_corners=min_charuco_corners,
            )
            compute_ms = (time.perf_counter() - started) * 1000.0
            board_pose_camera_board = (
                None if charuco_result.transform_se3 is None else Transform.from_SE3(charuco_result.transform_se3)
            )
            board_pose_board_camera = (
                None if board_pose_camera_board is None else Transform.from_SE3(_invert_transform(board_pose_camera_board))
            )
            preview_bgr = _draw_preview(
                frame_bgr=frame_bgr,
                frame_packet=frame_packet,
                robot_snapshot=robot_snapshot,
                charuco_result=charuco_result,
                calibration_result=last_result,
                camera_calibration=calibration,
                compute_ms=compute_ms,
                recorded_samples=samples,
                tool_name=tool_name,
                wobj_name=wobj_name,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (13, 32, ord("p"), ord("P")):
                sample_record = _capture_sample(
                    session_dir=session_dir,
                    raw_frames_dir=raw_frames_dir,
                    preview_frames_dir=preview_frames_dir,
                    frame_packet=frame_packet,
                    raw_frame_bgr=frame_bgr,
                    preview_frame_bgr=preview_bgr,
                    robot_snapshot=robot_snapshot,
                    charuco_result=charuco_result,
                    board_pose_camera_board=board_pose_camera_board,
                    board_pose_board_camera=board_pose_board_camera,
                    next_sample_index=len(samples) + 1,
                )
                samples.append(sample_record)
                _write_samples_csv(session_dir / "samples.csv", samples)
                _write_sample_guidance_file(session_dir / "sampling_guidance.txt", samples)
                last_result = _maybe_update_extrinsic_result(
                    output_path=extrinsic_result_path,
                    samples=samples,
                    tool_name=tool_name,
                    wobj_name=wobj_name,
                    sdk_rpy_sequence=sdk_rpy_sequence,
                )
                logger.success(
                    "已记录样本 #{} visible={} charuco={} reproj={}",
                    sample_record.sample_index,
                    sample_record.board_visible,
                    sample_record.charuco_count,
                    _format_optional_float(sample_record.reprojection_error_px, digits=4),
                )
            if cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        client.close()
        cv2.destroyAllWindows()
        _write_samples_csv(session_dir / "samples.csv", samples)
        _write_sample_guidance_file(session_dir / "sampling_guidance.txt", samples)
        _maybe_update_extrinsic_result(
            output_path=extrinsic_result_path,
            samples=samples,
            tool_name=tool_name,
            wobj_name=wobj_name,
            sdk_rpy_sequence=sdk_rpy_sequence,
        )


# endregion


# region 机器人连接与拖动
def _connect_left_arm(robot_ip: str, tool_name: str, wobj_name: str) -> ConnectedLeftArm:
    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(robot_ip)
    robot_info = robot.robotInfo(ec)
    _print_sdk_result(f"robotInfo({robot_ip})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"读取左臂机器人信息失败: ip={robot_ip}")
    if str(robot_info.type) != EXPECTED_LEFT_ARM_TYPE:
        raise RuntimeError(
            "连接到的机器人不是左臂控制器: "
            f"expected={EXPECTED_LEFT_ARM_TYPE}, actual={robot_info.type}"
        )
    _apply_named_toolset(robot, ec, tool_name=tool_name, wobj_name=wobj_name)
    logger.success(f"左臂已连接: ip={robot_ip}, type={robot_info.type}, uid={robot_info.id}")
    return ConnectedLeftArm(
        robot_ip=robot_ip,
        robot=robot,
        robot_type=str(robot_info.type),
        robot_uid=str(robot_info.id),
        ec=ec,
    )


def _apply_named_toolset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    tool_name: str,
    wobj_name: str,
) -> None:
    robot.setToolset(tool_name, wobj_name, ec)
    _print_sdk_result(f"setToolset({tool_name}, {wobj_name})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"设置工具/工件坐标系失败: tool={tool_name}, wobj={wobj_name}")


def _enable_left_arm_drag(left_arm: ConnectedLeftArm) -> None:
    robot = left_arm.robot
    ec = left_arm.ec
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("切换非实时运动模式失败")
    robot.setPowerState(False, ec)
    _print_sdk_result("setPowerState(False)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前下电失败")
    if not _wait_for_power_off(robot, ec):
        raise RuntimeError("拖动前未在超时内确认下电")
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前切换手动模式失败")
    robot.moveReset(ec)
    _print_sdk_result("moveReset", ec)
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
    logger.success("左臂拖动已开启，程序启动后即可直接拖动示教。")


def _read_left_arm_snapshot(left_arm: ConnectedLeftArm, sdk_rpy_sequence: str) -> RobotSnapshot:
    robot = left_arm.robot
    ec = left_arm.ec
    joint_values_rad = [float(value) for value in robot.jointPos(ec)]
    _print_sdk_result("jointPos", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取左臂关节角失败")
    cartesian_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取左臂 TCP 位姿失败")
    joint_degrees = tuple(float(np.degrees(value)) for value in joint_values_rad)
    translation_mm = (
        float(cartesian_pose.trans[0]) * 1000.0,
        float(cartesian_pose.trans[1]) * 1000.0,
        float(cartesian_pose.trans[2]) * 1000.0,
    )
    rpy_degrees = (
        float(np.degrees(float(cartesian_pose.rpy[0]))),
        float(np.degrees(float(cartesian_pose.rpy[1]))),
        float(np.degrees(float(cartesian_pose.rpy[2]))),
    )
    tcp_pose_base_tool = _sdk_pose_to_transform(
        translation_mm=translation_mm,
        rpy_degrees=rpy_degrees,
        euler_sequence=sdk_rpy_sequence,
    )
    return RobotSnapshot(
        host_timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        joint_degrees=joint_degrees,
        tcp_pose_base_tool=tcp_pose_base_tool,
        tcp_translation_mm=translation_mm,
        tcp_rpy_degrees=rpy_degrees,
        sdk_pose_has_elbow=bool(cartesian_pose.hasElbow),
        sdk_pose_elbow_deg=float(np.degrees(float(cartesian_pose.elbow))),
        sdk_pose_conf_data=tuple(int(value) for value in list(cartesian_pose.confData)),
    )


def _sdk_pose_to_transform(
    translation_mm: tuple[float, float, float],
    rpy_degrees: tuple[float, float, float],
    euler_sequence: str,
) -> Transform:
    rotation_matrix = Rotation3D.from_euler(euler_sequence, list(rpy_degrees), degrees=True).as_matrix()
    return Transform(
        translation=Translation(*translation_mm),
        rotation=Quaternion.from_SO3(rotation_matrix),
    )


def _wait_for_power_off(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = 3.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        power_state = robot.powerState(ec)
        if power_state == xCoreSDK_python.PowerState.off:
            return True
        time.sleep(0.1)
    return False


def _shutdown_left_arm(left_arm: ConnectedLeftArm) -> None:
    robot = left_arm.robot
    ec = left_arm.ec
    try:
        robot.disableDrag(ec)
        _print_sdk_result("disableDrag", ec)
    except Exception:
        pass
    try:
        robot.stop(ec)
    except Exception:
        pass
    try:
        robot.setPowerState(False, ec)
    except Exception:
        pass
    try:
        robot.disconnectFromRobot(ec)
    except Exception:
        pass


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    code = ec.get("ec", 0)
    message = str(ec.get("message", ""))
    logger.debug(f"{action}: ec={code}, message={message}")


# endregion


# region 相机与 ChArUco
def _read_camera_calibration(client: CameraPipelineClient) -> CameraCalibration:
    response = client.get_camera_intrinsics(timeout_s=DEFAULT_CAMERA_TIMEOUT_S)
    distortion = np.asarray(response.distortion, dtype=np.float64).reshape(-1, 1)
    if distortion.size == 0:
        distortion = np.zeros((5, 1), dtype=np.float64)
    return CameraCalibration(
        camera_name=str(response.camera_name),
        width=int(response.width),
        height=int(response.height),
        camera_matrix=np.asarray(
            [
                [float(response.fx), 0.0, float(response.cx)],
                [0.0, float(response.fy), float(response.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        dist_coeffs=distortion,
    )


def _invert_transform(transform: Transform) -> np.ndarray:
    matrix = transform.as_SE3().astype(np.float64, copy=False)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -(rotation.T @ translation)
    return inverted


# endregion


# region 采样与保存
def _capture_sample(
    session_dir: Path,
    raw_frames_dir: Path,
    preview_frames_dir: Path,
    frame_packet: CameraColorFramePacket,
    raw_frame_bgr: np.ndarray,
    preview_frame_bgr: np.ndarray,
    robot_snapshot: RobotSnapshot,
    charuco_result,
    board_pose_camera_board: Transform | None,
    board_pose_board_camera: Transform | None,
    next_sample_index: int,
) -> SampleRecord:
    host_timestamp_s = time.time()
    sample_name = f"sample_{next_sample_index:03d}"
    raw_frame_path = raw_frames_dir / f"{sample_name}.png"
    preview_frame_path = preview_frames_dir / f"{sample_name}.png"
    cv2.imwrite(str(raw_frame_path), raw_frame_bgr)
    cv2.imwrite(str(preview_frame_path), preview_frame_bgr)
    return SampleRecord(
        sample_index=next_sample_index,
        host_timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        host_timestamp_s=host_timestamp_s,
        frame_id=int(frame_packet.frame_id),
        camera_timestamp_ms=float(frame_packet.timestamp_ms),
        board_visible=bool(charuco_result.board_visible),
        marker_count=int(charuco_result.marker_count),
        charuco_count=int(charuco_result.charuco_count),
        reprojection_error_px=None
        if charuco_result.reprojection_error_px is None
        else float(charuco_result.reprojection_error_px),
        robot_snapshot=robot_snapshot,
        board_pose_camera_board=board_pose_camera_board,
        board_pose_board_camera=board_pose_board_camera,
        raw_frame_path=raw_frame_path.relative_to(session_dir),
        preview_frame_path=preview_frame_path.relative_to(session_dir),
    )


def _write_samples_csv(csv_path: Path, samples: list[SampleRecord]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "host_timestamp_iso",
                "host_timestamp_s",
                "frame_id",
                "camera_timestamp_ms",
                "board_visible",
                "marker_count",
                "charuco_count",
                "reprojection_error_px",
                "robot_joint_j1_deg",
                "robot_joint_j2_deg",
                "robot_joint_j3_deg",
                "robot_joint_j4_deg",
                "robot_joint_j5_deg",
                "robot_joint_j6_deg",
                "robot_joint_j7_deg",
                "robot_tcp_x_mm",
                "robot_tcp_y_mm",
                "robot_tcp_z_mm",
                "robot_tcp_roll_deg",
                "robot_tcp_pitch_deg",
                "robot_tcp_yaw_deg",
                "robot_tcp_qw",
                "robot_tcp_qx",
                "robot_tcp_qy",
                "robot_tcp_qz",
                "robot_has_elbow",
                "robot_elbow_deg",
                "robot_conf_data",
                "camera_board_x_mm",
                "camera_board_y_mm",
                "camera_board_z_mm",
                "camera_board_qw",
                "camera_board_qx",
                "camera_board_qy",
                "camera_board_qz",
                "board_camera_x_mm",
                "board_camera_y_mm",
                "board_camera_z_mm",
                "board_camera_qw",
                "board_camera_qx",
                "board_camera_qy",
                "board_camera_qz",
                "raw_frame_path",
                "preview_frame_path",
            ]
        )
        for sample in samples:
            robot_transform = sample.robot_snapshot.tcp_pose_base_tool
            robot_translation = robot_transform.translation
            robot_rotation = robot_transform.rotation
            camera_board_fields = _transform_csv_fields(sample.board_pose_camera_board)
            board_camera_fields = _transform_csv_fields(sample.board_pose_board_camera)

            # 这里显式展开每一个字段，而不是把复杂对象直接序列化成一列字符串。
            # 这样做的目的有两个：
            # 1. 后续如果要在 Excel、Pandas 或其它标定调试脚本中筛选异常样本，可以直接按列过滤。
            # 2. 对手眼标定这种“采样一次代价高、复查频率高”的数据，列式展开最利于人工追查单位、
            #    方向和同步问题，避免后续再去解析嵌套字符串。
            writer.writerow(
                [
                    sample.sample_index,
                    sample.host_timestamp_iso,
                    f"{sample.host_timestamp_s:.6f}",
                    sample.frame_id,
                    f"{sample.camera_timestamp_ms:.3f}",
                    int(sample.board_visible),
                    sample.marker_count,
                    sample.charuco_count,
                    _format_optional_float(sample.reprojection_error_px, digits=6),
                    *[f"{value:.6f}" for value in sample.robot_snapshot.joint_degrees],
                    f"{robot_translation.x:.6f}",
                    f"{robot_translation.y:.6f}",
                    f"{robot_translation.z:.6f}",
                    f"{sample.robot_snapshot.tcp_rpy_degrees[0]:.6f}",
                    f"{sample.robot_snapshot.tcp_rpy_degrees[1]:.6f}",
                    f"{sample.robot_snapshot.tcp_rpy_degrees[2]:.6f}",
                    f"{robot_rotation.w:.8f}",
                    f"{robot_rotation.x:.8f}",
                    f"{robot_rotation.y:.8f}",
                    f"{robot_rotation.z:.8f}",
                    int(sample.robot_snapshot.sdk_pose_has_elbow),
                    f"{sample.robot_snapshot.sdk_pose_elbow_deg:.6f}",
                    str(list(sample.robot_snapshot.sdk_pose_conf_data)),
                    *camera_board_fields,
                    *board_camera_fields,
                    str(sample.raw_frame_path),
                    str(sample.preview_frame_path),
                ]
            )


def _write_extrinsic_result_file(
    output_path: Path,
    result: HandEyeCalibrationResult,
    valid_samples: list[SampleRecord],
    tool_name: str,
    wobj_name: str,
    sdk_rpy_sequence: str,
) -> None:
    transform = result.transform
    translation = transform.translation
    rotation = transform.rotation
    matrix = transform.as_SE3()
    lines = [
        "T_tool_camera 外参结果",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"tool_name={tool_name}",
        f"wobj_name={wobj_name}",
        f"sdk_rpy_sequence_assumption={sdk_rpy_sequence}",
        "group_a_semantics=T_base_tool",
        "group_b_semantics=T_board_camera",
        "result_semantics=T_tool_camera",
        f"valid_sample_count={len(valid_samples)}",
        f"rotation_rmse_deg={result.residual.rotation_rmse_deg:.6f}",
        f"rotation_max_deg={result.residual.rotation_max_deg:.6f}",
        f"translation_rmse_mm={result.residual.translation_rmse:.6f}",
        f"translation_max_mm={result.residual.translation_max:.6f}",
        "",
        f"tool_camera_x_mm={translation.x:.6f}",
        f"tool_camera_y_mm={translation.y:.6f}",
        f"tool_camera_z_mm={translation.z:.6f}",
        f"tool_camera_qw={rotation.w:.8f}",
        f"tool_camera_qx={rotation.x:.8f}",
        f"tool_camera_qy={rotation.y:.8f}",
        f"tool_camera_qz={rotation.z:.8f}",
        "",
        "tool_camera_matrix_se3=",
    ]
    for row in matrix:
        lines.append("  " + ", ".join(f"{float(value): .8f}" for value in row))
    lines.append("")
    lines.append("valid_sample_indices=" + ", ".join(str(sample.sample_index) for sample in valid_samples))

    # 外参结果单独保存成一份文本文件，目的是把“最终结论”与“原始采样流水账”拆开：
    # 1. `samples.csv` 更适合做逐样本排查、筛选和二次计算；
    # 2. 当前这个结果文件更适合直接给人看，或在部署时手工复制到配置中；
    # 3. 这里明确把 A/B 组位姿语义、Euler 假设、残差指标和最终矩阵放在同一个文件里，
    #    是为了避免几周后回看时只记得结果数字，却忘了它到底是怎么定义出来的。
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sample_guidance_file(output_path: Path, samples: list[SampleRecord]) -> None:
    guidance_lines = _build_sampling_guidance_lines(samples)
    coverage = _summarize_sampling_coverage(samples)
    lines = [
        "拖动采样建议",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"sample_count={len(samples)}",
    ]
    if coverage is not None:
        lines.extend(
            [
                f"span_x_mm={coverage.span_x_mm:.2f}",
                f"span_y_mm={coverage.span_y_mm:.2f}",
                f"span_z_mm={coverage.span_z_mm:.2f}",
                f"span_roll_deg={coverage.span_roll_deg:.2f}",
                f"span_pitch_deg={coverage.span_pitch_deg:.2f}",
                f"span_yaw_deg={coverage.span_yaw_deg:.2f}",
            ]
        )
    lines.append("")
    lines.extend(guidance_lines)

    # 这份建议文件不是算法输入，而是给现场采样者的“即时操作说明”。
    # 单独落盘有几个实际好处：
    # 1. 即使预览窗口已经关闭，现场人员仍然可以复盘当时为什么提示“多做俯仰”或“多做左右展开”；
    # 2. 如果一次采样中断，下一次重新打开目录时，可以快速知道上一次还缺什么激励；
    # 3. 对预研阶段来说，把采样建议固化下来也方便后续把这套经验迁移到 GUI 或自动流程中。
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_update_extrinsic_result(
    output_path: Path,
    samples: list[SampleRecord],
    tool_name: str,
    wobj_name: str,
    sdk_rpy_sequence: str,
) -> HandEyeCalibrationResult | None:
    valid_samples = [
        sample
        for sample in samples
        if sample.board_pose_board_camera is not None
    ]
    if len(valid_samples) < 3:
        return None
    robot_poses: list[PoseLike] = [sample.robot_snapshot.tcp_pose_base_tool for sample in valid_samples]
    board_poses: list[PoseLike] = [
        sample.board_pose_board_camera
        for sample in valid_samples
        if sample.board_pose_board_camera is not None
    ]
    result = calibrate_hand_eye_from_pose_sequences(
        group_a_poses=robot_poses,
        group_b_poses=board_poses,
        pair_mode="all",
    )
    _write_extrinsic_result_file(
        output_path=output_path,
        result=result,
        valid_samples=valid_samples,
        tool_name=tool_name,
        wobj_name=wobj_name,
        sdk_rpy_sequence=sdk_rpy_sequence,
    )
    return result


def _transform_csv_fields(transform: Transform | None) -> list[str]:
    if transform is None:
        return [""] * 7
    translation = transform.translation
    rotation = transform.rotation
    return [
        f"{translation.x:.6f}",
        f"{translation.y:.6f}",
        f"{translation.z:.6f}",
        f"{rotation.w:.8f}",
        f"{rotation.x:.8f}",
        f"{rotation.y:.8f}",
        f"{rotation.z:.8f}",
    ]


# endregion


# region 预览绘制
def _draw_preview(
    frame_bgr: np.ndarray,
    frame_packet: CameraColorFramePacket,
    robot_snapshot: RobotSnapshot,
    charuco_result,
    calibration_result: HandEyeCalibrationResult | None,
    camera_calibration: CameraCalibration,
    compute_ms: float,
    recorded_samples: list[SampleRecord],
    tool_name: str,
    wobj_name: str,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    _draw_marker_corners(canvas, charuco_result.marker_corners_px, charuco_result.marker_ids)
    if charuco_result.charuco_corners_px is not None:
        points = np.round(charuco_result.charuco_corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
        for point in charuco_result.charuco_corners_px:
            cv2.circle(canvas, (int(round(point[0])), int(round(point[1]))), 4, (0, 0, 255), -1, cv2.LINE_AA)
    if charuco_result.rvec is not None and charuco_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            camera_calibration.camera_matrix,
            camera_calibration.dist_coeffs,
            np.asarray(charuco_result.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(charuco_result.tvec, dtype=np.float64).reshape(3, 1),
            AXIS_DRAW_LENGTH_MM,
        )

    coverage = _summarize_sampling_coverage(recorded_samples)
    overlay_lines = _build_overlay_lines(
        frame_packet=frame_packet,
        robot_snapshot=robot_snapshot,
        charuco_result=charuco_result,
        calibration_result=calibration_result,
        coverage=coverage,
        sample_count=len(recorded_samples),
        compute_ms=compute_ms,
        tool_name=tool_name,
        wobj_name=wobj_name,
    )
    guidance_lines = _build_sampling_guidance_lines(recorded_samples)
    canvas = _draw_text_block(canvas, overlay_lines, (18, 28), color=(255, 255, 255), line_gap=22)
    canvas = _draw_text_block(canvas, guidance_lines, (18, 412), color=(80, 255, 255), line_gap=22)
    return canvas


def _build_overlay_lines(
    frame_packet: CameraColorFramePacket,
    robot_snapshot: RobotSnapshot,
    charuco_result,
    calibration_result: HandEyeCalibrationResult | None,
    coverage: SamplingCoverageSummary | None,
    sample_count: int,
    compute_ms: float,
    tool_name: str,
    wobj_name: str,
) -> list[str]:
    joint_text = ", ".join(f"J{index + 1}={value:.1f}" for index, value in enumerate(robot_snapshot.joint_degrees))
    tcp_transform = robot_snapshot.tcp_pose_base_tool
    lines = [
        f"camera_frame={int(frame_packet.frame_id)} camera_ts_ms={float(frame_packet.timestamp_ms):.1f} compute_ms={compute_ms:.2f}",
        f"drag=ON arm=left tool={tool_name} wobj={wobj_name}",
        f"board_visible={bool(charuco_result.board_visible)} marker={int(charuco_result.marker_count)} charuco={int(charuco_result.charuco_count)} reproj={_format_optional_float(charuco_result.reprojection_error_px, digits=4)}",
        f"robot_tcp_mm=({robot_snapshot.tcp_translation_mm[0]:.1f}, {robot_snapshot.tcp_translation_mm[1]:.1f}, {robot_snapshot.tcp_translation_mm[2]:.1f})",
        f"robot_rpy_deg=({robot_snapshot.tcp_rpy_degrees[0]:.1f}, {robot_snapshot.tcp_rpy_degrees[1]:.1f}, {robot_snapshot.tcp_rpy_degrees[2]:.1f})",
        f"robot_quat=({tcp_transform.rotation.w:.4f}, {tcp_transform.rotation.x:.4f}, {tcp_transform.rotation.y:.4f}, {tcp_transform.rotation.z:.4f})",
        f"robot_joints_deg={joint_text}",
        f"sample_count={sample_count} host_time={robot_snapshot.host_timestamp_iso}",
        "capture: Enter/Space/P    quit: Esc/Q",
    ]
    if coverage is not None:
        lines.append(
            "coverage_mm_deg="
            f"x:{coverage.span_x_mm:.1f} y:{coverage.span_y_mm:.1f} z:{coverage.span_z_mm:.1f} "
            f"roll:{coverage.span_roll_deg:.1f} pitch:{coverage.span_pitch_deg:.1f} yaw:{coverage.span_yaw_deg:.1f}"
        )
    if calibration_result is not None:
        tool_camera = calibration_result.transform
        lines.append(
            f"T_tool_camera_mm=({tool_camera.translation.x:.1f}, {tool_camera.translation.y:.1f}, {tool_camera.translation.z:.1f})"
        )
        lines.append(
            "residual="
            f"rot_rmse:{calibration_result.residual.rotation_rmse_deg:.3f}deg "
            f"trans_rmse:{calibration_result.residual.translation_rmse:.3f}mm"
        )
    return lines


def _draw_marker_corners(
    canvas: np.ndarray,
    marker_corners_px: list[np.ndarray],
    marker_ids: np.ndarray | None,
) -> None:
    if not marker_corners_px:
        return
    for marker_index, corners in enumerate(marker_corners_px):
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        points_i32 = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points_i32], True, (255, 255, 0), 2, cv2.LINE_AA)
        center = np.mean(points, axis=0)
        marker_label = "" if marker_ids is None or marker_index >= len(marker_ids) else str(int(marker_ids[marker_index]))
        _draw_single_text(canvas, f"M{marker_label}", (int(round(center[0])), int(round(center[1]))), (0, 255, 255))


def _draw_text_block(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    line_gap: int,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(DEFAULT_FONT_SIZE)
    x, y = origin
    for line in lines:
        _draw_stroked_text(
            draw,
            (x, y),
            line,
            font=font,
            fill=(int(color[2]), int(color[1]), int(color[0])),
            stroke_fill=(0, 0, 0),
            stroke_width=2,
        )
        y += line_gap
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_single_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    _draw_stroked_text(
        draw,
        origin,
        text,
        font=_load_font(18),
        fill=(int(color[2]), int(color[1]), int(color[0])),
        stroke_fill=(0, 0, 0),
        stroke_width=2,
    )
    canvas[:, :, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(DEFAULT_FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_stroked_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_fill: tuple[int, int, int],
    stroke_width: int,
) -> None:
    draw.text(
        position,
        text,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=stroke_width,
        anchor="la",
    )


# endregion


# region 采样建议
def _summarize_sampling_coverage(samples: list[SampleRecord]) -> SamplingCoverageSummary | None:
    valid_samples = [sample for sample in samples if sample.board_pose_board_camera is not None]
    if not valid_samples:
        return None
    translations = np.asarray(
        [sample.robot_snapshot.tcp_translation_mm for sample in valid_samples],
        dtype=np.float64,
    )
    rpy_values = np.asarray(
        [sample.robot_snapshot.tcp_rpy_degrees for sample in valid_samples],
        dtype=np.float64,
    )
    spans_translation = np.ptp(translations, axis=0)
    spans_rpy = np.ptp(rpy_values, axis=0)
    return SamplingCoverageSummary(
        sample_count=len(valid_samples),
        span_x_mm=float(spans_translation[0]),
        span_y_mm=float(spans_translation[1]),
        span_z_mm=float(spans_translation[2]),
        span_roll_deg=float(spans_rpy[0]),
        span_pitch_deg=float(spans_rpy[1]),
        span_yaw_deg=float(spans_rpy[2]),
    )


def _build_sampling_guidance_lines(samples: list[SampleRecord]) -> list[str]:
    coverage = _summarize_sampling_coverage(samples)
    lines = ["sampling_guidance:"]
    if coverage is None:
        lines.extend(
            [
                "1. 先在板清晰可见的位置记录第 1 帧，作为基准位。",
                "2. 然后依次做左右、前后、上下和翻腕动作，每次停稳后再采样。",
                "3. 保持左臂拖动时标定板始终在画面内，避免只做平移不做旋转。",
            ]
        )
        return lines

    pending_lines: list[str] = []
    if coverage.span_x_mm < TRANSLATION_SPAN_TARGETS_MM["x"]:
        pending_lines.append(
            f"补左右横向展开，当前 X 跨度 {coverage.span_x_mm:.1f} mm，建议至少 {TRANSLATION_SPAN_TARGETS_MM['x']:.1f} mm。"
        )
    if coverage.span_y_mm < TRANSLATION_SPAN_TARGETS_MM["y"]:
        pending_lines.append(
            f"补前后距离变化，当前 Y 跨度 {coverage.span_y_mm:.1f} mm，建议至少 {TRANSLATION_SPAN_TARGETS_MM['y']:.1f} mm。"
        )
    if coverage.span_z_mm < TRANSLATION_SPAN_TARGETS_MM["z"]:
        pending_lines.append(
            f"补上下高度变化，当前 Z 跨度 {coverage.span_z_mm:.1f} mm，建议至少 {TRANSLATION_SPAN_TARGETS_MM['z']:.1f} mm。"
        )
    if coverage.span_roll_deg < ROTATION_SPAN_TARGETS_DEG["roll"]:
        pending_lines.append(
            f"补 roll 翻腕，当前 roll 跨度 {coverage.span_roll_deg:.1f} deg，建议至少 {ROTATION_SPAN_TARGETS_DEG['roll']:.1f} deg。"
        )
    if coverage.span_pitch_deg < ROTATION_SPAN_TARGETS_DEG["pitch"]:
        pending_lines.append(
            f"补 pitch 抬头/低头，当前 pitch 跨度 {coverage.span_pitch_deg:.1f} deg，建议至少 {ROTATION_SPAN_TARGETS_DEG['pitch']:.1f} deg。"
        )
    if coverage.span_yaw_deg < ROTATION_SPAN_TARGETS_DEG["yaw"]:
        pending_lines.append(
            f"补 yaw 左右摆头，当前 yaw 跨度 {coverage.span_yaw_deg:.1f} deg，建议至少 {ROTATION_SPAN_TARGETS_DEG['yaw']:.1f} deg。"
        )

    if not pending_lines:
        lines.extend(
            [
                "采样覆盖已经比较充分，可以补充 3-5 个斜向组合姿态做收尾。",
                "优先补“边平移边转动”的组合位，避免所有样本都围绕同一个姿态小范围抖动。",
                "若残差仍偏大，优先重采重投影误差高或姿态过于相似的样本。",
            ]
        )
        return lines

    lines.extend(f"{index + 1}. {text}" for index, text in enumerate(pending_lines[:3]))
    lines.append("停稳后再按采样键，优先让板保持清晰、无遮挡、非纯正视。")
    return lines


# endregion


# region 通用工具
def _create_session_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _format_optional_float(value: float | None, digits: int) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="左臂拖动示教 + Orin 左手相机流手眼标定实采页")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_ORIN_SERVICE_ADDR, help="Orin camera_pipeline_service 地址")
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME, help="逻辑相机名，默认使用左手相机")
    parser.add_argument("--left-arm-ip", type=str, default=DEFAULT_LEFT_ARM_IP, help="左臂控制器 IP")
    parser.add_argument("--tool-name", type=str, default=DEFAULT_TOOL_NAME, help="SDK 使用的工具坐标系名")
    parser.add_argument("--wobj-name", type=str, default=DEFAULT_WOBJ_NAME, help="SDK 使用的工件坐标系名")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="运行输出根目录")
    parser.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS, help="进入位姿估计所需的最小 ChArUco 角点数")
    parser.add_argument(
        "--sdk-rpy-sequence",
        type=str,
        default=DEFAULT_SDK_RPY_SEQUENCE,
        help="把 SDK 返回的 rpy 转为旋转矩阵时采用的 Euler 序列，默认 xyz",
    )
    return parser.parse_args()


def _validate_runtime_requirements() -> None:
    _ = cv2.aruco.ArucoDetector
    _ = cv2.aruco.CharucoBoard
    _ = Rotation3D.from_euler


# endregion


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
        raise
