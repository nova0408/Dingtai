from __future__ import annotations

import argparse
import ast
import csv
import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from test.wuji.common import (  # noqa: E402
    DEFAULT_PORT,
    GRIPPER_PORT,
    SshTunnelGroup,
    close_wuyou_channel,
    create_wuyou_channel,
    stop_ssh_process,
)
from test.wuji.xcoresdk_arm_cli_test import (  # noqa: E402
    DEFAULT_CARTESIAN_SPEED,
    DEFAULT_CARTESIAN_ZONE,
    DEFAULT_JOINT_SPEED,
    DEFAULT_JOINT_ZONE,
    DEFAULT_TOOL_NAME,
    DEFAULT_WOBJ_NAME,
    LEFT_ARM_IP,
    ConnectedArm,
    DahuanGripperClient,
    WujiBodyClient,
    _apply_named_toolset,
    _copy_cartesian_pose_context,
    _detect_arm_side,
    _ensure_nrt_motion_ready,
    _format_sequence,
    _mm_to_m,
    _m_to_mm,
    _deg_to_rad,
    _poll_gripper_until_idle,
    _print_sdk_result,
    _rad_to_deg,
    _shutdown_robot,
    _validate_cartesian_target,
    _wait_lift_until_near_target,
    _wait_until_idle,
)

DEFAULT_RECORD_DIR = PROJECT_ROOT / "record_left"
"默认左臂拖动示教 CSV 目录。"

DEFAULT_ARM_SIDE = "left"
"当前脚本固定回放左臂记录。"

DEFAULT_MAX_FILES: int | None = None
"默认加载的 CSV 文件数量；`None` 表示全部。"

DEFAULT_CARTESIAN_MOTION_MODE = "movej"
"笛卡尔目标执行方式默认值，仅作为 CLI 缺省；交互入口仍会展示并确认。"

CSV_CARTESIAN_OFFSET_TARGETS = [3, 4]
"需要应用全局笛卡尔纠偏的 CSV 序号列表。"

CSV_CARTESIAN_OFFSET_CALCULATE_AT = 2
"在该 CSV 的最后一个 arm pose 处计算一次全局笛卡尔纠偏。"


# region 数据结构


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """单条 CSV 回放记录。"""

    csv_name: str
    row_index: int
    action_type: str
    joints_text: str
    pose_text: str


@dataclass(frozen=True, slots=True)
class ParsedArmPose:
    """CSV 中单条笛卡尔目标。"""

    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    has_elbow: bool
    elbow_deg: float
    conf_data: tuple[int, ...]


@dataclass(slots=True)
class ReplayRuntime:
    """回放执行期上下文。"""

    connected_arm: ConnectedArm
    gripper_process: SshTunnelGroup
    gripper_channel: object
    gripper: DahuanGripperClient
    body_process: SshTunnelGroup
    body_channel: object
    body: WujiBodyClient
    global_cartesian_offset: tuple[tuple[float, float, float, float], ...] | None = None


# endregion


# region CSV 解析


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay left-arm drag-record CSV files in record_left.")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--cartesian-motion", choices=("movej", "movel"), default=DEFAULT_CARTESIAN_MOTION_MODE)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--auto-start", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def _discover_csv_paths(record_dir: Path, max_files: int | None) -> list[Path]:
    if not record_dir.is_dir():
        raise FileNotFoundError(f"CSV 目录不存在: {record_dir}")
    csv_paths = sorted(path for path in record_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    if max_files is not None:
        return csv_paths[:max_files]
    return csv_paths


def _load_replay_rows(csv_path: Path) -> list[ReplayRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[ReplayRow] = []
        for row_index, row in enumerate(reader, start=1):
            action_type = str(row.get("type", "")).strip().lower()
            joints_text = str(row.get("joints", "")).strip()
            pose_text = str(row.get("pose", "")).strip()
            if action_type == "":
                raise ValueError(f"CSV 缺少 type: file={csv_path}, row={row_index}")
            rows.append(
                ReplayRow(
                    csv_name=csv_path.name,
                    row_index=row_index,
                    action_type=action_type,
                    joints_text=joints_text,
                    pose_text=pose_text,
                )
            )
    if not rows:
        raise ValueError(f"CSV 没有可执行数据: {csv_path}")
    return rows


def _parse_joint_values(joints_text: str, expected_len: int = 7) -> list[float]:
    if joints_text.strip().lower() == "nan":
        raise ValueError("关节列为 NaN，不能解析为关节目标")
    parsed = ast.literal_eval(joints_text)
    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise ValueError(f"关节列长度无效: {joints_text}")
    return [float(value) for value in parsed]


def _parse_pose_values(pose_text: str) -> ParsedArmPose:
    if pose_text.strip().lower() == "nan":
        raise ValueError("pose 列为 NaN，不能解析为笛卡尔目标")
    parsed = ast.literal_eval(pose_text)
    if not isinstance(parsed, list) or len(parsed) != 9:
        raise ValueError(f"pose 列格式无效: {pose_text}")
    xyz_mm = (float(parsed[0]), float(parsed[1]), float(parsed[2]))
    rpy_deg = (float(parsed[3]), float(parsed[4]), float(parsed[5]))
    has_elbow = bool(parsed[6])
    elbow_deg = float(parsed[7])
    conf_data_raw = parsed[8]
    if not isinstance(conf_data_raw, list):
        raise ValueError(f"pose confData 格式无效: {pose_text}")
    conf_data = tuple(int(value) for value in conf_data_raw)
    return ParsedArmPose(
        xyz_mm=xyz_mm,
        rpy_deg=rpy_deg,
        has_elbow=has_elbow,
        elbow_deg=elbow_deg,
        conf_data=conf_data,
    )


def _extract_csv_sequence(csv_name: str) -> int:
    prefix = csv_name.split("_", maxsplit=1)[0]
    return int(prefix)


# endregion


# region 连接与执行


def _connect_left_arm() -> ConnectedArm:
    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(LEFT_ARM_IP)
    robot_info = robot.robotInfo(ec)
    _print_sdk_result(f"robotInfo({LEFT_ARM_IP})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"读取左臂机器人信息失败: ip={LEFT_ARM_IP}")
    if _apply_named_toolset(robot, ec) is None:
        raise RuntimeError(
            f"设置默认工具/工件失败: ip={LEFT_ARM_IP}, tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}"
        )
    arm_side = _detect_arm_side(robot_info.type)
    if arm_side != DEFAULT_ARM_SIDE:
        raise RuntimeError(f"连接到的机械臂不是左臂: ip={LEFT_ARM_IP}, actual={arm_side}")
    logger.success(
        "已连接左臂 ip={} type={} uid={} tool={} wobj={}",
        LEFT_ARM_IP,
        robot_info.type,
        robot_info.id,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    return ConnectedArm(
        arm_side=arm_side,
        robot_ip=LEFT_ARM_IP,
        robot=robot,
        robot_type=robot_info.type,
        robot_uid=robot_info.id,
        ec=ec,
    )


def _create_runtime() -> ReplayRuntime:
    connected_arm = _connect_left_arm()
    gripper_process, gripper_channel = create_wuyou_channel(GRIPPER_PORT)
    body_process, body_channel = create_wuyou_channel(DEFAULT_PORT)
    return ReplayRuntime(
        connected_arm=connected_arm,
        gripper_process=gripper_process,
        gripper_channel=gripper_channel,
        gripper=DahuanGripperClient(gripper_channel),
        body_process=body_process,
        body_channel=body_channel,
        body=WujiBodyClient(body_channel),
    )


def _prepare_runtime(runtime: ReplayRuntime) -> None:
    if not _ensure_nrt_motion_ready(runtime.connected_arm.robot, runtime.connected_arm.ec):
        raise RuntimeError("左臂未准备到可执行回放的 NRT 状态")
    runtime.body.lift.set_enable(True)
    logger.info("已确认机械臂基坐标采用 tool={} wobj={}", DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME)


def _execute_joint_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    target_joint_deg = _parse_joint_values(row.joints_text)
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    cmd_id = xCoreSDK_python.PyString()
    target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(target_joint_deg))
    robot.moveReset(ec)
    _print_sdk_result("moveReset(replay-joint)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放关节 moveReset 失败")
    robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, DEFAULT_JOINT_SPEED, DEFAULT_JOINT_ZONE)], cmd_id, ec)
    _print_sdk_result("moveAppend(MoveAbsJCommand)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放关节 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(replay-joint)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放关节 moveStart 失败")
    logger.info(
        "已下发关节运动 file={} row={} joints(deg)=[{}] cmd_id={}",
        row.csv_name,
        row.row_index,
        _format_sequence(target_joint_deg),
        cmd_id.content(),
    )
    if not _wait_until_idle(robot, ec, "等待回放关节运动"):
        raise TimeoutError("回放关节运动等待超时")


def _build_cartesian_target(runtime: ReplayRuntime, row: ReplayRow) -> xCoreSDK_python.CartesianPosition:
    parsed_pose = _parse_pose_values(row.pose_text)
    target_xyz_mm = list(parsed_pose.xyz_mm)
    target_rpy_deg = list(parsed_pose.rpy_deg)
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    current_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前笛卡尔位姿失败")
    target_pose = xCoreSDK_python.CartesianPosition(_mm_to_m(target_xyz_mm) + _deg_to_rad(target_rpy_deg))
    _copy_cartesian_pose_context(current_pose, target_pose)
    target_pose.hasElbow = parsed_pose.has_elbow
    target_pose.elbow = _deg_to_rad([parsed_pose.elbow_deg])[0]
    target_pose.confData = list(parsed_pose.conf_data)
    csv_sequence = _extract_csv_sequence(row.csv_name)
    if csv_sequence in CSV_CARTESIAN_OFFSET_TARGETS:
        target_pose = _apply_global_cartesian_offset(runtime, row, target_pose)
    return target_pose


def _frame_to_homogeneous_matrix(frame: xCoreSDK_python.CartesianPosition) -> list[list[float]]:
    rx, ry, rz = (float(value) for value in frame.rpy)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    matrix = [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(frame.trans[0])],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(frame.trans[1])],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(frame.trans[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix


def _multiply_homogeneous_matrix(
    left: tuple[tuple[float, float, float, float], ...],
    right: list[list[float]],
) -> list[list[float]]:
    result = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for row_index in range(4):
        for col_index in range(4):
            result[row_index][col_index] = sum(
                float(left[row_index][term_index]) * float(right[term_index][col_index])
                for term_index in range(4)
            )
    return result


def _homogeneous_matrix_to_rpy(matrix: list[list[float]]) -> tuple[float, float, float]:
    sy = -float(matrix[2][0])
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-9:
        rx = math.atan2(float(matrix[2][1]), float(matrix[2][2]))
        ry = math.atan2(sy, cy)
        rz = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
        return rx, ry, rz
    rx = math.atan2(-float(matrix[1][2]), float(matrix[1][1]))
    ry = math.atan2(sy, cy)
    rz = 0.0
    return rx, ry, rz


def _homogeneous_matrix_to_cartesian_position(
    source_pose: xCoreSDK_python.CartesianPosition,
    matrix: list[list[float]],
) -> xCoreSDK_python.CartesianPosition:
    xyz_m = [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
    rpy_rad = list(_homogeneous_matrix_to_rpy(matrix))
    target_pose = xCoreSDK_python.CartesianPosition(xyz_m + rpy_rad)
    target_pose.hasElbow = source_pose.hasElbow
    target_pose.elbow = source_pose.elbow
    target_pose.confData = list(source_pose.confData)
    return target_pose


def _apply_global_cartesian_offset(
    runtime: ReplayRuntime,
    row: ReplayRow,
    target_pose: xCoreSDK_python.CartesianPosition,
) -> xCoreSDK_python.CartesianPosition:
    if runtime.global_cartesian_offset is None:
        raise RuntimeError(
            f"CSV {row.csv_name} 需要使用全局笛卡尔纠偏，但当前尚未在 "
            f"{CSV_CARTESIAN_OFFSET_CALCULATE_AT:02d}_*.csv 末尾计算 offset"
        )
    original_matrix = _frame_to_homogeneous_matrix(target_pose)
    corrected_matrix = _multiply_homogeneous_matrix(runtime.global_cartesian_offset, original_matrix)
    corrected_pose = _homogeneous_matrix_to_cartesian_position(target_pose, corrected_matrix)
    logger.info(
        "已对笛卡尔目标应用全局左乘纠偏 file={} row={} base=tool:{} wobj:{}",
        row.csv_name,
        row.row_index,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    return corrected_pose


def _find_last_arm_pose_row(rows: list[ReplayRow]) -> ReplayRow | None:
    for row in reversed(rows):
        if row.action_type == "arm" and row.pose_text.strip().lower() != "nan":
            return row
    return None


def _calculate_global_cartesian_offset(
    runtime: ReplayRuntime,
    csv_path: Path,
    reference_row: ReplayRow,
) -> tuple[tuple[float, float, float, float], ...]:
    logger.warning(
        "全局 offset 计算暂未实现；当前先记录占位逻辑。file={} row={} base=tool:{} wobj:{}",
        csv_path.name,
        reference_row.row_index,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _execute_cartesian_row(
    runtime: ReplayRuntime,
    row: ReplayRow,
    cartesian_motion: Literal["movej", "movel"],
) -> None:
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    target_pose = _build_cartesian_target(runtime, row)
    cmd_id = xCoreSDK_python.PyString()
    if cartesian_motion == "movel" and not _validate_cartesian_target(robot, ec, target_pose):
        raise RuntimeError("回放笛卡尔目标未通过 checkPath 校验")
    robot.moveReset(ec)
    _print_sdk_result("moveReset(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔 moveReset 失败")
    if cartesian_motion == "movel":
        robot.moveAppend(
            [xCoreSDK_python.MoveLCommand(target_pose, DEFAULT_CARTESIAN_SPEED, DEFAULT_CARTESIAN_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveL)", ec)
    else:
        robot.moveAppend(
            [xCoreSDK_python.MoveJCommand(target_pose, DEFAULT_CARTESIAN_SPEED, DEFAULT_CARTESIAN_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveJ)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔 moveStart 失败")
    logger.info(
        "已下发笛卡尔运动 file={} row={} motion={} xyz(mm)=[{}] rpy(deg)=[{}] cmd_id={}",
        row.csv_name,
        row.row_index,
        cartesian_motion,
        _format_sequence(_m_to_mm(target_pose.trans)),
        _format_sequence(_rad_to_deg(target_pose.rpy)),
        cmd_id.content(),
    )
    if not _wait_until_idle(robot, ec, "等待回放笛卡尔运动"):
        raise TimeoutError("回放笛卡尔运动等待超时")


def _execute_gripper_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    target_value = int(round(float(row.pose_text)))
    if not runtime.gripper.set_pos(target_value):
        raise RuntimeError("夹爪 set_pos 下发失败")
    _poll_gripper_until_idle(runtime.gripper, target_value)
    logger.info("已下发夹爪目标 file={} row={} pos={}", row.csv_name, row.row_index, target_value)
    deadline_hint = runtime.gripper.get_status()
    logger.info("夹爪当前状态 pos={} calibrated={}", deadline_hint.position, bool(deadline_hint.calibrated))


def _execute_lift_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    target_height_mm = int(round(float(row.pose_text)))
    if target_height_mm < 0:
        raise ValueError(f"lift 目标高度非法: {target_height_mm}")
    runtime.body.lift.set_lift_physical_height(target_height_mm)
    actual_height_mm = _wait_lift_until_near_target(runtime.body, target_height_mm)
    logger.info(
        "lift 已执行 file={} row={} target={} mm actual={:.1f} mm",
        row.csv_name,
        row.row_index,
        target_height_mm,
        actual_height_mm,
    )


def _execute_row(
    runtime: ReplayRuntime,
    row: ReplayRow,
    cartesian_motion: Literal["movej", "movel"],
) -> None:
    logger.info(
        "开始执行 file={} row={} type={} joints={} pose={}",
        row.csv_name,
        row.row_index,
        row.action_type,
        row.joints_text,
        row.pose_text,
    )
    if row.action_type == "arm":
        if row.pose_text.strip().lower() == "nan":
            _execute_joint_row(runtime, row)
            return
        _execute_cartesian_row(runtime, row, cartesian_motion)
        return
    if row.action_type == "gripper":
        _execute_gripper_row(runtime, row)
        return
    if row.action_type == "lift":
        _execute_lift_row(runtime, row)
        return
    raise ValueError(f"当前脚本暂不支持的记录类型: {row.action_type}")


def _cleanup_runtime(runtime: ReplayRuntime | None) -> None:
    if runtime is None:
        return
    try:
        _shutdown_robot(runtime.connected_arm.robot, runtime.connected_arm.ec)
    finally:
        preserved_gripper_process = runtime.gripper_process
        preserved_body_process = runtime.body_process
        close_wuyou_channel(runtime.gripper_channel)
        close_wuyou_channel(runtime.body_channel)
        stop_ssh_process(preserved_gripper_process)
        stop_ssh_process(preserved_body_process)
        del runtime
        gc.collect()


# endregion


# region 交互流程


def _prompt_cartesian_motion(default_motion: Literal["movej", "movel"]) -> Literal["movej", "movel"]:
    print("请选择笛卡尔记录的执行方式：")
    print("  1. MoveJ 到笛卡尔目标")
    print("  2. MoveL 到笛卡尔目标")
    print(f"直接回车采用默认值: {default_motion}")
    raw_text = input("请选择: ").strip().lower()
    if raw_text == "":
        return default_motion
    if raw_text == "1":
        return "movej"
    if raw_text == "2":
        return "movel"
    raise ValueError("无效的笛卡尔执行方式选择")


def _print_csv_summary(csv_paths: list[Path]) -> None:
    print("本次将按以下顺序执行 CSV：")
    for index, csv_path in enumerate(csv_paths, start=1):
        print(f"  {index:02d}. {csv_path.name}")
    print(
        "全局笛卡尔纠偏配置: "
        f"calculate_at={CSV_CARTESIAN_OFFSET_CALCULATE_AT:02d}, "
        f"targets={[f'{value:02d}' for value in CSV_CARTESIAN_OFFSET_TARGETS]}"
    )


def _confirm_start(csv_paths: list[Path], cartesian_motion: str, auto_start: bool) -> bool:
    _print_csv_summary(csv_paths)
    print(f"左臂基坐标固定为 tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    print(f"笛卡尔记录执行方式: {cartesian_motion}")
    if auto_start:
        return True
    raw_text = input("输入回车开始，输入 q 退出: ").strip().lower()
    return raw_text != "q"


def _confirm_each_file(csv_path: Path) -> str:
    print("")
    print(f"准备执行 {csv_path.name}")
    return input("输入回车执行该文件，输入 s 跳过，输入 q 终止: ").strip().lower()


def main(
    record_dir: Path = DEFAULT_RECORD_DIR,
    cartesian_motion: Literal["movej", "movel"] = DEFAULT_CARTESIAN_MOTION_MODE,
    max_files: int | None = DEFAULT_MAX_FILES,
    auto_start: bool = False,
) -> int:
    logger.info("左臂拖动示教自动回放 CLI 启动")
    csv_paths = _discover_csv_paths(Path(record_dir), max_files)
    if not csv_paths:
        raise RuntimeError(f"没有在目录中发现 CSV: {record_dir}")
    selected_motion = cartesian_motion if auto_start else _prompt_cartesian_motion(cartesian_motion)
    if not _confirm_start(csv_paths, selected_motion, auto_start):
        logger.info("用户取消执行")
        return 0

    runtime: ReplayRuntime | None = None
    try:
        runtime = _create_runtime()
        _prepare_runtime(runtime)
        for csv_path in csv_paths:
            if not auto_start:
                file_choice = _confirm_each_file(csv_path)
                if file_choice == "q":
                    logger.warning("用户终止执行")
                    return 0
                if file_choice == "s":
                    logger.warning("跳过文件 {}", csv_path.name)
                    continue
            rows = _load_replay_rows(csv_path)
            logger.info("开始执行文件 {}，共 {} 行", csv_path.name, len(rows))
            for row in rows:
                _execute_row(runtime, row, selected_motion)
            csv_sequence = _extract_csv_sequence(csv_path.name)
            if csv_sequence == CSV_CARTESIAN_OFFSET_CALCULATE_AT:
                reference_row = _find_last_arm_pose_row(rows)
                if reference_row is None:
                    raise RuntimeError(f"无法在 {csv_path.name} 中找到用于计算全局 offset 的最后一个 arm pose")
                runtime.global_cartesian_offset = _calculate_global_cartesian_offset(runtime, csv_path, reference_row)
                logger.success("已更新全局笛卡尔纠偏矩阵，后续目标 CSV 将按左乘方式应用")
            logger.success("文件执行完成 {}", csv_path.name)
        logger.success("全部 CSV 执行完成")
        return 0
    finally:
        _cleanup_runtime(runtime)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            record_dir=Path(args.record_dir),
            cartesian_motion=args.cartesian_motion,
            max_files=args.max_files,
            auto_start=bool(args.auto_start),
        )
    )
