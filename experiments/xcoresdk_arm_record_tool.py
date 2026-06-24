#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk.xcoresdk import xCoreSDK_python  # noqa: E402

LOCAL_IP = "192.168.1.116"
EXPECTED_ARM_TYPES = {
    "left": "AR5-5_0.8L-W4C1C9-ZY2",
    "right": "AR5-5_0.8R-W4C1C9-ZY2",
}


@dataclass(frozen=True, slots=True)
class RobotConnectionConfig:
    """机器人连接配置。"""

    robot_ip: str
    local_ip: str | None


@dataclass(slots=True)
class ConnectedArm:
    """单台已连接机械臂的运行上下文。"""

    arm_side: str
    config: RobotConnectionConfig
    robot: xCoreSDK_python.xMateErProRobot
    robot_type: str
    robot_uid: str
    ec: dict[str, object]


def _format_sequence(values: list[float] | tuple[float, ...], decimals: int = 2) -> str:
    return ", ".join(f"{float(value):.{decimals}f}" for value in values)


def _mm_to_m(values_mm: list[float]) -> list[float]:
    return [float(value) / 1000.0 for value in values_mm]


def _rad_to_deg(values_rad: list[float] | tuple[float, ...]) -> list[float]:
    return [math.degrees(float(value)) for value in values_rad]


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    message = str(ec.get("message", ""))
    code = ec.get("ec", 0)
    print(f"{action}: ec={code}, message={message}")


def _connect_robot(config: RobotConnectionConfig) -> xCoreSDK_python.xMateErProRobot:
    if config.local_ip:
        return xCoreSDK_python.xMateErProRobot(config.robot_ip, config.local_ip)
    return xCoreSDK_python.xMateErProRobot(config.robot_ip)


def _default_connection_configs() -> list[RobotConnectionConfig]:
    return [
        RobotConnectionConfig(robot_ip="192.168.1.161", local_ip=LOCAL_IP),
        RobotConnectionConfig(robot_ip="192.168.1.160", local_ip=LOCAL_IP),
    ]


def _detect_arm_side(robot_type: str) -> str:
    for arm_side, expected_robot_type in EXPECTED_ARM_TYPES.items():
        if robot_type == expected_robot_type:
            return arm_side
    raise ValueError(f"未识别的机器人型号: {robot_type}")


def _prepare_robot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    robot.setPowerState(False, ec)


def _shutdown_robot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    try:
        robot.stop(ec)
    except Exception:
        pass
    try:
        robot.disableDrag(ec)
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


def _connect_arms(configs: list[RobotConnectionConfig]) -> dict[str, ConnectedArm]:
    connected_arms: dict[str, ConnectedArm] = {}
    try:
        for config in configs:
            ec: dict[str, object] = {}
            robot = _connect_robot(config)
            robot_info = robot.robotInfo(ec)
            _print_sdk_result(f"robotInfo({config.robot_ip})", ec)
            if ec.get("ec", 0) != 0:
                raise RuntimeError(f"读取机器人信息失败: ip={config.robot_ip}")
            arm_side = _detect_arm_side(robot_info.type)
            connected_arm = ConnectedArm(
                arm_side=arm_side,
                config=config,
                robot=robot,
                robot_type=robot_info.type,
                robot_uid=robot_info.id,
                ec=ec,
            )
            _prepare_robot(robot, ec)
            connected_arms[arm_side] = connected_arm
            print(f"已连接 {arm_side} arm: ip={config.robot_ip}, type={robot_info.type}, uid={robot_info.id}")
        missing_arm_sides = [arm_side for arm_side in EXPECTED_ARM_TYPES if arm_side not in connected_arms]
        if missing_arm_sides:
            raise RuntimeError(f"缺少目标机械臂连接: {', '.join(missing_arm_sides)}")
        return connected_arms
    except Exception:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)
        raise


def _select_active_arm() -> str:
    print("主菜单:")
    print("  1. left")
    print("  2. right")
    print("  q. 退出")
    choice = input("请选择机械臂: ").strip().lower()
    if choice == "1":
        return "left"
    if choice == "2":
        return "right"
    if choice == "q":
        raise SystemExit(0)
    print("无效选择，请重新输入")
    return _select_active_arm()


def _format_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _build_record(
    connected_arm: ConnectedArm,
    joint_values_rad: list[float] | tuple[float, ...],
    cart_pose: xCoreSDK_python.CartesianPosition,
) -> dict[str, str]:
    joint_values_deg = _rad_to_deg(joint_values_rad)
    record: dict[str, str] = {
        "timestamp": _format_timestamp(),
        "arm_side": connected_arm.arm_side,
        "robot_ip": connected_arm.config.robot_ip,
        "robot_type": connected_arm.robot_type,
        "robot_uid": connected_arm.robot_uid,
    }
    for index, value in enumerate(joint_values_deg, start=1):
        record[f"joint_{index}_deg"] = f"{value:.6f}"
    record.update(
        {
            "x_mm": f"{float(cart_pose.trans[0]) * 1000.0:.6f}",
            "y_mm": f"{float(cart_pose.trans[1]) * 1000.0:.6f}",
            "z_mm": f"{float(cart_pose.trans[2]) * 1000.0:.6f}",
            "rx_deg": f"{math.degrees(float(cart_pose.rpy[0])):.6f}",
            "ry_deg": f"{math.degrees(float(cart_pose.rpy[1])):.6f}",
            "rz_deg": f"{math.degrees(float(cart_pose.rpy[2])):.6f}",
            "has_elbow": str(bool(cart_pose.hasElbow)),
            "elbow_deg": f"{math.degrees(float(cart_pose.elbow)):.6f}",
            "conf_data": str(list(cart_pose.confData)),
        }
    )
    return record


def _print_current_snapshot(connected_arm: ConnectedArm) -> tuple[list[float], xCoreSDK_python.CartesianPosition]:
    robot = connected_arm.robot
    ec = connected_arm.ec
    joint_values = list(robot.jointPos(ec))
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("jointPos", ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    print(f"当前机械臂: {connected_arm.arm_side} (ip={connected_arm.config.robot_ip}, type={connected_arm.robot_type})")
    print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
    print(
        "当前笛卡尔位姿(mm/deg): "
        f"trans={_format_sequence([float(value) * 1000.0 for value in cart_pose.trans])} "
        f"rpy={_format_sequence(_rad_to_deg(list(cart_pose.rpy)))}"
    )
    print(
        "当前笛卡尔上下文: "
        f"hasElbow={cart_pose.hasElbow}, "
        f"elbow(deg)={math.degrees(float(cart_pose.elbow)):.2f}, "
        f"confData={list(cart_pose.confData)}"
    )
    return joint_values, cart_pose


def _record_loop(connected_arm: ConnectedArm) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    while True:
        print()
        print("直接回车记录当前关节与位姿，输入 q 结束并保存。")
        raw_text = input("请输入: ").strip().lower()
        if raw_text == "q":
            return records
        if raw_text != "":
            print("无效输入，请直接回车或输入 q")
            continue
        joint_values, cart_pose = _print_current_snapshot(connected_arm)
        record = _build_record(connected_arm, joint_values, cart_pose)
        records.append(record)
        print(f"已记录第 {len(records)} 条")


def _write_csv(records: list[dict[str, str]], arm_side: str) -> Path | None:
    if not records:
        return None
    csv_path = Path.cwd() / f"xcoresdk_arm_records_{arm_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def main() -> int:
    connected_arms = _connect_arms(_default_connection_configs())
    selected_arm_side = _select_active_arm()
    connected_arm = connected_arms[selected_arm_side]
    try:
        print(f"进入 {selected_arm_side} 记录模式")
        records = _record_loop(connected_arm)
        csv_path = _write_csv(records, selected_arm_side)
        if csv_path is None:
            print("没有记录到任何数据，未生成 CSV")
        else:
            print(f"已保存到: {csv_path}")
        return 0
    finally:
        for arm in connected_arms.values():
            _shutdown_robot(arm.robot, arm.ec)


if __name__ == "__main__":
    raise SystemExit(main())
