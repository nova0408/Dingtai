#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"
DEFAULT_CSV_PATH = PROJECT_ROOT / "record_left" / "close_door_left_20260629_143547.csv"
DEFAULT_ROW_INDEX = 8
MM_PER_M = 1000.0
DEFAULT_LOCAL_IP = os.environ.get("DINGTAI_XCORESDK_LOCAL_IP", "192.168.1.116").strip()
DEFAULT_ARM_ROBOT_IPS = {
    "left": os.environ.get("DINGTAI_XCORESDK_LEFT_IP", "192.168.1.161").strip(),
    "right": os.environ.get("DINGTAI_XCORESDK_RIGHT_IP", "192.168.1.160").strip(),
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk.xcoresdk import xCoreSDK_python  # noqa: E402


@dataclass(frozen=True, slots=True)
class ArmRow:
    """单条 arm 记录。"""

    row_index: int
    timestamp: str
    joints_deg: tuple[float, ...]
    pose_values: tuple[Any, ...]


def _parse_list_field(raw_text: str) -> list[Any]:
    parsed = ast.literal_eval(raw_text)
    if not isinstance(parsed, list):
        raise ValueError(f"字段不是列表: {raw_text!r}")
    return parsed


def _infer_arm_side_from_csv_path(csv_path: Path) -> str:
    lowered = str(csv_path).replace("\\", "/").lower()
    if "record_left" in lowered or "_left_" in lowered:
        return "left"
    if "record_right" in lowered or "_right_" in lowered:
        return "right"
    raise ValueError(f"无法从路径判断左右臂: {csv_path}")


def _read_arm_rows(csv_path: Path) -> list[ArmRow]:
    rows: list[ArmRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头")
        for row_index, row in enumerate(reader):
            if str(row.get("type", "")).strip().lower() != "arm":
                continue
            joints_raw = str(row.get("joints", "")).strip()
            pose_raw = str(row.get("pose", "")).strip()
            if not joints_raw.startswith("[") or not pose_raw.startswith("["):
                continue
            rows.append(
                ArmRow(
                    row_index=row_index,
                    timestamp=str(row.get("timestamp", "")),
                    joints_deg=tuple(float(value) for value in _parse_list_field(joints_raw)),
                    pose_values=tuple(_parse_list_field(pose_raw)),
                )
            )
    return rows


def _connect_robot_model(csv_path: Path) -> Any:
    arm_side = _infer_arm_side_from_csv_path(csv_path)
    robot_ip = DEFAULT_ARM_ROBOT_IPS[arm_side]
    logger.debug("连接机器人模型: arm_side={} robot_ip={} local_ip={}", arm_side, robot_ip, DEFAULT_LOCAL_IP)
    if DEFAULT_LOCAL_IP:
        robot = xCoreSDK_python.xMateErProRobot(robot_ip, DEFAULT_LOCAL_IP)
    else:
        robot = xCoreSDK_python.xMateErProRobot(robot_ip)
    return robot.model()


def _to_cartesian_position(
    pose_values: tuple[Any, ...],
    *,
    include_has_elbow: bool,
    include_elbow: bool,
    include_conf_data: bool,
    forced_has_elbow: bool | None = None,
) -> xCoreSDK_python.CartesianPosition:
    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = (float(value) for value in pose_values[:6])
    cart_pose = xCoreSDK_python.CartesianPosition(
        [
            x_mm / MM_PER_M,
            y_mm / MM_PER_M,
            z_mm / MM_PER_M,
            float(np.deg2rad(rx_deg)),
            float(np.deg2rad(ry_deg)),
            float(np.deg2rad(rz_deg)),
        ]
    )
    if include_has_elbow:
        cart_pose.hasElbow = bool(pose_values[6]) if forced_has_elbow is None else bool(forced_has_elbow)
    if include_elbow and len(pose_values) >= 8:
        cart_pose.elbow = float(np.deg2rad(float(pose_values[7])))
    if include_conf_data and len(pose_values) >= 9 and isinstance(pose_values[8], (list, tuple)):
        cart_pose.confData = [int(value) for value in pose_values[8]]
    return cart_pose


def _normalize_ik_output(raw_result: Any) -> list[tuple[float, ...]]:
    if raw_result is None:
        return []
    if isinstance(raw_result, (list, tuple)) and raw_result and isinstance(raw_result[0], (list, tuple, np.ndarray)):
        return [tuple(float(np.rad2deg(value)) for value in candidate) for candidate in raw_result]
    if isinstance(raw_result, (list, tuple, np.ndarray)):
        return [tuple(float(np.rad2deg(value)) for value in raw_result)]
    return []


def _probe_case(
    robot_model: Any,
    label: str,
    pose: xCoreSDK_python.CartesianPosition,
    *,
    with_toolset: bool,
) -> None:
    ec: dict[str, object] = {}
    logger.debug(
        "probe start: label={} with_toolset={} trans_m={} rpy_rad={} hasElbow={} elbow_rad={} confData={}",
        label,
        with_toolset,
        list(getattr(pose, "trans", [])),
        list(getattr(pose, "rpy", [])),
        bool(getattr(pose, "hasElbow", False)),
        float(getattr(pose, "elbow", 0.0)),
        list(getattr(pose, "confData", [])),
    )
    if with_toolset:
        toolset = xCoreSDK_python.Toolset()
        raw_result = robot_model.calcIk(pose, toolset, ec)
    else:
        raw_result = robot_model.calcIk(pose, ec)
    deg_candidates = _normalize_ik_output(raw_result)
    logger.debug(
        "probe done: label={} ec={} message={} raw_result={} deg_candidates={}",
        label,
        ec.get("ec", 0),
        ec.get("message", ""),
        raw_result,
        deg_candidates,
    )


def main() -> int:
    csv_path = DEFAULT_CSV_PATH
    arm_rows = _read_arm_rows(csv_path)
    target_row = arm_rows[DEFAULT_ROW_INDEX]
    logger.debug(
        "目标记录: csv={} row_index={} timestamp={} joints_deg={} pose_values={}",
        csv_path,
        target_row.row_index,
        target_row.timestamp,
        list(target_row.joints_deg),
        list(target_row.pose_values),
    )

    robot_model = _connect_robot_model(csv_path)

    cases: list[tuple[str, xCoreSDK_python.CartesianPosition, bool]] = [
        ("6d_only/no_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=False, include_elbow=False, include_conf_data=False), False),
        ("6d_only/with_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=False, include_elbow=False, include_conf_data=False), True),
        ("elbow_only/no_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=False, include_elbow=True, include_conf_data=False), False),
        ("elbow_only/with_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=False, include_elbow=True, include_conf_data=False), True),
        ("hasElbow_true+elbow/no_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=True, include_elbow=True, include_conf_data=False, forced_has_elbow=True), False),
        ("hasElbow_true+elbow/with_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=True, include_elbow=True, include_conf_data=False, forced_has_elbow=True), True),
        ("conf_only/no_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=False, include_elbow=False, include_conf_data=True), False),
        ("conf_only/with_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=False, include_elbow=False, include_conf_data=True), True),
        ("full_raw/no_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=True, include_elbow=True, include_conf_data=True), False),
        ("full_raw/with_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=True, include_elbow=True, include_conf_data=True), True),
        ("hasElbow_false+elbow+conf/no_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=True, include_elbow=True, include_conf_data=True, forced_has_elbow=False), False),
        ("hasElbow_false+elbow+conf/with_toolset", _to_cartesian_position(target_row.pose_values, include_has_elbow=True, include_elbow=True, include_conf_data=True, forced_has_elbow=False), True),
    ]

    for label, pose, with_toolset in cases:
        _probe_case(robot_model, label, pose, with_toolset=with_toolset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

