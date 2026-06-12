from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_orin_channel, stop_ssh_process  # noqa: E402
from src.wuji.arm_client import WujiArmClient  # noqa: E402

DEFAULT_ARM_NAME = "left_arm"
MOVE_WAIT_S = 3.0
JOINT_ANGLE_OFFSET_DEG = 5.0
IK_X_OFFSET_M = 0.005
IK_RY_OFFSET_DEG = 1.0
JOINT_TOLERANCE_DEG = 3.0
IK_POSITION_TOLERANCE_M = 0.05


def main() -> None:
    """机械臂基础冒烟测试。"""

    ssh_process, qmlinker_channel = create_orin_channel(DEFAULT_PORT)
    arm_client: WujiArmClient | None = None
    try:
        arm_client = WujiArmClient(qmlinker_channel, DEFAULT_ARM_NAME)

        logger.debug("关闭使能目标：期望=False")
        set_result = arm_client.set_enable(False)
        actual_enable = bool(arm_client.get_enable())
        if actual_enable:
            raise RuntimeError(f"关闭使能失败：返回={set_result} 实际={actual_enable}")

        logger.debug("打开使能目标：期望=True")
        set_result = arm_client.set_enable(True)
        actual_enable = bool(arm_client.get_enable())
        if not actual_enable:
            raise RuntimeError(f"打开使能失败：返回={set_result} 实际={actual_enable}")

        joint_states = arm_client.get_joint_states()
        baseline_angles = [float(joint.angle_deg) for joint in joint_states]
        if len(baseline_angles) != 6:
            raise RuntimeError(f"关节数量异常：{len(baseline_angles)}")
        logger.info("初始关节角 {}", baseline_angles)

        probe_angles = list(baseline_angles)
        probe_angles[0] = baseline_angles[0] + JOINT_ANGLE_OFFSET_DEG
        set_result = arm_client.set_joints(probe_angles)
        time.sleep(MOVE_WAIT_S)
        after_up_states = arm_client.get_joint_states()
        after_up_angles = [float(joint.angle_deg) for joint in after_up_states]
        up_error_deg = abs(after_up_angles[0] - probe_angles[0])
        if not set_result or up_error_deg > JOINT_TOLERANCE_DEG:
            raise RuntimeError(f"J1 正向微调失败：误差={up_error_deg:.3f}deg")

        current_pose = np.asarray(arm_client.current_fk_fast(), dtype=float)
        if current_pose.shape != (4, 4) or not np.all(np.isfinite(current_pose)):
            raise RuntimeError("FK 结果异常")
        logger.info("FK 当前 xyz {}", current_pose[:3, 3].tolist())

        target_pose = np.array(current_pose, dtype=float, copy=True)
        target_pose[0, 3] += IK_X_OFFSET_M
        angle_rad = np.deg2rad(IK_RY_OFFSET_DEG)
        rotate_y = np.array(
            [
                [np.cos(angle_rad), 0.0, np.sin(angle_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle_rad), 0.0, np.cos(angle_rad)],
            ],
            dtype=float,
        )
        target_pose[:3, :3] = target_pose[:3, :3] @ rotate_y

        ik_result = arm_client.ik(target_pose, np.deg2rad(after_up_angles))
        if ik_result is None or len(ik_result) != 6:
            raise RuntimeError("IK 求解失败")
        ik_angles_deg = [float(np.rad2deg(angle)) for angle in ik_result]
        logger.info("IK 结果角度 {}", ik_angles_deg)

        set_result = arm_client.set_joints(ik_angles_deg)
        time.sleep(MOVE_WAIT_S)
        verify_pose = np.asarray(arm_client.current_fk_fast(), dtype=float)
        if verify_pose.shape != (4, 4) or not np.all(np.isfinite(verify_pose)):
            raise RuntimeError("IK 验证 FK 结果异常")
        position_error_m = float(np.linalg.norm(verify_pose[:3, 3] - target_pose[:3, 3]))
        if not set_result or position_error_m > IK_POSITION_TOLERANCE_M:
            raise RuntimeError(f"IK 验证失败：位置误差={position_error_m:.4f}m")

        set_result = arm_client.set_joints(baseline_angles)
        time.sleep(MOVE_WAIT_S)
        final_states = arm_client.get_joint_states()
        final_angles = [float(joint.angle_deg) for joint in final_states]
        restore_error_deg = float(max(abs(actual - target) for actual, target in zip(final_angles, baseline_angles)))
        if not set_result or restore_error_deg > JOINT_TOLERANCE_DEG:
            raise RuntimeError(f"恢复初始关节失败：最大误差={restore_error_deg:.3f}deg")

        set_result = arm_client.set_enable(False)
        actual_enable = bool(arm_client.get_enable())
        if actual_enable:
            raise RuntimeError(f"再次关闭使能失败：返回={set_result} 实际={actual_enable}")

        logger.success("机械臂冒烟测试通过：最终关节角度 {}", final_angles)
    finally:
        if arm_client is not None:
            try:
                arm_client.set_enable(False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("清理阶段关闭使能失败：{}", exc)
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
