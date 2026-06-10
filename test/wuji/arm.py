from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wuji.arm_client import WujiArmClient  # noqa: E402
from src.wuji.client_base import WujiQmlinkerBaseClient  # noqa: E402
from src.wuji.protocol import WujiQmlinkerConfig  # noqa: E402


DEFAULT_HOST = "192.168.100.60"
DEFAULT_PORT = 50062

# 动作下发后的等待时间。
# 参考 `arm_example.py` 的实际节奏，这里给足 3 秒，避免读回过早。
MOVE_WAIT_S = 3.0

# 单关节测试的角度偏移量。
# 正向测试时 J1 = baseline + JOINT_ANGLE_OFFSET_DEG。
# 反向测试时 J1 = baseline - JOINT_ANGLE_OFFSET_DEG。
JOINT_ANGLE_OFFSET_DEG = 5.0

# IK 测试的末端位姿偏移量。
# IK_X_OFFSET_M 表示末端沿当前坐标的 x 方向平移。
# IK_RY_OFFSET_DEG 表示末端姿态绕 y 轴旋转。
IK_X_OFFSET_M = 0.005
IK_RY_OFFSET_DEG = 1.0

# 关节动作允许误差。
# qmlinker 或机械臂底层可能存在运动未完全到位、回读延迟等情况，所以这里不要求严格等于。
JOINT_TOLERANCE_DEG = 3.0

# IK 下发后的末端位置允许误差。
# 这个阈值只用于冒烟测试，主要判断 IK 解是否大体可执行。
IK_POSITION_TOLERANCE_M = 0.05


# region 主入口


def main() -> None:
    """机械臂基础冒烟测试。"""

    parser = argparse.ArgumentParser(description="无际机械臂基础冒烟测试")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="qmlinker 主机地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="qmlinker 端口")
    parser.add_argument("--arm", choices=("left", "right"), default="left", help="测试左臂或右臂，默认 left")
    args = parser.parse_args()

    # CLI 只暴露 left / right，内部仍使用项目里的设备名。
    arm_name = "left_arm" if args.arm == "left" else "right_arm"

    # 只在最上方打印一次测试对象。
    # 后续日志不再重复打印“机械臂=xxx”，避免日志过长。
    logger.info("测试对象：目标={}:{}，机械臂={}", args.host, args.port, arm_name)

    base_client = WujiQmlinkerBaseClient(WujiQmlinkerConfig(host=args.host, port=args.port))
    arm_client: WujiArmClient | None = None

    try:
        arm_client = WujiArmClient(base_client, arm_name)

        # ---------------------------------------------------------------------
        # 1. 先关闭使能。
        # 目的：确认 set_enable(False) 链路可用，并保证后续测试从非使能状态开始。
        # ---------------------------------------------------------------------
        logger.debug("关闭使能目标：期望=False")
        set_result = arm_client.set_enable(False)
        actual_enable = bool(arm_client.get_enable())
        passed = not actual_enable
        logger.info(
            "关闭使能结果：实际={}，返回={}，对比={}",
            actual_enable,
            set_result,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"关闭使能失败：实际使能={actual_enable}")

        # ---------------------------------------------------------------------
        # 2. 打开使能。
        # 目的：确认机械臂可以进入可控制状态。
        # ---------------------------------------------------------------------
        logger.debug("打开使能目标：期望=True")
        set_result = arm_client.set_enable(True)
        actual_enable = bool(arm_client.get_enable())
        passed = actual_enable
        logger.info(
            "打开使能结果：实际={}，返回={}，对比={}",
            actual_enable,
            set_result,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"打开使能失败：实际使能={actual_enable}")

        # ---------------------------------------------------------------------
        # 3. 读取当前关节状态。
        # 目的：确认状态读取链路可用，并保存基线角度，后续测试结束后恢复到该位置。
        # ---------------------------------------------------------------------
        logger.debug("读取关节目标：期望关节数量=6")
        joint_states = arm_client.get_joint_states(arm_name)
        joint_count = len(joint_states)
        baseline_angles = [float(joint.angle_deg) for joint in joint_states]
        passed = joint_count == 6
        logger.info(
            "读取关节结果：数量={}，角度={}，对比={}",
            joint_count,
            baseline_angles,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"关节数量异常：期望=6，实际={joint_count}")

        # ---------------------------------------------------------------------
        # 4. J1 正向微调。
        # 目的：确认整臂关节目标下发接口可用。
        # 注意：这里只改第 0 个关节，也就是底层 joint_id=1。
        # ---------------------------------------------------------------------
        arm_client.set_enable(True)  # 确保使能状态，避免测试过程中被误操作关闭使能后无法继续测试。
        probe_angles = list(baseline_angles)
        probe_angles[0] = baseline_angles[0] + JOINT_ANGLE_OFFSET_DEG

        logger.debug("J1 正向微调目标：目标={:.3f}deg", probe_angles[0])
        set_result = arm_client.set_joints(probe_angles)
        time.sleep(MOVE_WAIT_S)

        after_up_states = arm_client.get_joint_states(arm_name)
        after_up_angles = [float(joint.angle_deg) for joint in after_up_states]
        up_error_deg = abs(after_up_angles[0] - probe_angles[0])
        passed = bool(set_result) and up_error_deg <= JOINT_TOLERANCE_DEG

        logger.info(
            "J1 正向微调结果：实际={:.3f}deg，误差={:.3f}deg，阈值={:.3f}deg，返回={}，对比={}",
            after_up_angles[0],
            up_error_deg,
            JOINT_TOLERANCE_DEG,
            set_result,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"J1 正向微调失败：误差={up_error_deg:.3f}deg")

        # ---------------------------------------------------------------------
        # 6. FK 测试。
        # 目的：读取当前末端位姿，检查 FK 接口是否返回有效 4x4 矩阵。
        # 这里不打印完整矩阵，只打印末端 xyz。
        # ---------------------------------------------------------------------
        logger.debug("FK 目标：读取当前末端 4x4 位姿矩阵")
        time.sleep(MOVE_WAIT_S)

        current_pose = arm_client.current_fk_fast(arm_name)
        if current_pose is None:
            raise RuntimeError("FK 失败：返回 None")

        current_pose = np.asarray(current_pose, dtype=float)
        if current_pose.shape != (4, 4):
            raise RuntimeError(f"FK 失败：矩阵尺寸异常，shape={current_pose.shape}")
        if not np.all(np.isfinite(current_pose)):
            raise RuntimeError("FK 失败：矩阵中存在非有限数值")

        current_xyz = current_pose[:3, 3]
        logger.info(
            "FK 结果：xyz=({:.4f}, {:.4f}, {:.4f})m，对比=通过",
            current_xyz[0],
            current_xyz[1],
            current_xyz[2],
        )

        # ---------------------------------------------------------------------
        # 7. 构造 IK 目标位姿。
        # 目的：在当前 FK 位姿基础上构造一个很小的偏移目标，降低 IK 失败概率。
        # 注意：这里只做冒烟测试，不做大范围空间规划。
        # ---------------------------------------------------------------------
        target_pose = np.array(current_pose, dtype=float, copy=True)
        target_pose[0, 3] = target_pose[0, 3] + IK_X_OFFSET_M

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

        # ---------------------------------------------------------------------
        # 8. IK 求解。
        # 目的：确认 IK 接口可以根据目标位姿和参考关节求解出 6 个关节角。
        # 参考关节使用当前反向微调后的关节角度。
        # ---------------------------------------------------------------------
        logger.debug("IK 目标：x偏移={:.4f}m，ry偏移={:.3f}deg", IK_X_OFFSET_M, IK_RY_OFFSET_DEG)
        reference_angles_rad = np.deg2rad(after_down_angles)
        ik_result = arm_client.ik(target_pose, reference_angles_rad)

        if ik_result is None:
            raise RuntimeError("IK 失败：返回 None")
        if len(ik_result) != 6:
            raise RuntimeError(f"IK 失败：结果数量异常，数量={len(ik_result)}")

        ik_angles_deg = [float(np.rad2deg(angle)) for angle in ik_result]
        logger.info("IK 结果：关节角度={}，对比=通过", ik_angles_deg)

        # ---------------------------------------------------------------------
        # 9. 下发 IK 解，并通过 FK 验证末端位置误差。
        # 目的：确认 IK 求解结果不仅能返回，也能被机械臂执行。
        # ---------------------------------------------------------------------
        logger.debug("IK 下发目标：关节角度={}", ik_angles_deg)
        set_result = arm_client.set_joints(ik_angles_deg)
        time.sleep(MOVE_WAIT_S)

        verify_pose = arm_client.current_fk_fast(arm_name)
        if verify_pose is None:
            raise RuntimeError("IK 验证失败：FK 返回 None")

        verify_pose = np.asarray(verify_pose, dtype=float)
        if verify_pose.shape != (4, 4):
            raise RuntimeError(f"IK 验证失败：矩阵尺寸异常，shape={verify_pose.shape}")
        if not np.all(np.isfinite(verify_pose)):
            raise RuntimeError("IK 验证失败：矩阵中存在非有限数值")

        position_error_m = float(np.linalg.norm(verify_pose[:3, 3] - target_pose[:3, 3]))
        passed = bool(set_result) and position_error_m <= IK_POSITION_TOLERANCE_M

        logger.info(
            "IK 验证结果：位置误差={:.4f}m，阈值={:.4f}m，返回={}，对比={}",
            position_error_m,
            IK_POSITION_TOLERANCE_M,
            set_result,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"IK 验证失败：位置误差={position_error_m:.4f}m")

        # ---------------------------------------------------------------------
        # 10. 恢复初始关节角度。
        # 目的：冒烟测试结束后尽量回到测试开始位置。
        # ---------------------------------------------------------------------
        logger.debug("恢复初始关节目标：角度={}", baseline_angles)
        set_result = arm_client.set_joints(baseline_angles)
        time.sleep(MOVE_WAIT_S)

        final_states = arm_client.get_joint_states(arm_name)
        final_angles = [float(joint.angle_deg) for joint in final_states]
        restore_error_deg = float(max(abs(actual - target) for actual, target in zip(final_angles, baseline_angles)))
        passed = bool(set_result) and restore_error_deg <= JOINT_TOLERANCE_DEG

        logger.info(
            "恢复初始关节结果：最大误差={:.3f}deg，阈值={:.3f}deg，返回={}，对比={}",
            restore_error_deg,
            JOINT_TOLERANCE_DEG,
            set_result,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"恢复初始关节失败：最大误差={restore_error_deg:.3f}deg")

        # ---------------------------------------------------------------------
        # 11. 最后关闭使能。
        # 目的：保证测试结束后机械臂不保持使能状态。
        # ---------------------------------------------------------------------
        logger.debug("再次关闭使能目标：期望=False")
        set_result = arm_client.set_enable(False)
        actual_enable = bool(arm_client.get_enable())
        passed = not actual_enable

        logger.info(
            "再次关闭使能结果：实际={}，返回={}，对比={}",
            actual_enable,
            set_result,
            "通过" if passed else "失败",
        )
        if not passed:
            raise RuntimeError(f"再次关闭使能失败：实际使能={actual_enable}")

        logger.success("机械臂冒烟测试通过：最终使能={}，最终关节角度={}", actual_enable, final_angles)

    finally:
        # 测试脚本可能中途异常。
        # finally 中兜底关闭使能，避免异常退出后机械臂仍处于使能状态。
        if arm_client is not None:
            try:
                arm_client.set_enable(False)
            except Exception as exc:
                logger.warning("清理阶段关闭使能失败：{}", exc)

        base_client.close()


if __name__ == "__main__":
    main()


# endregion
