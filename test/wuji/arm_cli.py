from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

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

# 指令下发后最多等待 1 秒。
# 等待期间持续轮询关节状态，直到关节值稳定，而不是下发后马上读一次。
MOVE_SETTLE_TIMEOUT_S = 1.0

# 轮询间隔。
# 50 ms 读一次，1 秒内最多约 20 次。
MOVE_POLL_INTERVAL_S = 0.05

# 判断“关节值不再变化”的阈值。
# 如果相邻两帧最大关节变化量小于该值，认为这一帧稳定。
JOINT_STABLE_EPS_DEG = 0.02

# 连续多少次稳定后认为动作结果可以读取。
JOINT_STABLE_COUNT = 3


def _read_angles(arm_client: WujiArmClient, arm_name: str) -> list[float] | None:
    """读取当前 6 轴角度。"""

    joint_states = arm_client.get_joint_states(arm_name)
    angles = [float(joint.angle_deg) for joint in joint_states]

    if len(angles) != 6:
        return None

    return angles


def _wait_angles_stable(
    arm_client: WujiArmClient,
    arm_name: str,
    timeout_s: float,
) -> tuple[list[float] | None, bool, float, int]:
    """轮询读取关节角，直到稳定或超时。

    Returns
    -------
    tuple[list[float] | None, bool, float, int]
        返回 ``(angles, settled, elapsed_s, sample_count)``。

        - ``angles``：最后一次有效 6 轴角度；
        - ``settled``：是否在超时前稳定；
        - ``elapsed_s``：实际等待时间；
        - ``sample_count``：有效采样次数。
    """

    start_s = time.monotonic()
    last_angles: list[float] | None = None
    latest_angles: list[float] | None = None
    stable_count = 0
    sample_count = 0

    while time.monotonic() - start_s <= timeout_s:
        current_angles = _read_angles(arm_client, arm_name)

        if current_angles is None:
            time.sleep(MOVE_POLL_INTERVAL_S)
            continue

        sample_count += 1
        latest_angles = current_angles

        if last_angles is not None:
            max_delta_deg = max(
                abs(current_angle - last_angle)
                for current_angle, last_angle in zip(current_angles, last_angles)
            )

            if max_delta_deg <= JOINT_STABLE_EPS_DEG:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= JOINT_STABLE_COUNT:
                elapsed_s = time.monotonic() - start_s
                return current_angles, True, elapsed_s, sample_count

        last_angles = current_angles
        time.sleep(MOVE_POLL_INTERVAL_S)

    elapsed_s = time.monotonic() - start_s
    return latest_angles, False, elapsed_s, sample_count


def _xyzrpy_to_pose(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """将 xyzrpy 转为 4x4 齐次矩阵。

    Notes
    -----
    RPY 使用 ZYX 欧拉角约定：

    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``
    """

    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0.0, np.sin(roll_rad), np.cos(roll_rad)],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rz @ ry @ rx
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def main() -> None:
    """无际机械臂手动控制 CLI。"""

    parser = argparse.ArgumentParser(description="无际机械臂手动控制 CLI")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="qmlinker 主机地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="qmlinker 端口")
    parser.add_argument("--arm", choices=("left", "right"), default="left", help="控制左臂或右臂，默认 left")
    args = parser.parse_args()

    arm_name = "left_arm" if args.arm == "left" else "right_arm"

    logger.info("控制对象：目标={}:{}，机械臂={}", args.host, args.port, arm_name)

    base_client = WujiQmlinkerBaseClient(WujiQmlinkerConfig(host=args.host, port=args.port))
    arm_client: WujiArmClient | None = None

    try:
        arm_client = WujiArmClient(base_client, arm_name)

        while True:
            print()
            print("请选择控制模式：")
            print("  enable  : 打开使能")
            print("  disable : 关闭使能")
            print("  joint   : 单轴控制，进入后持续控制，输入 q 返回主菜单")
            print("  fk      : xyzrpy 控制，进入后持续控制，输入 q 返回主菜单")
            print("  s       : 停止当前动作")
            print("  exit    : 退出程序")

            mode = input("模式> ").strip().lower()

            if mode in {"exit", "quit"}:
                break

            if mode == "s":
                logger.debug("停止目标：调用 stop")
                stop_result = arm_client.stop()
                logger.info("停止结果：返回={}", stop_result)
                continue

            if mode == "enable":
                logger.debug("打开使能目标：期望=True")
                set_result = arm_client.set_enable(True)
                actual_enable = bool(arm_client.get_enable())
                logger.info("打开使能结果：实际={}，返回={}", actual_enable, set_result)
                continue

            if mode == "disable":
                logger.debug("关闭使能目标：期望=False")
                set_result = arm_client.set_enable(False)
                actual_enable = bool(arm_client.get_enable())
                logger.info("关闭使能结果：实际={}，返回={}", actual_enable, set_result)
                continue

            # -----------------------------------------------------------------
            # joint 模式：
            # 进入后停留在 joint 模式。
            #
            # 输入流程：
            #   1. 每轮先显示最新 6 个 axis 当前角度；
            #   2. 输入 axis index；
            #   3. 再次读取最新角度，并显示该 axis 当前值；
            #   4. 输入真实目标角度 deg；
            #   5. 调用 arm_client.set_joint(...) 执行单轴控制；
            #   6. 下发后轮询状态，直到关节角稳定或 1 秒超时；
            #   7. 再输出目标、实际、误差。
            #
            # 注意：
            #   CLI 中 axis_index 是 0 开始；
            #   WujiArmClient.set_joint() 的 joint_index 是 1 开始；
            #   所以调用 set_joint 时传 axis_index + 1。
            # -----------------------------------------------------------------
            if mode == "joint":
                while True:
                    current_angles = _read_angles(arm_client, arm_name)

                    if current_angles is None:
                        logger.warning("关节数量异常：未读取到 6 个关节")
                        continue

                    print()
                    print("当前 axis 值：")
                    for index, angle_deg in enumerate(current_angles):
                        print(f"  [{index}] J{index + 1}: {angle_deg:.3f} deg")

                    print()
                    print("joint 模式输入说明：")
                    print("  输入 axis index[0-5] 选择要控制的轴")
                    print("  输入 s 停止当前动作并返回主菜单")
                    print("  输入 q 返回主菜单")

                    index_text = input("axis index> ").strip().lower()

                    if index_text == "q":
                        break

                    if index_text == "s":
                        logger.debug("停止目标：调用 stop")
                        stop_result = arm_client.stop()
                        logger.info("停止结果：返回={}", stop_result)
                        break

                    try:
                        axis_index = int(index_text)
                    except ValueError:
                        logger.warning("axis index 输入错误：必须是整数")
                        continue

                    if not 0 <= axis_index < len(current_angles):
                        logger.warning("axis index 越界：index={}，合法范围=[0, {}]", axis_index, len(current_angles) - 1)
                        continue

                    refreshed_angles = _read_angles(arm_client, arm_name)

                    if refreshed_angles is None:
                        logger.warning("关节数量异常：未读取到 6 个关节")
                        continue

                    print()
                    print(f"已选择 axis：J{axis_index + 1}")
                    print(f"当前值：{refreshed_angles[axis_index]:.3f} deg")
                    print("输入 s 停止当前动作并返回主菜单，输入 q 返回主菜单。")

                    target_text = input("请输入真实目标角度 deg> ").strip().lower()

                    if target_text == "q":
                        break

                    if target_text == "s":
                        logger.debug("停止目标：调用 stop")
                        stop_result = arm_client.stop()
                        logger.info("停止结果：返回={}", stop_result)
                        break

                    try:
                        target_angle_deg = float(target_text)
                    except ValueError:
                        logger.warning("目标角度输入错误：必须是数字")
                        continue

                    latest_angles = _read_angles(arm_client, arm_name)

                    if latest_angles is None:
                        logger.warning("关节数量异常：未读取到 6 个关节")
                        continue

                    logger.debug(
                        "单轴控制目标：J{} 当前={:.3f}deg，目标={:.3f}deg",
                        axis_index + 1,
                        latest_angles[axis_index],
                        target_angle_deg,
                    )

                    set_result = arm_client.set_joint(
                        device_name=arm_name,
                        joint_index=axis_index + 1,
                        target_angle_deg=target_angle_deg,
                    )

                    if not set_result:
                        logger.warning(
                            "单轴控制未执行：J{} 目标={:.3f}deg，返回={}",
                            axis_index + 1,
                            target_angle_deg,
                            set_result,
                        )
                        continue

                    after_angles, settled, elapsed_s, sample_count = _wait_angles_stable(
                        arm_client=arm_client,
                        arm_name=arm_name,
                        timeout_s=MOVE_SETTLE_TIMEOUT_S,
                    )

                    if after_angles is None:
                        logger.warning("单轴控制结果读取失败：1 秒内未获得有效关节状态")
                        continue

                    actual_angle_deg = after_angles[axis_index]
                    error_deg = abs(actual_angle_deg - target_angle_deg)

                    logger.info(
                        "单轴控制结果：J{} 目标={:.3f}deg，实际={:.3f}deg，误差={:.3f}deg，返回={}，稳定={}，等待={:.3f}s，采样={}",
                        axis_index + 1,
                        target_angle_deg,
                        actual_angle_deg,
                        error_deg,
                        set_result,
                        settled,
                        elapsed_s,
                        sample_count,
                    )

                continue

            # -----------------------------------------------------------------
            # fk 模式：
            # 这里保留模式名 fk，但实际是 xyzrpy 末端位姿控制。
            #
            # 输入流程：
            #   1. 每轮先读取并显示最新关节角；
            #   2. 每轮先读取并显示最新 FK 正解 xyzrpy；
            #   3. 输入 x y z roll pitch yaw；
            #   4. 输入后重新读取最新关节角作为 IK 参考值；
            #   5. xyzrpy -> 4x4 位姿矩阵；
            #   6. IK 求解目标关节角；
            #   7. set_joints 下发；
            #   8. 下发后轮询状态，直到关节角稳定或 1 秒超时；
            #   9. 执行后重新读取最新 FK 做对比。
            #
            # 单位：
            #   x / y / z 使用 m；
            #   roll / pitch / yaw 使用 deg。
            #
            # RPY 约定：
            #   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
            # -----------------------------------------------------------------
            if mode == "fk":
                while True:
                    current_angles = _read_angles(arm_client, arm_name)

                    if current_angles is None:
                        logger.warning("关节数量异常：未读取到 6 个关节")
                        continue

                    try:
                        current_x, current_y, current_z, current_roll, current_pitch, current_yaw = arm_client.current_fk_xyzrpy(arm_name)
                    except Exception as exc:
                        logger.warning("FK 正解读取失败：{}", exc)
                        continue

                    print()
                    print("当前关节角度：")
                    for index, angle_deg in enumerate(current_angles):
                        print(f"  [{index}] J{index + 1}: {angle_deg:.3f} deg")

                    print()
                    print("当前 FK 正解结果：")
                    print(f"  x    : {current_x:.6f} m")
                    print(f"  y    : {current_y:.6f} m")
                    print(f"  z    : {current_z:.6f} m")
                    print(f"  roll : {current_roll:.3f} deg")
                    print(f"  pitch: {current_pitch:.3f} deg")
                    print(f"  yaw  : {current_yaw:.3f} deg")

                    print()
                    print("fk 模式输入说明：")
                    print("  输入 x y z roll pitch yaw")
                    print("  示例：0.4 0.26 -0.44 -89.5 -85 -90")
                    print("  输入 s 停止当前动作并返回主菜单")
                    print("  输入 q 返回主菜单")

                    values_text = input("目标 xyzrpy> ").strip().lower()

                    if values_text == "q":
                        break

                    if values_text == "s":
                        logger.debug("停止目标：调用 stop")
                        stop_result = arm_client.stop()
                        logger.info("停止结果：返回={}", stop_result)
                        break

                    value_parts = values_text.replace(",", " ").split()

                    if len(value_parts) != 6:
                        logger.warning("输入数量错误：需要 6 个值，实际输入 {} 个", len(value_parts))
                        continue

                    try:
                        target_x = float(value_parts[0])
                        target_y = float(value_parts[1])
                        target_z = float(value_parts[2])
                        target_roll_deg = float(value_parts[3])
                        target_pitch_deg = float(value_parts[4])
                        target_yaw_deg = float(value_parts[5])
                    except ValueError:
                        logger.warning("输入格式错误：x y z roll pitch yaw 都必须是数字")
                        continue

                    # 用户输入完成后，再重新读取一次最新关节角。
                    # 这组角度用于 IK 参考值，不能直接使用输入前显示用的 current_angles。
                    latest_angles = _read_angles(arm_client, arm_name)

                    if latest_angles is None:
                        logger.warning("IK 前关节状态读取异常：未读取到 6 个关节")
                        continue

                    target_pose = _xyzrpy_to_pose(
                        x=target_x,
                        y=target_y,
                        z=target_z,
                        roll_deg=target_roll_deg,
                        pitch_deg=target_pitch_deg,
                        yaw_deg=target_yaw_deg,
                    )

                    logger.debug(
                        "xyzrpy 控制目标：x={:.6f}m，y={:.6f}m，z={:.6f}m，roll={:.3f}deg，pitch={:.3f}deg，yaw={:.3f}deg",
                        target_x,
                        target_y,
                        target_z,
                        target_roll_deg,
                        target_pitch_deg,
                        target_yaw_deg,
                    )
                    logger.debug("IK 参考关节角度：{}", latest_angles)

                    reference_angles_rad = np.deg2rad(latest_angles)
                    ik_result: Any = arm_client.ik(target_pose, reference_angles_rad)

                    if ik_result is None:
                        logger.warning("IK 求解失败：返回 None")
                        continue

                    if len(ik_result) != 6:
                        logger.warning("IK 求解失败：结果数量异常，数量={}", len(ik_result))
                        continue

                    target_angles = [float(np.rad2deg(angle_rad)) for angle_rad in ik_result]

                    logger.debug("IK 求解结果：目标关节角度={}", target_angles)

                    set_result = arm_client.set_joints(target_angles)

                    if not set_result:
                        logger.warning("xyzrpy 控制未执行：set_joints 返回 False，目标关节角度={}", target_angles)
                        continue

                    after_angles, settled, elapsed_s, sample_count = _wait_angles_stable(
                        arm_client=arm_client,
                        arm_name=arm_name,
                        timeout_s=MOVE_SETTLE_TIMEOUT_S,
                    )

                    if after_angles is None:
                        logger.warning("xyzrpy 控制结果读取失败：1 秒内未获得有效关节状态")
                        continue

                    try:
                        actual_x, actual_y, actual_z, actual_roll, actual_pitch, actual_yaw = arm_client.current_fk_xyzrpy(arm_name)
                    except Exception as exc:
                        logger.warning("执行后 FK 正解读取失败：{}", exc)
                        continue

                    position_error = float(
                        np.linalg.norm(
                            np.array(
                                [
                                    actual_x - target_x,
                                    actual_y - target_y,
                                    actual_z - target_z,
                                ],
                                dtype=float,
                            )
                        )
                    )

                    # 角度误差 wrap 到 [-180, 180]。
                    # 避免 179 deg 和 -181 deg 这种等价角被误判为 360 deg。
                    roll_error = abs((actual_roll - target_roll_deg + 180.0) % 360.0 - 180.0)
                    pitch_error = abs((actual_pitch - target_pitch_deg + 180.0) % 360.0 - 180.0)
                    yaw_error = abs((actual_yaw - target_yaw_deg + 180.0) % 360.0 - 180.0)

                    joint_errors = [
                        abs(actual - target)
                        for actual, target in zip(after_angles, target_angles)
                    ]

                    logger.info(
                        "xyzrpy 控制结果：返回={}，稳定={}，等待={:.3f}s，采样={}",
                        set_result,
                        settled,
                        elapsed_s,
                        sample_count,
                    )
                    logger.info(
                        "目标 xyzrpy：x={:.6f}，y={:.6f}，z={:.6f}，roll={:.3f}，pitch={:.3f}，yaw={:.3f}",
                        target_x,
                        target_y,
                        target_z,
                        target_roll_deg,
                        target_pitch_deg,
                        target_yaw_deg,
                    )
                    logger.info(
                        "实际 xyzrpy：x={:.6f}，y={:.6f}，z={:.6f}，roll={:.3f}，pitch={:.3f}，yaw={:.3f}",
                        actual_x,
                        actual_y,
                        actual_z,
                        actual_roll,
                        actual_pitch,
                        actual_yaw,
                    )
                    logger.info(
                        "xyzrpy 误差：位置={:.6f}m，roll={:.3f}deg，pitch={:.3f}deg，yaw={:.3f}deg",
                        position_error,
                        roll_error,
                        pitch_error,
                        yaw_error,
                    )
                    logger.info("目标关节角度：{}", target_angles)
                    logger.info("实际关节角度：{}", after_angles)
                    logger.info("关节角度误差：{}", joint_errors)

                continue

            logger.warning("未知模式：{}，请输入 enable、disable、joint、fk、s 或 exit", mode)

    finally:
        if arm_client is not None:
            try:
                logger.debug("退出清理目标：关闭使能")
                set_result = arm_client.set_enable(False)
                actual_enable = bool(arm_client.get_enable())
                logger.info("退出清理结果：实际使能={}，返回={}", actual_enable, set_result)
            except Exception as exc:
                logger.warning("清理阶段关闭使能失败：{}", exc)

        base_client.close()


if __name__ == "__main__":
    main()