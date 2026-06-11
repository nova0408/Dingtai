from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

DEFAULT_HOST = "192.168.100.60"
"默认基础控制工控机地址。"

DEFAULT_PORT = 50062
"默认 qmlinker 端口。"

DEFAULT_REQUEST_TIMEOUT_S = 10.0
"夹爪冒烟超时，单位 s。"

DEFAULT_POSITION_DELTA = 100
"夹爪位置小幅运动幅度，单位原始位置计数。"

POSITION_POLL_INTERVAL_S = 0.2
"夹爪位置轮询间隔，单位 s。"

POSITION_STABLE_CONFIRM_S = 2.0
"夹爪位置稳定后的确认等待时长，单位 s。"

POSITION_HARD_TIMEOUT_S = 20.0
"夹爪位置轮询硬超时，单位 s。"


@dataclass(frozen=True, slots=True)
class GripperSmokeConfig:
    """大寰夹爪冒烟测试配置。"""

    host: str = DEFAULT_HOST
    "基础控制工控机地址。"

    port: int = DEFAULT_PORT
    "qmlinker 端口。"

    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    "单次远端命令超时时间，单位 s。"

    position_delta: int = DEFAULT_POSITION_DELTA
    "夹爪位置小幅运动幅度，单位原始位置计数。"

    def target(self) -> str:
        """返回 qmlinker 连接目标。"""

        return f"{self.host}:{self.port}"


def _clamp_position(value: int) -> int:
    """将夹爪位置限制到常见的原始位置区间。"""

    return max(0, min(1000, int(value)))


def _wait_for_stable_position(
    gripper: DahuanGripperClient,
    target_position: int,
    *,
    timeout_s: float,
) -> int:
    """轮询夹爪位置，直到读数稳定后再额外确认。"""

    deadline = time.monotonic() + max(float(timeout_s), 0.0, POSITION_HARD_TIMEOUT_S)
    last_position: int | None = None
    stable_since: float | None = None

    while True:
        info = gripper.get_status()
        if info is None:
            logger.warning("failed to get gripper status, retrying...")
            time.sleep(POSITION_POLL_INTERVAL_S)
            continue
        current_position = info.position
        logger.info(
            "gripper polling position={} speed={} force={} state={}",
            current_position,
            info.speed,
            info.force,
            info.grip_state,
        )

        if current_position != last_position:
            last_position = current_position
            stable_since = None
        elif stable_since is None:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= POSITION_STABLE_CONFIRM_S:
            if current_position != int(target_position):
                raise RuntimeError(
                    f"gripper position stable at {current_position}, expected {target_position}"
                )
            logger.info("gripper position stable, confirm wait -> {}s", POSITION_STABLE_CONFIRM_S)
            time.sleep(POSITION_STABLE_CONFIRM_S)
            final_info = gripper.get_gripper_info()
            if int(final_info.position) != int(target_position):
                raise RuntimeError(
                    f"gripper final position mismatch: expected {target_position}, actual {final_info.position}"
                )
            return current_position

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"gripper position did not stabilize within {max(float(timeout_s), POSITION_HARD_TIMEOUT_S):.1f}s"
            )
        time.sleep(POSITION_POLL_INTERVAL_S)


def _append_repo_root_to_sys_path() -> None:
    """允许直接运行 test 下的脚本。"""

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "src").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_append_repo_root_to_sys_path()


from src.wuji.client_base import WujiQmlinkerBaseClient  # noqa: E402
from src.wuji.dahuan_gripper_client import DahuanGripperClient  # noqa: E402
from src.wuji.protocol import WujiQmlinkerConfig  # noqa: E402


# region 主入口


def run_gripper_smoke(config: GripperSmokeConfig) -> None:
    """按新 qmlinker SDK 读取并控制左手夹爪。

    这次冒烟验证只做夹爪位置的小幅往返运动，避免长时间保持极限状态。
    """

    base_client = WujiQmlinkerBaseClient(
        WujiQmlinkerConfig(
            host=config.host,
            port=config.port,
            request_timeout_s=float(config.request_timeout_s),
        )
    )
    try:
        gripper = DahuanGripperClient(base_client)

        logger.info("gripper smoke ready: target={}", config.target())
        info = gripper.get_gripper_info()
        logger.info("gripper online={} calibrated={} enabled={}", info.online, info.calibrated, info.enabled)
        logger.info("gripper position={} speed={} force={} state={}", info.position, info.speed, info.force, info.grip_state)

        if not bool(info.enabled):
            logger.info("gripper enable -> True")
            gripper.set_enable(True)
            time.sleep(0.2)

        current_info = gripper.get_gripper_info()
        current_position = int(current_info.position)
        delta = abs(int(config.position_delta))
        if delta <= 0:
            raise ValueError(f"position_delta 必须是正整数，当前为 {config.position_delta!r}")

        target_position = _clamp_position(current_position + delta)
        if target_position == current_position:
            target_position = _clamp_position(current_position - delta)

        logger.info("gripper position before={} target={}", current_position, target_position)
        logger.info("gripper speed -> 10")
        gripper.set_speed(10)
        logger.info("gripper force -> 10")
        gripper.set_force(10)
        logger.info("gripper position -> {}", target_position)
        gripper.set_pos(target_position)
        moved_position = _wait_for_stable_position(
            gripper,
            target_position,
            timeout_s=float(config.request_timeout_s),
        )
        logger.info("gripper after move position={}", moved_position)

        if target_position != current_position:
            logger.info("gripper position restore -> {}", current_position)
            gripper.set_pos(current_position)
            restored_position = _wait_for_stable_position(
                gripper,
                current_position,
                timeout_s=float(config.request_timeout_s),
            )
            logger.info("gripper restored position={}", restored_position)

        logger.info("gripper calibrate -> {}", gripper.calibrate())
        logger.success("gripper smoke passed")
    finally:
        base_client.close()


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S, position_delta: int = DEFAULT_POSITION_DELTA) -> None:
    """读取夹爪当前状态并验证基础控制链路。"""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    config = GripperSmokeConfig(request_timeout_s=float(request_timeout_s), position_delta=int(position_delta))
    run_gripper_smoke(config)


# endregion


# region CLI


def _parse_cli() -> tuple[float, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取并控制无际夹爪")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--position-delta", type=int, default=DEFAULT_POSITION_DELTA)
    args = parser.parse_args()
    return float(args.request_timeout_s), int(args.position_delta)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        request_timeout_s, position_delta = _parse_cli()
        main(request_timeout_s=request_timeout_s, position_delta=position_delta)
    else:
        main()


# endregion
