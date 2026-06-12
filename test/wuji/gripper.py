from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


# region 路径初始化


current_file = Path(__file__).resolve()
script_dir = current_file.parent
for parent in current_file.parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))


from gripper_common import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_TEST_POSITION,
    DahuanGripperInfo,
    GripperSmokeConfig,
    build_gripper_client,
    choose_test_position,
    ensure_success,
    read_status,
    wait_for_stable_position,
)


# endregion


# region 辅助输出


def _print_status(title: str, status: DahuanGripperInfo) -> None:
    """中文打印夹爪状态。"""

    print("")
    print(f"========== {title} ==========")
    print(f"在线状态      : {status.online}")
    print(f"校准状态      : {status.calibrated}")
    print(f"使能状态      : {status.enable}")
    print(f"当前位置      : {status.position}")
    print(f"夹持状态码    : {status.state}")
    print("", flush=True)


# endregion


# region 冒烟测试


def run_gripper_smoke(config: GripperSmokeConfig) -> None:
    """读取并控制左手大寰夹爪。"""

    base_client, client = build_gripper_client(config.host, config.request_timeout_s)
    try:
        print("")
        print("========== 大寰夹爪冒烟测试开始 ==========")
        print(f"目标主机      : {config.host}")
        print(f"请求超时      : {config.request_timeout_s} s")
        print(f"测试位置      : {config.test_position}", flush=True)

        status = read_status(client)
        _print_status("初始夹爪状态", status)
        if not bool(status.online):
            raise RuntimeError("状态测试失败：夹爪不在线。")
        logger.info("状态读取测试通过")

        print("")
        print("========== 测试夹爪使能 ==========", flush=True)
        ensure_success(client.set_enable(True), "设置夹爪使能")
        enabled_status = read_status(client)
        _print_status("使能后的夹爪状态", enabled_status)
        if enabled_status.enable is None:
            logger.info("夹爪使能回读字段不可用，已确认设置调用成功")
        elif enabled_status.enable:
            logger.info("夹爪使能测试通过")
        else:
            logger.warning("夹爪使能回读仍为 False，继续执行位置测试")

        current_status = read_status(client)
        current_position = current_status.position
        target_position = choose_test_position(current_position, config.test_position)

        print("")
        print("========== 测试夹爪位置设置 ==========", flush=True)
        print(f"设置位置：{current_position} -> {target_position}", flush=True)
        status = read_status(client)
        _print_status("夹爪状态", status)
        ensure_success(client.set_pos(target_position), "设置夹爪位置")
        stable_result = wait_for_stable_position(
            client,
            current_position,
            target_position,
            timeout_s=config.request_timeout_s,
        )
        logger.info(
            "夹爪位置设置测试通过 position={} elapsed_s={:.3f} sample_count={} stable_s={:.3f}",
            stable_result.final_position,
            stable_result.elapsed_s,
            stable_result.sample_count,
            stable_result.stable_duration_s,
        )

        final_status = read_status(client)
        _print_status("测试完成时夹爪状态", final_status)
        logger.success("大寰夹爪冒烟测试通过")
    finally:
        base_client.close()

    print("")
    print("========== 大寰夹爪冒烟测试结束 ==========", flush=True)


def main(
    host: str = DEFAULT_HOST,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    test_position: int = DEFAULT_TEST_POSITION,
) -> None:
    """读取夹爪当前状态并验证基础控制链路。"""

    config = GripperSmokeConfig(
        host=host,
        request_timeout_s=request_timeout_s,
        test_position=test_position,
    )
    run_gripper_smoke(config)


# endregion


# region CLI


def _parse_cli() -> tuple[str, float, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取并控制无际左手大寰夹爪")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
    )
    parser.add_argument("--test-position", type=int, default=DEFAULT_TEST_POSITION)
    args = parser.parse_args()

    return (
        str(args.host),
        float(args.request_timeout_s),
        int(args.test_position),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        host, request_timeout_s, test_position = _parse_cli()
        main(
            host=host,
            request_timeout_s=request_timeout_s,
            test_position=test_position,
        )
    else:
        main()


# endregion
