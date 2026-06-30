from __future__ import annotations

# region 依赖导入
import argparse
import sys
from pathlib import Path
import time
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.body_client import WujiBodyClient

# endregion


# region 默认参数

DEFAULT_REQUEST_TIMEOUT_S = 3.0

# endregion


# region 主入口


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """交互式控制 body 的 lift 和 waist。"""

    logger.info("身体控制脚本启动，请先确认 Orin 连接正常。")
    logger.info("请求超时 {} s", request_timeout_s)

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    body_client = WujiBodyClient(qmlinker_channel)
    try:
        _interactive_loop(body_client=body_client)
    finally:
        stop_ssh_process(ssh_process)


# endregion


# region 交互逻辑


def _interactive_loop(body_client: WujiBodyClient) -> None:
    """通过命令行交互控制 lift 与 waist。"""

    while True:
        print()
        print("控制举升：lift")
        print("控制腰部：waist")
        print("退出：q")
        value = input("请输入指令: ").strip().lower()
        if value == "q":
            break
        if value == "lift":
            _control_lift(body_client)
            continue
        if value == "waist":
            _control_waist(body_client)
            continue
        logger.warning("未知指令 {}", value)


def _control_lift(body_client: WujiBodyClient) -> None:
    """控制升降机构。"""

    lift = body_client.lift
    lift.set_enable(True)
    logger.info("lift 当前使能 {}", lift.get_enable())
    physical_height_mm = _read_lift_height_mm(lift.get_lift_physical_height())
    logger.info("lift 当前物理高度 {:.1f} mm", physical_height_mm)
    logger.info("lift 当前目标高度 {}", lift.get_lift_height()[0])
    value = input("请输入升降高度（mm），输入 q 返回: ").strip().lower()
    if value == "q":
        return
    target_height = _parse_float(value, "升降高度")
    if target_height < 0.0:
        raise ValueError(f"升降高度超出范围: {target_height} mm，合法范围 [0, +inf) mm")
    target_height_mm = int(round(target_height))
    lift.set_lift_physical_height(target_height_mm)
    logger.info("已下发 lift 目标高度 {} mm", target_height_mm)
    logger.info("等待执行结果...")
    time.sleep(3.0)
    physical_height_mm = _read_lift_height_mm(lift.get_lift_physical_height())
    logger.info("lift 当前物理高度 {:.1f} mm", physical_height_mm)
    logger.info("lift 当前目标高度 {}", lift.get_lift_height()[0])


def _control_waist(body_client: WujiBodyClient) -> None:
    """控制腰部俯仰。"""

    waist = body_client.waist
    waist.set_enable(True)
    logger.info("waist 当前使能 {}", waist.get_enable())
    logger.info("waist 当前俯仰 {} deg", waist.get_waist_pitch())
    value = input("请输入腰部俯仰角度（deg），输入 q 返回: ").strip().lower()
    if value == "q":
        return
    target_pitch = _parse_float(value, "腰部俯仰角度")
    waist.set_waist_pitch(target_pitch)
    logger.info("已下发 waist 目标俯仰 {} deg", target_pitch)
    logger.info("waist 当前俯仰 {} deg", waist.get_waist_pitch())


# endregion


# region 工具


def _parse_float(value: str, label: str) -> float:
    """解析数值输入。"""

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} 输入不是有效数字: {value}") from exc


def _read_lift_height_mm(result: object) -> float:
    """将 lift 读取结果转换成毫米。"""

    if isinstance(result, tuple) and len(result) == 2:
        return float(result[0])
    return float(result)


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="控制无际身体 lift 和 waist")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args()
    return float(args.request_timeout_s)


# endregion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(request_timeout_s=_parse_cli())
    else:
        main()
