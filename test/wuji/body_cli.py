from __future__ import annotations

# region 依赖导入
import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.body_client import WujiBodyClient, WujiLiftClient

# endregion


# region 默认参数

LIFT_WAIT_S = 3.0  # 升降柱命令执行后的等待时间，单位 s。

# endregion


# region 主入口


def main() -> None:
    """交互式控制 body 的 lift 和 waist。"""

    logger.info("身体控制脚本启动，请先确认 Orin 连接正常。")

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    body_client = WujiBodyClient(qmlinker_channel)
    try:
        _interactive_loop(body_client=body_client)
        logger.success("无际身体交互式控制结束")
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
    if not lift.get_enable():
        raise RuntimeError("升降柱使能失败")
    logger.info("升降柱当前使能 {}", lift.get_enable())
    _log_lift_height(lift)

    value = input("请输入升降高度（mm），输入 q 返回: ").strip().lower()
    if value == "q":
        return
    try:
        target_height = _parse_float(value, "升降高度")
    except ValueError as exc:
        logger.warning("{}", exc)
        return
    if target_height < 0.0:
        logger.warning("升降高度超出范围：{} mm，合法范围 [0, +inf) mm", target_height)
        return
    target_height_mm = int(round(target_height))
    set_result = lift.set_lift_physical_height(target_height_mm)
    if set_result is None or not set_result[0]:
        raise RuntimeError(f"升降柱目标高度设置失败：{set_result!r}")
    logger.info("已下发升降柱目标高度 {} mm", target_height_mm)
    time.sleep(LIFT_WAIT_S)
    _log_lift_height(lift)


def _control_waist(body_client: WujiBodyClient) -> None:
    """控制腰部俯仰。"""

    waist = body_client.waist
    waist.set_enable(True)
    if not waist.get_enable():
        raise RuntimeError("腰部使能失败")
    waist_pitch_deg = waist.get_waist_pitch()
    if waist_pitch_deg is None:
        raise RuntimeError("腰部俯仰角读取失败")
    logger.info("腰部当前使能 {}", waist.get_enable())
    logger.info("腰部当前俯仰角 {:.1f} deg", waist_pitch_deg)

    value = input("请输入腰部俯仰角度（deg），输入 q 返回: ").strip().lower()
    if value == "q":
        return
    try:
        target_pitch_deg = _parse_float(value, "腰部俯仰角度")
    except ValueError as exc:
        logger.warning("{}", exc)
        return
    if not waist.set_waist_pitch(target_pitch_deg):
        raise RuntimeError("腰部目标俯仰角设置失败")
    logger.info("已下发腰部目标俯仰角 {:.1f} deg", target_pitch_deg)

    waist_pitch_deg = waist.get_waist_pitch()
    if waist_pitch_deg is None:
        raise RuntimeError("腰部俯仰角读取失败")
    logger.info("腰部当前俯仰角 {:.1f} deg", waist_pitch_deg)


# endregion


# region 工具


def _parse_float(value: str, label: str) -> float:
    """解析数值输入。"""

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} 输入不是有效数字: {value}") from exc


def _log_lift_height(lift: WujiLiftClient) -> None:
    """读取并记录升降柱的物理高度与比例。"""

    physical_height_result = lift.get_lift_physical_height()
    if physical_height_result is None:
        raise RuntimeError("升降柱物理高度读取失败")
    physical_height_mm, _ = physical_height_result

    scale_result = lift.get_lift_height()
    if scale_result is None:
        raise RuntimeError("升降柱比例读取失败")
    scale, _ = scale_result
    logger.info("升降柱当前物理高度 {:.1f} mm", physical_height_mm)
    logger.info("升降柱当前高度比例 {:.3f}", scale)


# endregion


if __name__ == "__main__":
    main()
