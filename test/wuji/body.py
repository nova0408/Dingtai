from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.body_client import WujiBodyClient

LIFT_TARGET_HEIGHT_MM = 380  # 升降柱冒烟测试目标高度，单位 mm。
LIFT_TIMEOUT_S = 15.0  # 升降柱到位超时时间，单位 s。
LIFT_POLL_INTERVAL_S = 0.5  # 升降柱高度轮询间隔，单位 s。
LIFT_TOLERANCE_MM = 10.0  # 升降柱目标高度允许误差，单位 mm。


def main(
    lift_target_height_mm: int = LIFT_TARGET_HEIGHT_MM,
    lift_timeout_s: float = LIFT_TIMEOUT_S,
    lift_poll_interval_s: float = LIFT_POLL_INTERVAL_S,
    lift_tolerance_mm: float = LIFT_TOLERANCE_MM,
) -> None:
    """验证升降柱与腰部的基础控制链路。"""

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    body_client = WujiBodyClient(qmlinker_channel)
    try:
        lift = body_client.lift
        lift.set_enable(True)
        if not lift.get_enable():
            raise RuntimeError("升降柱使能失败")
        logger.info("升降柱使能状态 {}", lift.get_enable())

        set_result = lift.set_lift_physical_height(lift_target_height_mm)
        if set_result is None or not set_result[0]:
            raise RuntimeError(f"升降柱目标高度设置失败：{set_result!r}")

        deadline = time.monotonic() + lift_timeout_s
        while True:
            height_result = lift.get_lift_physical_height()
            if height_result is None:
                raise RuntimeError("升降柱高度读取失败")
            height_mm, _ = height_result
            if abs(height_mm - lift_target_height_mm) <= lift_tolerance_mm:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"升降柱到位超时：目标 {lift_target_height_mm:.1f} mm，当前 {height_mm:.1f} mm"
                )
            time.sleep(lift_poll_interval_s)
        logger.info("升降柱当前高度 {:.1f} mm", height_mm)

        waist = body_client.waist
        waist.set_enable(True)
        if not waist.get_enable():
            raise RuntimeError("腰部使能失败")
        waist_pitch_deg = waist.get_waist_pitch()
        if waist_pitch_deg is None:
            raise RuntimeError("腰部俯仰角读取失败")
        logger.info("腰部当前俯仰角 {:.1f} deg", waist_pitch_deg)

        logger.success("无际身体冒烟测试通过")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
