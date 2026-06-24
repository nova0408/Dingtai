from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_orin_channel, stop_ssh_process  # noqa: E402
from src.wuji.body_client import WujiBodyClient  # noqa: E402
from qmlinker.grpc_py import lift_pb2  # noqa: E402

def _read_lift_height_mm(result: object) -> float:
    """将 lift 读取结果转换成毫米。"""

    if isinstance(result, tuple) and len(result) == 2:
        return float(result[0])
    return float(result)


def _set_lift_physical_height_direct(lift: object, height_mm: int, timeout_s: float = 3.0) -> tuple[bool, str]:
    """按旧版实现直接调用 stub 设置升降物理高度。"""

    request = lift_pb2.SetLiftPhysicalHeightRequest(height_mm=int(height_mm))
    response = lift.stub.SetLiftPhysicalHeight(request, timeout=timeout_s)
    return bool(response.status.success), str(response.status.message)


def main() -> None:
    """读取身体当前状态。"""

    ssh_process, qmlinker_channel = create_orin_channel(DEFAULT_PORT)
    body_client = WujiBodyClient(qmlinker_channel)
    try:
        logger.info("身体冒烟测试")

        lift = body_client.lift
        logger.info("lift 初始使能 {}", lift.get_enable())
        lift.set_enable(False)
        logger.info("lift 关闭后使能 {}", lift.get_enable())
        lift.set_enable(True)
        logger.info("lift 打开后使能 {}", lift.get_enable())
        lift_height_raw = lift.get_lift_physical_height()
        logger.info("lift 原始返回 {}", lift_height_raw)
        logger.info("lift 当前高度 {:.1f} mm", _read_lift_height_mm(lift_height_raw))

        test_scale = 0.38
        logger.info("lift set_lift_height(scale) 测试目标 {:.3f}", test_scale)
        scale_result = lift.set_lift_height(test_scale)
        logger.info("lift set_lift_height 返回 {}", scale_result)
        time.sleep(2.0)
        lift_height_raw = lift.get_lift_physical_height()
        logger.info("lift set_lift_height 后原始返回 {}", lift_height_raw)
        logger.info("lift set_lift_height 后高度 {:.1f} mm", _read_lift_height_mm(lift_height_raw))

        test_height_mm = 380
        logger.info("lift set_lift_physical_height 测试目标 {} mm", test_height_mm)
        physical_result = lift.set_lift_physical_height(test_height_mm)
        logger.info("lift set_lift_physical_height 返回 {}", physical_result)
        time.sleep(2.0)
        lift_height_raw = lift.get_lift_physical_height()
        logger.info("lift set_lift_physical_height 后原始返回 {}", lift_height_raw)
        logger.info("lift set_lift_physical_height 后高度 {:.1f} mm", _read_lift_height_mm(lift_height_raw))

        test_height_mm_direct = 400
        logger.info("lift 旧版直连 stub 测试目标 {} mm", test_height_mm_direct)
        direct_success, direct_message = _set_lift_physical_height_direct(lift, test_height_mm_direct)
        logger.info("lift 旧版直连 stub 返回 success={} message={}", direct_success, direct_message)
        time.sleep(2.0)
        lift_height_raw = lift.get_lift_physical_height()
        logger.info("lift 旧版直连 stub 后原始返回 {}", lift_height_raw)
        logger.info("lift 旧版直连 stub 后高度 {:.1f} mm", _read_lift_height_mm(lift_height_raw))

        waist = body_client.waist
        logger.info("waist 初始使能 {}", waist.get_enable())
        waist.set_enable(False)
        logger.info("waist 关闭后使能 {}", waist.get_enable())
        waist.set_enable(True)
        logger.info("waist 打开后使能 {}", waist.get_enable())
        logger.info("waist 当前俯仰 {} deg", waist.get_waist_pitch())

        logger.success("无际身体信息冒烟通过")
    finally:
        stop_ssh_process(ssh_process)
        logger.info("无际身体信息冒烟结束")
        os._exit(0)


if __name__ == "__main__":
    main()
