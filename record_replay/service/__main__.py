"""Orin RecordReplay 常驻服务入口。"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from loguru import logger

from camera_pipeline.client import CameraName
from src.wuji import WujiAgvClient
from src.wuji.qmlinker_session import create_qmlinker_channel

from ..context import ReplayContext
from ..cycle_service import RecordReplayCycleService
from ..offset_detector_gateway import CameraPipelineThreeBallDetector, load_three_ball_priors
from ..offset_updater import GlobalOffsetUpdater
from ..settings import OffsetConfig, ReplayCycleConfig, ReplayDeviceConnection
from .application import RecordReplayApplication
from .config_store import RuntimeConfigStore
from .server import RecordReplayServer

SERVICE_ROOT = Path(__file__).resolve().parents[1]
"record_replay 服务根目录。"

PRIOR_DATA_DIR = SERVICE_ROOT / "prior_data"
"先验记录固定目录。"

LEFT_RECORD_DIR = SERVICE_ROOT / "records" / "left"
"左臂预录 CSV 固定目录。"

RIGHT_RECORD_DIR = SERVICE_ROOT / "records" / "right"
"右臂预录 CSV 固定目录。"

BALL_POSE_PRIOR_PATH = PRIOR_DATA_DIR / "ball_pose_prior.json"
"prior_record.py 生成的三球先验固定路径。"

RUNTIME_CONFIG_PATH = SERVICE_ROOT / "config.json"
"本机 API 修改后持久化的运行参数路径。"

LEFT_ARM_IP = "192.168.100.161"
RIGHT_ARM_IP = "192.168.100.160"
QMLINKER_HOST = "192.168.100.60"
QMLINKER_PORT = 50062
GRIPPER_PORT = 50066
AGV_HOST = "192.168.100.70"


class _OrinAgvGateway:
    """收窄 WujiAgvClient 为回放服务需要的显式 AGV 协议。"""

    def __init__(self, client: WujiAgvClient) -> None:
        self._client = client

    def navigate_to(self, target_name: str) -> object:
        """下发目标站点名称。"""

        return self._client.navigate_to(target_name)

    def get_runtime_info(self) -> dict[str, object]:
        """读取 AGV 导航状态。"""

        return self._client.get_runtime_info()


def main(argv: Sequence[str] | None = None) -> int:
    """启动只由 API 触发动作的 Orin 常驻服务。"""

    args = _parse_args(argv)
    config_store = RuntimeConfigStore(RUNTIME_CONFIG_PATH)
    settings = config_store.load().to_service_settings()
    device_connection = ReplayDeviceConnection(
        left_arm_ip=LEFT_ARM_IP,
        right_arm_ip=RIGHT_ARM_IP,
        qmlinker_host=QMLINKER_HOST,
        qmlinker_port=QMLINKER_PORT,
        gripper_port=GRIPPER_PORT,
    )
    context = ReplayContext(
        ReplayCycleConfig(
            left_record_dir=LEFT_RECORD_DIR,
            right_record_dir=RIGHT_RECORD_DIR,
            device_connection=device_connection,
            settings=settings,
            start_station=args.start_station,
            finish_station=args.finish_station,
        )
    )
    agv_channel = create_qmlinker_channel(f"{AGV_HOST}:{QMLINKER_PORT}")
    agv_client = _OrinAgvGateway(WujiAgvClient(agv_channel))
    offset_config = OffsetConfig(
        prior_capture_path=BALL_POSE_PRIOR_PATH,
        hand_eye_result_path=args.hand_eye_result_path,
        camera_name=args.camera_name,
    )
    detector = CameraPipelineThreeBallDetector(
        CameraName(offset_config.camera_name),
        load_three_ball_priors(offset_config.prior_capture_path, settings.offset),
        settings.offset,
    )
    cycle_service = RecordReplayCycleService(
        context,
        agv_client,
        offset_updater=GlobalOffsetUpdater(offset_config, detector),
    )
    application = RecordReplayApplication(context, cycle_service, config_store)
    server = RecordReplayServer(args.host, args.port, application)

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.warning("record replay service received stop signal={}", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info("record replay service started http://{}:{}", args.host, args.port)
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("record replay service stopping")
    finally:
        server.close()
        application.join()
        logger.info("record replay service stopped")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析 Orin 现场路径与设备地址。"""

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Dingtai RecordReplay service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6300)
    parser.add_argument("--start-station", default="3")
    parser.add_argument("--finish-station", default="1")
    parser.add_argument("--camera-name", default="left_hand_camera")
    parser.add_argument(
        "--hand-eye-result-path",
        type=Path,
        default=project_root / "experiments" / "hand_eye" / "runs" / "20260708_152829" / "hand_eye_result.txt",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
