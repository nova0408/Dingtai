"""Orin RecordReplay 常驻服务入口。"""

from __future__ import annotations

import argparse
import signal
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from loguru import logger
from qmlinker import QMGripper, QMHead, QMLift, QMMoveBase, create_channel

from camera_pipeline.client import CameraName

from ..context import ReplayContext
from ..cycle_service import RecordReplayCycleService
from ..device_status import DeviceStatusReader
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

HAND_EYE_RESULT_PATH = PRIOR_DATA_DIR / "hand_eye_result.txt"
"当前现场手眼标定结果的固定路径。"

RUNTIME_CONFIG_PATH = SERVICE_ROOT / "config.json"
"本机 API 修改后持久化的运行参数路径。"

LEFT_ARM_IP = "192.168.100.161"
RIGHT_ARM_IP = "192.168.100.160"
QMLINKER_HOST = "192.168.100.60"
QMLINKER_PORT = 50062
GRIPPER_PORT = 50066
AGV_HOST = "192.168.100.70"


def main(argv: Sequence[str] | None = None) -> int:
    """启动只由 API 触发动作的 Orin 常驻服务。"""

    startup_started_at = time.perf_counter()
    args = _parse_args(argv)
    logger.info(
        "record replay service initializing host={} port={} prior_path={} config_path={}",
        args.host,
        args.port,
        BALL_POSE_PRIOR_PATH,
        RUNTIME_CONFIG_PATH,
    )
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
    agv_channel = create_channel(f"{AGV_HOST}:{QMLINKER_PORT}")
    agv_client = QMMoveBase(agv_channel)
    qmlinker_channel = create_channel(f"{QMLINKER_HOST}:{QMLINKER_PORT}")
    gripper_channel = create_channel(f"{QMLINKER_HOST}:{GRIPPER_PORT}")
    device_status_reader = DeviceStatusReader(
        device_connection,
        settings,
        QMGripper(gripper_channel),
        QMHead(qmlinker_channel),
        QMLift(qmlinker_channel),
    )
    offset_config = OffsetConfig(
        prior_capture_path=BALL_POSE_PRIOR_PATH,
        hand_eye_result_path=HAND_EYE_RESULT_PATH,
        camera_name=args.camera_name,
    )
    prior_load_started_at = time.perf_counter()
    ball_priors = load_three_ball_priors(
        offset_config.prior_capture_path,
    )
    logger.info(
        "record replay ball priors loaded count={} elapsed_ms={:.3f}",
        len(ball_priors),
        (time.perf_counter() - prior_load_started_at) * 1000.0,
    )
    detector = CameraPipelineThreeBallDetector(
        CameraName(offset_config.camera_name),
        ball_priors,
        settings.offset,
    )
    cycle_service = RecordReplayCycleService(
        context,
        agv_client,
        offset_updater=GlobalOffsetUpdater(offset_config, detector),
    )
    application = RecordReplayApplication(
        context,
        cycle_service,
        config_store,
        device_status_reader,
    )
    server = RecordReplayServer(args.host, args.port, application)

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.warning("record replay service received stop signal={}", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info(
        "record replay service started http://{}:{} startup_elapsed_ms={:.3f}",
        args.host,
        args.port,
        (time.perf_counter() - startup_started_at) * 1000.0,
    )
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("record replay service stopping")
    finally:
        shutdown_started_at = time.perf_counter()
        server.close()
        logger.info("record replay HTTP server socket closed")
        logger.info("record replay waiting for active worker to finish")
        application.join()
        logger.info(
            "record replay service stopped shutdown_elapsed_ms={:.3f}",
            (time.perf_counter() - shutdown_started_at) * 1000.0,
        )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析 Orin 现场路径与设备地址。"""

    parser = argparse.ArgumentParser(description="Dingtai RecordReplay service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6300)
    parser.add_argument("--start-station", default="3")
    parser.add_argument("--finish-station", default="1")
    parser.add_argument("--camera-name", default="left_hand_camera")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
