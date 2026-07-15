"""双臂自动回放业务层的无硬件冒烟验证。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business.record_replay.csv_repository import state_name_from_left_csv
from src.business.record_replay.context import ReplayContext
from src.business.record_replay.contracts import CsvExecutionPlan, ReplayServiceState
from src.business.record_replay.settings import (
    OffsetConfig,
    ReplayCycleConfig,
    ReplayDeviceConnection,
    ReplayNetworkSettings,
    ReplayOffsetSettings,
)
from src.business.record_replay.cycle_service import RecordReplayCycleService
from src.business.record_replay.dual_arm_executor import DualArmExecutor
from src.business.record_replay.execution_plan import build_execution_plans
from src.business.record_replay.offset_updater import GlobalOffsetUpdater
from src.business.record_replay.offset_detection import build_three_ball_basis_transform, robust_mean_three_ball_centers
from src.business.record_replay.offset_detector_gateway import load_three_ball_priors
from src.business.record_replay.offset_math import calculate_global_offset


# region 默认数据

DEFAULT_LEFT_PATHS = (Path("01_left.csv"), Path("03_S02_left.csv"), Path("04_left.csv"))
DEFAULT_RIGHT_PATHS = (Path("01_right.csv"), Path("02_right.csv"), Path("03_right.csv"))


# endregion


# region 验证


class _FakeAgvClient:
    """无硬件循环服务验证使用的最小 AGV 替身。"""

    def __init__(self) -> None:
        self.targets: list[str] = []

    def navigate_to(self, target_name: str) -> None:
        """记录导航目标。"""

        self.targets.append(target_name)

    def get_runtime_info(self) -> dict[str, object]:
        """每次查询均模拟 AGV 已到位。"""

        return {"agv_navi_status": "arrived"}


class _FakeDualArmExecutor(DualArmExecutor):
    """无硬件循环服务验证使用的双臂执行器替身。"""

    def __init__(self) -> None:
        self.executed_plan_count = 0
        self.observed_snapshot = None

    def execute(
        self,
        context: ReplayContext,
        plans: list[CsvExecutionPlan],
        offset_updater: GlobalOffsetUpdater | None = None,
    ) -> None:
        """记录计划数量，并模拟执行器写入当前左臂状态。"""

        del offset_updater
        self.executed_plan_count = len(plans)
        context.set_state(ReplayServiceState.REPLAYING, left_csv_state="pick", plan_index=0)
        self.observed_snapshot = context.snapshot()


def verify_execution_plan() -> None:
    """验证左/右 CSV 的同步计划边界。"""

    plans = build_execution_plans(list(DEFAULT_LEFT_PATHS), list(DEFAULT_RIGHT_PATHS))
    if len(plans) != 3:
        raise AssertionError(f"计划数量异常: {len(plans)}")
    if plans[0].right_start_csv_path != Path("01_right.csv"):
        raise AssertionError("启动同步右臂 CSV 不一致")
    if plans[1].right_sync_csv_path != Path("02_right.csv"):
        raise AssertionError("S02 同步右臂 CSV 不一致")
    logger.success("CSV 执行计划验证通过")


def verify_offset_math() -> None:
    """验证三球坐标系、鲁棒均值和单位矩阵 offset。"""

    centers_mm = np.asarray(((0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (0.0, 10.0, 100.0)), dtype=np.float64)
    basis = build_three_ball_basis_transform(centers_mm)
    if basis is None or not np.allclose(basis[:3, 3], centers_mm[0]):
        raise AssertionError("三球坐标系构造失败")
    offset_settings = ReplayOffsetSettings()
    mean = robust_mean_three_ball_centers([centers_mm, centers_mm, centers_mm + 1000.0], offset_settings)
    if not np.allclose(mean, centers_mm):
        raise AssertionError("MAD 异常值剔除失败")
    identity = np.eye(4, dtype=np.float64)
    offset = calculate_global_offset(identity, identity, identity, identity)
    if not np.allclose(offset, identity):
        raise AssertionError("单位矩阵 offset 计算失败")
    logger.success("Offset 数学验证通过")


def verify_three_ball_prior_loading() -> None:
    """验证旧 `balls.ballinfo` 与 `detections` 先验文件均能生成同一模型。"""

    balls = [
        {"color_hex": "#ffff00", "radius_mm": 20.0, "position_camera_mm": [0.0, 0.0, 0.0]},
        {"color_hex": "#ff0000", "radius_mm": 20.0, "position_camera_mm": [10.0, 0.0, 0.0]},
        {"color_hex": "#ff00ff", "radius_mm": 20.0, "position_camera_mm": [0.0, 10.0, 0.0]},
    ]
    with tempfile.TemporaryDirectory() as directory:
        old_schema_path = Path(directory) / "old.json"
        detection_schema_path = Path(directory) / "detection.json"
        old_schema_path.write_text(json.dumps({"balls": {"ballinfo": balls}}), encoding="utf-8")
        detection_schema_path.write_text(json.dumps({"detections": balls}), encoding="utf-8")
        offset_settings = ReplayOffsetSettings()
        old_priors = load_three_ball_priors(old_schema_path, offset_settings)
        detection_priors = load_three_ball_priors(detection_schema_path, offset_settings)
    if old_priors != detection_priors:
        raise AssertionError(f"两种先验 schema 解析结果不一致: {old_priors}, {detection_priors}")
    if old_priors[1].model_center_mm != (10.0, 0.0, 0.0):
        raise AssertionError(f"红球模型坐标异常: {old_priors[1]}")
    logger.success("三球先验 schema 兼容验证通过")


def verify_state_name() -> None:
    """验证左臂 CSV 状态名去前缀规则。"""

    if state_name_from_left_csv("left_pick.csv", "left_") != "pick":
        raise AssertionError("状态名前缀去除失败")
    logger.success("状态名验证通过")


def verify_orin_network_defaults() -> None:
    """验证 service 在 Orin 默认直连固定的 qmlinker、ZMQ 与 AGV 地址。"""

    connection = ReplayDeviceConnection("192.0.2.11", "192.0.2.12")
    if connection.qmlinker_host != "192.168.100.60" or connection.qmlinker_port != 50062:
        raise AssertionError(f"service qmlinker Orin 默认地址异常: {connection}")
    if connection.gripper_port != 50066:
        raise AssertionError(f"service 夹爪默认端口异常: {connection}")
    network = ReplayNetworkSettings()
    if network.zmq_host != "192.168.100.60" or network.agv_host != "192.168.100.70":
        raise AssertionError(f"Orin ZMQ/AGV 默认地址异常: {network}")
    if network.local_service_host != "127.0.0.1":
        raise AssertionError(f"本机 service 默认地址异常: {network}")
    offset_config = OffsetConfig(Path("prior.json"), Path("hand_eye.txt"))
    if offset_config.service_addr != "tcp://192.168.100.60:6200":
        raise AssertionError(f"Orin ZMQ 默认 service 地址异常: {offset_config}")
    logger.success("Orin 与本机网络默认配置验证通过")


def verify_cycle_service() -> None:
    """验证 AGV 到 3、回放、返回 1、等待下一轮的服务时序。"""

    with tempfile.TemporaryDirectory() as directory:
        left_dir = Path(directory) / "left"
        right_dir = Path(directory) / "right"
        left_dir.mkdir()
        right_dir.mkdir()
        (left_dir / "01_left.csv").write_text("type,joints,pose\n", encoding="utf-8")
        device_connection = ReplayDeviceConnection("192.0.2.11", "192.0.2.12", "192.0.2.13", 50062, 50066)
        context = ReplayContext(ReplayCycleConfig(left_dir, right_dir, device_connection))
        agv = _FakeAgvClient()
        executor = _FakeDualArmExecutor()
        service = RecordReplayCycleService(context, agv, executor=executor)
        service.run_once()
        if agv.targets != ["3", "1"]:
            raise AssertionError(f"AGV 导航顺序异常: {agv.targets}")
        if executor.executed_plan_count != 1:
            raise AssertionError(f"执行计划数量异常: {executor.executed_plan_count}")
        if executor.observed_snapshot is None:
            raise AssertionError("执行器未发布执行期状态快照")
        if executor.observed_snapshot.state is not ReplayServiceState.REPLAYING:
            raise AssertionError(f"执行期服务状态异常: {executor.observed_snapshot}")
        if executor.observed_snapshot.left_csv_state != "pick":
            raise AssertionError(f"执行期左臂 CSV 状态异常: {executor.observed_snapshot}")
        if context.snapshot().state is not ReplayServiceState.WAITING:
            raise AssertionError(f"服务完成后状态异常: {context.snapshot()}")
    logger.success("AGV-双臂循环服务时序验证通过")


# endregion


def main() -> None:
    """运行全部无硬件业务层验证。"""

    verify_execution_plan()
    verify_offset_math()
    verify_three_ball_prior_loading()
    verify_state_name()
    verify_orin_network_defaults()
    verify_cycle_service()
    logger.success("双臂回放业务层无硬件冒烟通过")


if __name__ == "__main__":
    main()
