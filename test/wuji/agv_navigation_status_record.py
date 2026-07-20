from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger
from qmlinker import create_channel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT
from network_discovery import get_cached_orin_host

from src.wuji.agv_client import WujiAgvClient
from src.wuji.qmlinker_session import WujiQmlinkerSession

# region 默认参数
DEFAULT_TARGET_NAME = "3"
"navigate_to 使用的目标站点名称。"

DEFAULT_QMLINKER_HOST = get_cached_orin_host()
"Qmlinker 服务主机地址。"

DEFAULT_SSH_ALIAS = "orin"
"建立 Qmlinker SSH 转发使用的主机别名。"

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "wuji" / ".archive" / "agv_navigation_status_record"
"AGV 导航状态记录输出根目录。"

DEFAULT_PRE_NAVIGATION_SAMPLE_COUNT = 3
"调用 navigate_to 前连续读取的状态数量。"

DEFAULT_PRE_NAVIGATION_SAMPLE_INTERVAL_S = 0.2
"调用 navigate_to 前状态采样间隔，单位 s。"

DEFAULT_POST_NAVIGATION_POLL_INTERVAL_S = 0.2
"调用 navigate_to 后连续状态采样间隔，单位 s。"

DEFAULT_POST_NAVIGATION_TIMEOUT_S = 120.0
"调用 navigate_to 后最大连续记录时间，单位 s。"

DEFAULT_IDLE_STABLE_SAMPLE_COUNT = 3
"观察到运动后，判定最终空闲或到位所需的连续状态数量。"

AGV_IDLE_OR_ARRIVED_STATUSES = frozenset(
    {"idel"}
)
"明确表示 AGV 空闲、成功或运动到位的原始状态值。"

AGV_MOVING_STATUSES = frozenset(
    {"busy"}
)
"可能表示 AGV 正在运动或执行导航的原始状态值。"
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class AgvStatusSample:
    """单次 AGV 运行状态采样。"""

    sample_index: int
    timestamp_iso: str
    elapsed_s: float
    phase: str
    raw_navigation_status: str
    status_class: str
    agv_x: float
    agv_y: float
    agv_yaw: float
    agv_battery: float


# endregion


# region 状态采样
def _classify_navigation_status(raw_status: object) -> tuple[str, str]:
    normalized_status = str(raw_status).strip().lower()
    if normalized_status in AGV_IDLE_OR_ARRIVED_STATUSES:
        return normalized_status, "idle_or_arrived"
    if normalized_status in AGV_MOVING_STATUSES:
        return normalized_status, "moving"
    return normalized_status, "unknown"


def _runtime_float(runtime_info: dict[str, object], field_name: str) -> float:
    value = runtime_info.get(field_name, 0.0)
    if not isinstance(value, int | float):
        raise TypeError(f"AGV 运行时字段不是数值：field={field_name}, value={value!r}")
    return float(value)


def _capture_status_sample(
    *,
    client: WujiAgvClient,
    sample_index: int,
    phase: str,
    started_at: float,
) -> AgvStatusSample:
    runtime_info = client.get_runtime_info()
    raw_status, status_class = _classify_navigation_status(
        runtime_info.get("agv_navi_status", "")
    )
    sample = AgvStatusSample(
        sample_index=sample_index,
        timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        elapsed_s=time.monotonic() - started_at,
        phase=phase,
        raw_navigation_status=raw_status,
        status_class=status_class,
        agv_x=_runtime_float(runtime_info, "agv_x"),
        agv_y=_runtime_float(runtime_info, "agv_y"),
        agv_yaw=_runtime_float(runtime_info, "agv_yaw"),
        agv_battery=_runtime_float(runtime_info, "agv_battery"),
    )
    logger.info(
        "AGV 状态 sample={} phase={} raw_status={!r} class={} "
        "x={:.4f} m y={:.4f} m yaw={:.3f} deg battery={:.1f} %",
        sample.sample_index,
        sample.phase,
        sample.raw_navigation_status,
        sample.status_class,
        sample.agv_x,
        sample.agv_y,
        sample.agv_yaw,
        sample.agv_battery,
    )
    return sample


def _record_before_navigation(
    client: WujiAgvClient,
    samples: list[AgvStatusSample],
    started_at: float,
) -> None:
    for _ in range(DEFAULT_PRE_NAVIGATION_SAMPLE_COUNT):
        samples.append(
            _capture_status_sample(
                client=client,
                sample_index=len(samples) + 1,
                phase="before_navigate_to",
                started_at=started_at,
            )
        )
        time.sleep(DEFAULT_PRE_NAVIGATION_SAMPLE_INTERVAL_S)


def _record_after_navigation(
    client: WujiAgvClient,
    samples: list[AgvStatusSample],
    started_at: float,
) -> tuple[bool, bool]:
    deadline = time.monotonic() + DEFAULT_POST_NAVIGATION_TIMEOUT_S
    observed_moving = False
    stable_idle_count = 0
    while time.monotonic() < deadline:
        sample = _capture_status_sample(
            client=client,
            sample_index=len(samples) + 1,
            phase="after_navigate_to",
            started_at=started_at,
        )
        samples.append(sample)
        if sample.status_class == "moving":
            observed_moving = True
            stable_idle_count = 0
        elif observed_moving and sample.status_class == "idle_or_arrived":
            stable_idle_count += 1
            if stable_idle_count >= DEFAULT_IDLE_STABLE_SAMPLE_COUNT:
                logger.success(
                    "AGV 已从运动状态进入稳定空闲/到位状态，连续样本数={}",
                    stable_idle_count,
                )
                return True, True
        else:
            stable_idle_count = 0
        time.sleep(DEFAULT_POST_NAVIGATION_POLL_INTERVAL_S)
    logger.warning(
        "AGV 状态连续记录超时 timeout={:.1f} s observed_moving={} stable_idle_count={}",
        DEFAULT_POST_NAVIGATION_TIMEOUT_S,
        observed_moving,
        stable_idle_count,
    )
    return observed_moving, False


# endregion


# region 结果保存
def _create_session_dir() -> Path:
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _save_samples(
    *,
    session_dir: Path,
    samples: list[AgvStatusSample],
    observed_moving: bool,
    observed_final_idle: bool,
) -> None:
    csv_path = session_dir / "agv_navigation_status.csv"
    fieldnames = list(AgvStatusSample.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)

    before_statuses = [
        sample.raw_navigation_status
        for sample in samples
        if sample.phase == "before_navigate_to"
    ]
    after_statuses = [
        sample.raw_navigation_status
        for sample in samples
        if sample.phase == "after_navigate_to"
    ]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_name": DEFAULT_TARGET_NAME,
        "sample_count": len(samples),
        "observed_moving": observed_moving,
        "observed_final_idle_or_arrived": observed_final_idle,
        "before_raw_statuses": before_statuses,
        "after_raw_statuses": after_statuses,
        "observed_raw_statuses": sorted(
            {sample.raw_navigation_status for sample in samples}
        ),
        "status_classification": {
            "idle_or_arrived": sorted(AGV_IDLE_OR_ARRIVED_STATUSES),
            "moving": sorted(AGV_MOVING_STATUSES),
            "unmatched_values": "unknown",
        },
    }
    summary_path = session_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success("AGV 状态记录已保存：{}", csv_path)
    logger.success("AGV 状态摘要已保存：{}", summary_path)


# endregion


# region 主流程
def main() -> None:
    session_dir = _create_session_dir()
    session = WujiQmlinkerSession(
        host=DEFAULT_QMLINKER_HOST,
        port=DEFAULT_PORT,
        ssh_alias=DEFAULT_SSH_ALIAS,
    )
    samples: list[AgvStatusSample] = []
    started_at = time.monotonic()
    observed_moving = False
    observed_final_idle = False
    try:
        session.check_ready()
        client = WujiAgvClient(
            create_channel(session.move_base_target),
            request_timeout_s=session.request_timeout_s,
        )
        logger.info("开始记录 navigate_to 前 AGV 状态，target={}", DEFAULT_TARGET_NAME)
        _record_before_navigation(client, samples, started_at)
        logger.warning("即将执行 AGV navigate_to target={}", DEFAULT_TARGET_NAME)
        client.navigate_to(DEFAULT_TARGET_NAME)
        logger.success("navigate_to 已下发，开始连续记录执行后状态")
        observed_moving, observed_final_idle = _record_after_navigation(
            client,
            samples,
            started_at,
        )
    finally:
        session.close()
        _save_samples(
            session_dir=session_dir,
            samples=samples,
            observed_moving=observed_moving,
            observed_final_idle=observed_final_idle,
        )
    if not observed_moving:
        logger.warning("本次记录未观察到明确的 moving 状态，请检查原始状态值并补充分类常量")
    if observed_moving and not observed_final_idle:
        logger.warning("已观察到运动状态，但未在超时前观察到稳定空闲/到位状态")


# endregion


if __name__ == "__main__":
    main()
