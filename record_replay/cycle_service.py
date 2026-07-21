"""AGV 导航与双臂回放的常态化循环服务。"""

from __future__ import annotations

import time

from .agv_navigation import AgvClient, wait_until_arrived
from .context import ReplayContext
from .contracts import CsvExecutionPlan, ReplayServiceState
from .csv_repository import discover_csv_paths, extract_sync_csv_sequence
from .dual_arm_executor import DualArmExecutor
from .execution_plan import build_execution_plans
from .offset_updater import GlobalOffsetUpdater


class RecordReplayCycleService:
    """按导航、回放、返回顺序执行的 Linux 常态化服务。"""

    def __init__(
        self,
        context: ReplayContext,
        agv_client: AgvClient,
        executor: DualArmExecutor | None = None,
        offset_updater: GlobalOffsetUpdater | None = None,
    ) -> None:
        self._context = context
        self._agv_client = agv_client
        self._executor = executor if executor is not None else DualArmExecutor()
        self._offset_updater = offset_updater

    def run_once(self, enable_agv_navigation: bool = True) -> None:
        """执行一轮可选 AGV 导航、双臂回放与返回流程。"""

        try:
            config = self._context.config
            if enable_agv_navigation:
                self._context.set_state(ReplayServiceState.NAVIGATING_TO_START)
                wait_until_arrived(
                    self._agv_client,
                    config.start_station,
                    timeout_s=config.settings.agv_navigation_timeout_s,
                    poll_s=config.settings.agv_navigation_poll_interval_s,
                )
            left_paths = discover_csv_paths(config.left_record_dir)
            if not left_paths:
                raise RuntimeError(f"没有在目录中发现 CSV: {config.left_record_dir}")
            right_paths = discover_csv_paths(config.right_record_dir) if any(extract_sync_csv_sequence(item.name) is not None for item in left_paths) else []
            plans = build_execution_plans(left_paths, right_paths) if right_paths else [CsvExecutionPlan(item) for item in left_paths]
            self._executor.execute(self._context, plans, self._offset_updater)
            if enable_agv_navigation:
                self._context.set_state(ReplayServiceState.NAVIGATING_TO_FINISH)
                wait_until_arrived(
                    self._agv_client,
                    config.finish_station,
                    timeout_s=config.settings.agv_navigation_timeout_s,
                    poll_s=config.settings.agv_navigation_poll_interval_s,
                )
            self._context.reset_for_next_cycle()
        except Exception as error:
            self._context.set_state(ReplayServiceState.FAILED, error_text=str(error))
            raise

    def run_forever(self) -> None:
        """等待触发文件存在后执行下一轮。"""

        trigger_file = self._context.config.trigger_file
        if trigger_file is None:
            raise ValueError("常态化服务必须配置 trigger_file")
        while True:
            if trigger_file.exists():
                trigger_file.unlink()
                self.run_once()
            else:
                time.sleep(self._context.config.settings.trigger_poll_interval_s)
