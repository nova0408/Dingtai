"""AGV 导航与双臂回放的常态化循环服务。"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .action_sequence import ActionSequencePlan, NamedActionPlan, load_action_sequence
from .agv_navigation import AgvClient, stop_navigation, wait_until_arrived
from .arm_gateway import stop_arm
from .charuco_offset import CharucoOffsetInitializer
from .context import ReplayContext
from .contracts import ReplayCsvFileStatus, ReplayServiceState
from .dual_arm_executor import DualArmExecutor
from .execution_plan import build_execution_task_statuses
from .offset_updater import GlobalOffsetUpdater


class RecordReplayCycleService:
    """按导航、回放、返回顺序执行的 Linux 常态化服务。"""

    def __init__(
        self,
        context: ReplayContext,
        agv_client: AgvClient,
        executor: DualArmExecutor | None = None,
        offset_updater: GlobalOffsetUpdater | None = None,
        charuco_initializer: CharucoOffsetInitializer | None = None,
    ) -> None:
        self._context = context
        self._agv_client = agv_client
        self._executor = executor if executor is not None else DualArmExecutor()
        self._offset_updater = offset_updater
        self._charuco_initializer = charuco_initializer
        self._agv_command_lock = threading.Lock()

    def run_once(
        self,
        enable_agv_navigation: bool = True,
        plan: ActionSequencePlan | None = None,
    ) -> None:
        """执行一轮可选 AGV 导航、双臂回放与返回流程。"""

        try:
            if self._context.snapshot().state is ReplayServiceState.RAPID_STOP:
                raise RuntimeError("服务处于 rapid_stop，禁止直接执行回放")
            if self._context.stop_event.is_set():
                raise RuntimeError("检测到停止请求，禁止直接执行回放")
            config = self._context.config
            self._context.reset_run_progress()
            if plan is None:
                left_paths, plan = self.refresh_deployment_status()
            else:
                left_paths = [action.csv_asset.path for action in plan.left_actions]
            if enable_agv_navigation:
                self._context.set_state(ReplayServiceState.BUSY)
                wait_until_arrived(
                    self._agv_client,
                    config.start_station,
                    timeout_s=config.settings.agv_navigation_timeout_s,
                    poll_s=config.settings.agv_navigation_poll_interval_s,
                    stop_event=self._context.stop_event,
                    command_lock=self._agv_command_lock,
                )
            if not left_paths:
                raise RuntimeError(f"没有在目录中发现 CSV: {config.left_record_dir}")
            self._executor.execute(
                self._context,
                plan,
                self._offset_updater,
                self._charuco_initializer,
            )
            self._context.reset_for_next_cycle()
        except Exception as error:
            self._context.stop_event.set()
            stop_error_text: str | None = None
            if self._context.snapshot().state is not ReplayServiceState.RAPID_STOP:
                try:
                    self.stop_devices()
                except Exception as stop_error:
                    stop_error_text = f"{type(stop_error).__name__}: {stop_error}"
            self._context.reset_run_progress()
            error_text = str(error)
            if stop_error_text is not None:
                error_text = f"{error_text}；自动停止设备失败：{stop_error_text}"
            self._context.set_state(ReplayServiceState.RAPID_STOP, error_text=error_text)
            raise

    def refresh_deployment_status(
        self,
        plan: ActionSequencePlan | None = None,
    ) -> tuple[list[Path], ActionSequencePlan]:
        """读取命名动作 JSON，校验 CSV 并发布冻结前的部署摘要。"""

        config = self._context.config
        if plan is None:
            plan = load_action_sequence(
                config.action_sequence_path,
                config.left_record_dir,
                config.right_record_dir,
            )
        left_paths = _unique_action_csv_paths(plan.left_actions)
        right_paths = _unique_action_csv_paths(plan.right_actions)
        frozen_rows_by_path = dict(plan.preloaded_rows_by_path)
        left_files = tuple(
            ReplayCsvFileStatus(path.name, len(frozen_rows_by_path[path]))
            for path in left_paths
        )
        right_files = tuple(
            ReplayCsvFileStatus(path.name, len(frozen_rows_by_path[path]))
            for path in right_paths
        )
        self._context.set_deployment_status(
            left_files,
            right_files,
            build_execution_task_statuses(plan),
            plan.source_sha256,
        )
        return left_paths, plan

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

    def stop_devices(self) -> None:
        """并行停止 AGV、左 AR5 和右 AR5；一个失败不阻断其它调用。"""

        errors: list[BaseException] = []
        left_runtime = self._context.resources.left_runtime
        right_runtime = self._context.resources.right_runtime
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="record-replay-stop") as executor:
            futures = [
                executor.submit(
                    stop_navigation,
                    self._agv_client,
                    self._context.config.settings.agv_stop_timeout_s,
                    command_lock=self._agv_command_lock,
                )
            ]
            if left_runtime is not None:
                futures.append(executor.submit(stop_arm, left_runtime.connected_arm))
            if right_runtime is not None:
                futures.append(executor.submit(stop_arm, right_runtime.connected_arm))
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    errors.append(error)
        if errors:
            raise RuntimeError("设备停止调用未全部成功") from errors[0]


def _unique_action_csv_paths(actions: tuple[NamedActionPlan, ...]) -> list[Path]:
    """按 JSON 动作顺序返回去重后的已引用 CSV 路径。"""

    paths: list[Path] = []
    seen_paths: set[Path] = set()
    for action in actions:
        path = action.csv_asset.path
        if path in seen_paths:
            continue
        seen_paths.add(path)
        paths.append(path)
    return paths
