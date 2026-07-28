"""把 HTTP API 桥接到单实例回放业务线程和配置存储。"""

from __future__ import annotations

import threading

from loguru import logger

from ..context import ReplayContext
from ..cycle_service import RecordReplayCycleService
from ..device_status import DeviceStatusReader, DeviceStatusResponse
from .config_store import RuntimeConfigStore
from .protocol import RecordReplayResponse


class RecordReplayApplication:
    """管理单个硬件任务，并原子更新后续轮次使用的持久化参数。"""

    def __init__(
        self,
        context: ReplayContext,
        cycle_service: RecordReplayCycleService,
        config_store: RuntimeConfigStore,
        device_status_reader: DeviceStatusReader,
    ) -> None:
        self._context = context
        self._cycle_service = cycle_service
        self._config_store = config_store
        self._device_status_reader = device_status_reader
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._total_execution_count = 0
        self._cycle_service.refresh_deployment_status()

    def start(self, enable_agv_navigation: bool) -> RecordReplayResponse:
        """启动唯一业务线程；已有任务运行时拒绝重复启动。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status(accepted=False)
            self._worker = threading.Thread(
                target=self._run_once,
                args=(enable_agv_navigation,),
                name="record-replay-worker",
                daemon=False,
            )
            self._total_execution_count += 1
            self._worker.start()
        return self.status()

    def status(self, *, accepted: bool = True) -> RecordReplayResponse:
        """返回当前原子状态快照。"""

        snapshot = self._context.snapshot()
        current_task_sequence = (
            0
            if snapshot.current_task_index is None
            else snapshot.current_task_index + 1
        )
        return RecordReplayResponse(
            state=snapshot.state,
            accepted=accepted,
            left_csv_state=snapshot.left_csv_state,
            plan_index=snapshot.plan_index,
            error_text=snapshot.error_text,
            left_csv_files=snapshot.left_csv_files,
            right_csv_files=snapshot.right_csv_files,
            execution_tasks=snapshot.execution_tasks,
            current_task_sequence=current_task_sequence,
            current_task_active=snapshot.current_task_active,
            total_execution_count=self._total_execution_count,
            current_left_csv=snapshot.current_left_csv,
            current_right_csv=snapshot.current_right_csv,
            current_left_row=snapshot.current_left_row,
            current_right_row=snapshot.current_right_row,
            current_left_total_rows=snapshot.current_left_total_rows,
            current_right_total_rows=snapshot.current_right_total_rows,
        )

    def get_parameters(self) -> RecordReplayResponse:
        """读取当前持久化参数。"""

        response = self.status()
        return RecordReplayResponse(
            state=response.state,
            accepted=response.accepted,
            left_csv_state=response.left_csv_state,
            plan_index=response.plan_index,
            error_text=response.error_text,
            left_csv_files=response.left_csv_files,
            right_csv_files=response.right_csv_files,
            execution_tasks=response.execution_tasks,
            current_task_sequence=response.current_task_sequence,
            current_task_active=response.current_task_active,
            total_execution_count=response.total_execution_count,
            current_left_csv=response.current_left_csv,
            current_right_csv=response.current_right_csv,
            current_left_row=response.current_left_row,
            current_right_row=response.current_right_row,
            current_left_total_rows=response.current_left_total_rows,
            current_right_total_rows=response.current_right_total_rows,
            parameters=self._config_store.load(),
        )

    def update_parameters(self, changes: dict[str, float | int]) -> RecordReplayResponse:
        """校验、保存参数，并更新下一轮业务配置。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("回放正在执行，不能修改运行参数")
            parameters = self._config_store.update(changes)
            self._context.update_settings(parameters.to_service_settings())
        return self.get_parameters()

    def get_device_status(self) -> DeviceStatusResponse:
        """回放空闲时读取现场设备连接与当前状态。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("回放正在执行，拒绝并发读取设备状态")
            return self._device_status_reader.read()

    def join(self) -> None:
        """服务退出时等待正在运行的硬件业务完成。"""

        worker = self._worker
        if worker is not None:
            worker.join()

    def _run_once(self, enable_agv_navigation: bool) -> None:
        """在唯一业务线程中执行一轮并保留失败快照。"""

        try:
            self._cycle_service.run_once(enable_agv_navigation=enable_agv_navigation)
        except Exception:
            logger.exception("record replay cycle failed")

