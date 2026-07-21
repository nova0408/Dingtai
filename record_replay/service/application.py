"""把 HTTP API 桥接到单实例回放业务线程和配置存储。"""

from __future__ import annotations

import threading

from loguru import logger

from ..context import ReplayContext
from ..cycle_service import RecordReplayCycleService
from .config_store import RuntimeConfigStore
from .protocol import RecordReplayResponse


class RecordReplayApplication:
    """管理单个硬件任务，并原子更新后续轮次使用的持久化参数。"""

    def __init__(
        self,
        context: ReplayContext,
        cycle_service: RecordReplayCycleService,
        config_store: RuntimeConfigStore,
    ) -> None:
        self._context = context
        self._cycle_service = cycle_service
        self._config_store = config_store
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def start(self) -> RecordReplayResponse:
        """启动唯一业务线程；已有任务运行时拒绝重复启动。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status(accepted=False)
            self._worker = threading.Thread(
                target=self._run_once,
                name="record-replay-worker",
                daemon=False,
            )
            self._worker.start()
        return self.status()

    def status(self, *, accepted: bool = True) -> RecordReplayResponse:
        """返回当前原子状态快照。"""

        snapshot = self._context.snapshot()
        return RecordReplayResponse(
            state=snapshot.state,
            accepted=accepted,
            left_csv_state=snapshot.left_csv_state,
            plan_index=snapshot.plan_index,
            error_text=snapshot.error_text,
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

    def join(self) -> None:
        """服务退出时等待正在运行的硬件业务完成。"""

        worker = self._worker
        if worker is not None:
            worker.join()

    def _run_once(self) -> None:
        """在唯一业务线程中执行一轮并保留失败快照。"""

        try:
            self._cycle_service.run_once()
        except Exception:
            logger.exception("record replay cycle failed")

