"""把 HTTP API 桥接到单实例回放业务线程和配置存储。"""

from __future__ import annotations

import threading
from collections.abc import Callable

from loguru import logger

from ..action_sequence import ActionSequencePlan, load_action_sequence
from ..context import ReplayContext
from ..contracts import ReplayServiceState
from ..cycle_service import RecordReplayCycleService
from ..device_status import DeviceStatusReader, DeviceStatusResponse
from .config_store import RuntimeConfigStore, RuntimeParameterValue
from .prior_store import PriorKind, PriorReplacement, RecordReplayPriorStore
from .protocol import PriorUploadResponse, RecordReplayResponse
from .state_store import ReplayStateStore


class RecordReplayApplication:
    """管理单个硬件任务，并原子更新后续轮次使用的持久化参数。"""

    def __init__(
        self,
        context: ReplayContext,
        cycle_service_factory: Callable[[ActionSequencePlan], RecordReplayCycleService],
        config_store: RuntimeConfigStore,
        device_status_reader: DeviceStatusReader,
        prior_store: RecordReplayPriorStore,
        state_store: ReplayStateStore,
    ) -> None:
        self._context = context
        self._cycle_service_factory = cycle_service_factory
        self._cycle_service: RecordReplayCycleService | None = None
        self._config_store = config_store
        self._device_status_reader = device_status_reader
        self._prior_store = prior_store
        self._state_store = state_store
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._total_execution_count = 0
        initial_state = self._state_store.load()
        self._context.set_state(initial_state)
        self._context.set_total_execution_count(self._total_execution_count)
        self._state_store.save(initial_state)

    def start(self, enable_agv_navigation: bool) -> RecordReplayResponse:
        """启动唯一业务线程；已有任务运行时拒绝重复启动。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status(accepted=False)
            current_state = self._context.snapshot().state
            if current_state is ReplayServiceState.RAPID_STOP:
                return self.status(accepted=False, error_text="服务处于 rapid_stop，必须人工 reset")
            if current_state is not ReplayServiceState.IDLE:
                return self.status(accepted=False, error_text="服务当前不在 idle 状态，拒绝重复 start")
            try:
                plan = load_action_sequence(
                    self._context.config.action_sequence_path,
                    self._context.config.left_record_dir,
                    self._context.config.right_record_dir,
                )
                prior_result = self._prior_store.validate_all(plan.deployment.offset_config)
                if not prior_result.valid:
                    error_text = prior_result.error_text()
                    self._context.set_state(ReplayServiceState.IDLE, error_text=error_text)
                    return self.status(
                        accepted=False,
                        error_text=error_text,
                    )
                self._context.apply_offset_settings(plan.deployment.offset_settings)
                cycle_service = self._cycle_service_factory(plan)
                _, plan = cycle_service.refresh_deployment_status(plan)
            except Exception as exc:
                self._context.set_state(
                    ReplayServiceState.IDLE,
                    error_text=f"回放初始化失败：{type(exc).__name__}: {exc}",
                )
                return self.status(
                    accepted=False,
                    error_text=f"回放初始化失败：{type(exc).__name__}: {exc}",
                )
            self._context.reset_for_next_cycle()
            self._context.set_state(ReplayServiceState.BUSY)
            self._state_store.save(ReplayServiceState.BUSY)
            self._cycle_service = cycle_service
            self._worker = threading.Thread(
                target=self._run_once,
                args=(enable_agv_navigation, plan),
                name="record-replay-worker",
                daemon=False,
            )
            self._total_execution_count += 1
            self._context.set_total_execution_count(self._total_execution_count)
            self._worker.start()
        return self.status()

    def stop(self) -> RecordReplayResponse:
        """锁存 rapid_stop，停止 AGV 和当前已连接的左右 AR5。"""

        with self._lock:
            self._context.stop_event.set()
            self._context.set_state(
                ReplayServiceState.RAPID_STOP,
                error_text="收到人工停止请求，等待 reset",
            )
            self._state_store.save(ReplayServiceState.RAPID_STOP)
            cycle_service = self._cycle_service
            worker = self._worker
        if cycle_service is not None:
            try:
                cycle_service.stop_devices()
            except Exception as error:
                logger.error("设备停止调用存在失败：{}", error)
                with self._lock:
                    self._context.set_state(
                        ReplayServiceState.RAPID_STOP,
                        error_text=f"设备停止调用存在失败：{type(error).__name__}: {error}",
                    )
                    self._state_store.save(ReplayServiceState.RAPID_STOP)
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        return self.status()

    def reset(self) -> RecordReplayResponse:
        """人工处理完成后清除 rapid_stop 锁存并恢复 idle。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status(accepted=False, error_text="回放线程仍在退出，暂不能 reset")
            if self._context.snapshot().state is not ReplayServiceState.RAPID_STOP:
                return self.status(accepted=False, error_text="当前不在 rapid_stop 状态")
            self._context.reset_after_manual_reset()
            self._state_store.save(ReplayServiceState.IDLE)
        return self.status()

    def status(
        self,
        *,
        accepted: bool = True,
        error_text: str | None = None,
    ) -> RecordReplayResponse:
        """返回当前原子状态快照。"""

        snapshot = self._context.snapshot()
        current_task_sequence = 0 if snapshot.current_task_index is None else snapshot.current_task_index + 1
        return RecordReplayResponse(
            state=snapshot.state,
            accepted=accepted,
            action_sequence_sha256=snapshot.action_sequence_sha256,
            left_csv_state=snapshot.left_csv_state,
            plan_index=snapshot.plan_index,
            error_text=snapshot.error_text if error_text is None else error_text,
            left_csv_files=snapshot.left_csv_files,
            right_csv_files=snapshot.right_csv_files,
            execution_tasks=snapshot.execution_tasks,
            current_task_sequence=current_task_sequence,
            current_task_active=snapshot.current_task_active,
            total_execution_count=snapshot.total_execution_count,
            current_left_csv=snapshot.current_left_csv,
            current_left_action_name=snapshot.current_left_action_name,
            current_left_action_index=snapshot.current_left_action_index,
            current_right_csv=snapshot.current_right_csv,
            current_right_action_name=snapshot.current_right_action_name,
            current_right_action_index=snapshot.current_right_action_index,
            current_left_row=snapshot.current_left_row,
            current_right_row=snapshot.current_right_row,
            current_left_total_rows=snapshot.current_left_total_rows,
            current_right_total_rows=snapshot.current_right_total_rows,
            offset_statuses=snapshot.offset_statuses,
        )

    def get_parameters(self) -> RecordReplayResponse:
        """读取当前持久化参数。"""

        response = self.status()
        return RecordReplayResponse(
            state=response.state,
            accepted=response.accepted,
            action_sequence_sha256=response.action_sequence_sha256,
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
            current_left_action_name=response.current_left_action_name,
            current_left_action_index=response.current_left_action_index,
            current_right_csv=response.current_right_csv,
            current_right_action_name=response.current_right_action_name,
            current_right_action_index=response.current_right_action_index,
            current_left_row=response.current_left_row,
            current_right_row=response.current_right_row,
            current_left_total_rows=response.current_left_total_rows,
            current_right_total_rows=response.current_right_total_rows,
            offset_statuses=response.offset_statuses,
            parameters=self._config_store.load(),
        )

    def update_parameters(self, changes: dict[str, RuntimeParameterValue]) -> RecordReplayResponse:
        """校验、保存参数，并更新下一轮业务配置。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("回放正在执行，不能修改运行参数")
            if self._context.snapshot().state is not ReplayServiceState.IDLE:
                raise RuntimeError("服务当前不在 idle 状态，不能修改运行参数")
            parameters = self._config_store.update(changes)
            self._context.update_settings(parameters.to_service_settings())
        return self.get_parameters()

    def get_device_status(self) -> DeviceStatusResponse:
        """回放空闲时读取现场设备连接与当前状态。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("回放正在执行，拒绝并发读取设备状态")
            return self._device_status_reader.read()

    def replace_ball_pose_prior(self, payload: object) -> PriorUploadResponse:
        """替换三球 JSON 先验并备份旧文件。"""

        return self._replace_prior("ball_pose", payload)

    def replace_charuco_prior(self, payload: object) -> PriorUploadResponse:
        """替换 ChArUco JSON 先验并备份旧文件。"""

        return self._replace_prior("charuco", payload)

    def join(self) -> None:
        """服务退出时等待正在运行的硬件业务完成。"""

        worker = self._worker
        if worker is not None:
            worker.join()

    def _run_once(self, enable_agv_navigation: bool, plan: ActionSequencePlan) -> None:
        """在唯一业务线程中执行一轮并保留失败快照。"""

        try:
            cycle_service = self._cycle_service
            if cycle_service is None:
                raise RuntimeError("回放业务尚未初始化")
            cycle_service.run_once(enable_agv_navigation=enable_agv_navigation, plan=plan)
        except Exception:
            logger.error("record replay cycle failed")
        finally:
            self._state_store.save(self._context.snapshot().state)

    def _replace_prior(self, kind: PriorKind, payload: object) -> PriorUploadResponse:
        """按统一动作 JSON 的先验路径替换 JSON 并转换为 HTTP 响应对象。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("回放正在执行，不能替换先验")
            if self._context.snapshot().state is not ReplayServiceState.IDLE:
                raise RuntimeError("服务当前不在 idle 状态，不能替换先验")
            plan = load_action_sequence(
                self._context.config.action_sequence_path,
                self._context.config.left_record_dir,
                self._context.config.right_record_dir,
            )
            offset_config = plan.deployment.offset_config
            if kind == "ball_pose":
                target_path = offset_config.prior_capture_path
            else:
                target_path = offset_config.charuco_prior_path
            if target_path is None:
                raise RuntimeError(f"统一 JSON 未配置 {kind} 先验路径")
            replacement: PriorReplacement = self._prior_store.replace_json(
                kind,
                payload,
                target_path=target_path,
            )
            return PriorUploadResponse(True, replacement.file_name, replacement.backup_file)
