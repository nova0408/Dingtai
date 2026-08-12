"""把 HTTP API 桥接到单实例回放业务线程和配置存储。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from .. import RECORD_REPLAY_VERSION
from ..action_sequence import (
    ActionSequencePlan,
    ActionSequenceValidationError,
    NamedActionPlan,
    load_action_sequence,
    load_replay_deployment_config,
)
from ..context import ReplayContext
from ..contracts import ReplayErrorCode, ReplayExecutionPhase, ReplayServiceState
from ..cycle_service import RecordReplayCycleService
from ..device_status import DeviceStatusReader, DeviceStatusResponse
from .config_store import RuntimeConfigStore, RuntimeParameterValue
from .prior_store import PriorKind, PriorReplacement, RecordReplayPriorStore
from .protocol import (
    PriorUploadResponse,
    RECORD_REPLAY_API_VERSION,
    RecordReplayHealthResponse,
    RecordReplayPlanResponse,
    ReplayPlanAction,
    RecordReplayResponse,
)
from .state_store import ReplayStateStore


class RecordReplayApplicationError(RuntimeError):
    """应用层拒绝请求时携带稳定错误码的异常。"""

    def __init__(self, error_code: ReplayErrorCode, message: str) -> None:
        super().__init__(message)
        self.error_code: ReplayErrorCode = error_code


def _preview_actions(
    actions: tuple[NamedActionPlan, ...],
    row_counts: dict[Path, int],
) -> tuple[ReplayPlanAction, ...]:
    """把单次动作按实际执行顺序转换为 GUI 可展示的摘要。"""

    result: list[ReplayPlanAction] = []
    for sequence, action in enumerate(actions, start=1):
        item = action.item
        result.append(
            ReplayPlanAction(
                sequence=sequence,
                csv=action.csv_asset.path.name,
                action_name=item.function_name,
                action_type=item.action_type,
                speed=item.speed,
                zone=item.zone,
                index=item.index,
                final_speed=item.final_speed,
                settle_delay=item.settle_delay,
                row_count=row_counts[action.csv_asset.path],
            )
        )
    return tuple(result)


class RecordReplayApplication:
    """管理单次硬件任务，并原子更新后续 start 使用的持久化参数。"""

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
        initial_state = self._state_store.load()
        if initial_state is ReplayServiceState.RAPID_STOP:
            recovery_text = (
                "服务启动时从 runtime_state.json 恢复到 rapid_stop；上次持久化状态为 busy、"
                "rapid_stop 或状态文件无效。为防止异常中止后自动续跑，请检查上次进程日志并人工 reset"
            )
            self._context.set_state(
                initial_state,
                error_code="rapid_stop",
                error_text=recovery_text,
            )
            logger.warning("RecordReplay 启动状态恢复：{}", recovery_text)
        else:
            self._context.set_state(initial_state)
            logger.info("RecordReplay 启动状态恢复 state={}", initial_state)
        self._context.set_total_execution_count(0)
        self._state_store.save(initial_state)

    def start(
        self,
        old_tray_current_index: int,
        old_tray_put_index: int,
        new_tray_current_index: int,
        new_tray_put_index: int,
        enable_agv_navigation: bool,
        agv_target: str,
    ) -> RecordReplayResponse:
        """启动唯一业务线程；已有任务运行时拒绝重复启动。"""

        with self._lock:
            logger.info(
                "收到 RecordReplay 启动请求 old_current={} old_put={} new_current={} "
                "new_put={} agv_enabled={} agv_target={}",
                old_tray_current_index,
                old_tray_put_index,
                new_tray_current_index,
                new_tray_put_index,
                enable_agv_navigation,
                agv_target,
            )
            _validate_start_parameters(
                old_tray_current_index,
                old_tray_put_index,
                new_tray_current_index,
                new_tray_put_index,
                enable_agv_navigation,
                agv_target,
            )
            if self._worker is not None and self._worker.is_alive():
                logger.warning("拒绝 RecordReplay 启动请求：已有回放线程仍在运行")
                return self.status(
                    accepted=False,
                    error_code="busy",
                    error_text="已有回放线程仍在运行，请等待当前任务结束后重试",
                )
            current_state = self._context.snapshot().state
            if current_state is ReplayServiceState.RAPID_STOP:
                return self.status(
                    accepted=False,
                    error_code="rapid_stop",
                    error_text="服务处于 rapid_stop，必须人工 reset",
                )
            if current_state is not ReplayServiceState.IDLE:
                return self.status(
                    accepted=False,
                    error_code="invalid_state",
                    error_text="服务当前不在 idle 状态，拒绝重复 start",
                )
            try:
                plan = load_action_sequence(
                    self._context.config.action_sequence_path,
                    self._context.config.left_record_dir,
                    self._context.config.right_record_dir,
                    old_tray_current_index=old_tray_current_index,
                    old_tray_put_index=old_tray_put_index,
                    new_tray_current_index=new_tray_current_index,
                    new_tray_put_index=new_tray_put_index,
                )
            except ActionSequenceValidationError as error:
                error_text = f"动作计划校验失败：{error}"
                logger.warning("RecordReplay 启动前计划校验失败：{}", error)
                self._context.set_state(
                    ReplayServiceState.IDLE,
                    error_code="invalid_plan",
                    error_text=error_text,
                )
                return self.status(
                    accepted=False,
                    error_code="invalid_plan",
                    error_text=error_text,
                )
            prior_result = self._prior_store.validate_all(plan.deployment.offset_config)
            if not prior_result.valid:
                error_text = prior_result.error_text()
                logger.warning("RecordReplay 启动前先验校验失败：{}", error_text)
                self._context.set_state(
                    ReplayServiceState.IDLE,
                    error_code="invalid_plan",
                    error_text=error_text,
                )
                return self.status(
                    accepted=False,
                    error_code="invalid_plan",
                    error_text=error_text,
                )
            self._context.apply_offset_settings(plan.deployment.offset_settings)
            cycle_service = self._cycle_service_factory(plan)
            _, plan = cycle_service.refresh_deployment_status(plan)
            logger.info(
                "RecordReplay 启动计划已冻结 sha256={} left_actions={} right_actions={}",
                plan.source_sha256,
                len(plan.left_actions),
                len(plan.right_actions),
            )
            self._context.reset_for_next_cycle()
            self._context.set_start_parameters(
                old_tray_current_index,
                old_tray_put_index,
                new_tray_current_index,
                new_tray_put_index,
                enable_agv_navigation,
                agv_target,
            )
            self._cycle_service = cycle_service
            worker = threading.Thread(
                target=self._run_once,
                args=(enable_agv_navigation, agv_target, plan),
                name="record-replay-worker",
                daemon=False,
            )
            self._worker = worker
            try:
                self._context.set_state(
                    ReplayServiceState.BUSY,
                    execution_phase=ReplayExecutionPhase.PREPARING_DEVICES,
                )
                self._state_store.save(ReplayServiceState.BUSY)
                worker.start()
            except Exception as error:
                logger.exception("RecordReplay 回放线程启动失败 type={}", type(error).__name__)
                self._worker = None
                self._cycle_service = None
                self._context.set_state(
                    ReplayServiceState.IDLE,
                    error_code="internal_error",
                    error_text=f"回放线程启动失败：{type(error).__name__}: {error}",
                )
                try:
                    self._state_store.save(ReplayServiceState.IDLE)
                except Exception:
                    logger.exception("回放线程启动回滚时，持久化 idle 状态失败")
                raise
            logger.info("RecordReplay 回放线程已启动 worker={}", worker.name)
        return self.status()

    def stop(self) -> RecordReplayResponse:
        """锁存 rapid_stop，停止 AGV 和当前已连接的左右 AR5。"""

        with self._lock:
            logger.warning("收到 RecordReplay 人工停止请求")
            self._context.stop_event.set()
            self._context.set_state(
                ReplayServiceState.RAPID_STOP,
                error_code="stop_requested",
                error_text="收到人工停止请求，等待 reset",
            )
            self._state_store.save(ReplayServiceState.RAPID_STOP)
            cycle_service = self._cycle_service
            worker = self._worker
        if cycle_service is not None:
            try:
                cycle_service.stop_devices()
            except Exception as error:
                logger.exception("RecordReplay 人工停止设备失败 type={}", type(error).__name__)
                with self._lock:
                    self._context.set_state(
                        ReplayServiceState.RAPID_STOP,
                        error_code="stop_failed",
                        error_text=f"设备停止调用存在失败：{error}",
                    )
                    self._state_store.save(ReplayServiceState.RAPID_STOP)
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        logger.info("RecordReplay 人工停止流程结束 state={}", self._context.snapshot().state)
        return self.status()

    def reset(self) -> RecordReplayResponse:
        """人工处理完成后清除 rapid_stop 锁存并恢复 idle。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status(
                    accepted=False,
                    error_code="busy",
                    error_text="回放线程仍在退出，暂不能 reset",
                )
            if self._context.snapshot().state is not ReplayServiceState.RAPID_STOP:
                return self.status(
                    accepted=False,
                    error_code="invalid_state",
                    error_text="当前不在 rapid_stop 状态",
                )
            self._context.reset_after_manual_reset()
            self._state_store.save(ReplayServiceState.IDLE)
            logger.info("RecordReplay rapid_stop 已人工复位为 idle")
        return self.status()

    def status(
        self,
        *,
        accepted: bool = True,
        error_code: ReplayErrorCode | None = None,
        error_text: str | None = None,
    ) -> RecordReplayResponse:
        """返回当前原子状态快照。"""

        snapshot = self._context.snapshot()
        current_task_sequence = 0 if snapshot.current_task_index is None else snapshot.current_task_index + 1
        return RecordReplayResponse(
            state=snapshot.state,
            execution_phase=snapshot.execution_phase,
            accepted=accepted,
            error_code=snapshot.error_code if error_code is None else error_code,
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
            old_tray_current_index=snapshot.old_tray_current_index,
            old_tray_put_index=snapshot.old_tray_put_index,
            new_tray_current_index=snapshot.new_tray_current_index,
            new_tray_put_index=snapshot.new_tray_put_index,
            agv_navigation_enabled=snapshot.agv_navigation_enabled,
            agv_target=snapshot.agv_target,
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

    def health(self) -> RecordReplayHealthResponse:
        """返回服务版本和状态，不连接或控制任何现场设备。"""

        return RecordReplayHealthResponse(
            service_version=RECORD_REPLAY_VERSION,
            api_version=RECORD_REPLAY_API_VERSION,
            state=self._context.snapshot().state,
        )

    def get_plan(
        self,
        old_tray_current_index: int,
        old_tray_put_index: int,
        new_tray_current_index: int,
        new_tray_put_index: int,
    ) -> RecordReplayPlanResponse:
        """读取一次 start 对应的回放计划，不连接设备、不创建线程。"""

        with self._lock:
            logger.info(
                "读取 RecordReplay 计划 old_current={} old_put={} new_current={} new_put={}",
                old_tray_current_index,
                old_tray_put_index,
                new_tray_current_index,
                new_tray_put_index,
            )
            snapshot = self._context.snapshot()
            if self._worker is not None and self._worker.is_alive():
                return RecordReplayPlanResponse(
                    state=snapshot.state,
                    accepted=False,
                    error_code="busy",
                    error_text="回放正在执行，请通过 WebSocket 读取实时状态",
                )
            if snapshot.state is not ReplayServiceState.IDLE:
                return RecordReplayPlanResponse(
                    state=snapshot.state,
                    accepted=False,
                    error_code="invalid_state",
                    error_text="服务当前不在 idle 状态，不能读取下一轮计划",
                )
            try:
                plan = load_action_sequence(
                    self._context.config.action_sequence_path,
                    self._context.config.left_record_dir,
                    self._context.config.right_record_dir,
                    old_tray_current_index=old_tray_current_index,
                    old_tray_put_index=old_tray_put_index,
                    new_tray_current_index=new_tray_current_index,
                    new_tray_put_index=new_tray_put_index,
                )
            except ActionSequenceValidationError as error:
                logger.warning("读取 RecordReplay 计划失败：{}", error)
                return RecordReplayPlanResponse(
                    state=snapshot.state,
                    accepted=False,
                    error_code="invalid_plan",
                    error_text=str(error),
                )
            row_counts = {
                path: len(rows) for path, rows in plan.preloaded_rows_by_path
            }
            logger.info(
                "RecordReplay 计划读取完成 sha256={} left_actions={} right_actions={}",
                plan.source_sha256,
                len(plan.left_actions),
                len(plan.right_actions),
            )
            return RecordReplayPlanResponse(
                state=snapshot.state,
                action_sequence_sha256=plan.source_sha256,
                old_tray_current_index=old_tray_current_index,
                old_tray_put_index=old_tray_put_index,
                new_tray_current_index=new_tray_current_index,
                new_tray_put_index=new_tray_put_index,
                left=_preview_actions(plan.left_actions, row_counts),
                right=_preview_actions(plan.right_actions, row_counts),
            )

    def get_parameters(self) -> RecordReplayResponse:
        """读取当前持久化参数。"""

        response = self.status()
        return RecordReplayResponse(
            state=response.state,
            execution_phase=response.execution_phase,
            accepted=response.accepted,
            error_code=response.error_code,
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
            old_tray_current_index=response.old_tray_current_index,
            old_tray_put_index=response.old_tray_put_index,
            new_tray_current_index=response.new_tray_current_index,
            new_tray_put_index=response.new_tray_put_index,
            agv_navigation_enabled=response.agv_navigation_enabled,
            agv_target=response.agv_target,
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
                raise RecordReplayApplicationError("busy", "回放正在执行，不能修改运行参数")
            if self._context.snapshot().state is not ReplayServiceState.IDLE:
                raise RecordReplayApplicationError(
                    "invalid_state",
                    "服务当前不在 idle 状态，不能修改运行参数",
                )
            try:
                parameters = self._config_store.update(changes)
            except ValueError as error:
                raise RecordReplayApplicationError(
                    "invalid_request",
                    f"运行参数校验失败：{error}",
                ) from error
            self._context.update_settings(parameters.to_service_settings())
            logger.info("RecordReplay 运行参数已更新 fields={}", sorted(changes))
        return self.get_parameters()

    def get_device_status(self) -> DeviceStatusResponse:
        """回放空闲时读取现场设备连接与当前状态。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RecordReplayApplicationError(
                    "busy",
                    "回放正在执行，拒绝并发读取设备状态",
                )
            response = self._device_status_reader.read()
            logger.info("RecordReplay 设备状态读取完成 all_connected={}", response.all_connected)
            return response

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

    def _run_once(
        self,
        enable_agv_navigation: bool,
        agv_target: str,
        plan: ActionSequencePlan,
    ) -> None:
        """在唯一业务线程中执行一次并保留失败快照。"""

        try:
            logger.info(
                "RecordReplay 后台执行开始 sha256={} agv_enabled={} agv_target={}",
                plan.source_sha256,
                enable_agv_navigation,
                agv_target,
            )
            cycle_service = self._cycle_service
            if cycle_service is None:
                raise RuntimeError("回放业务尚未初始化")
            cycle_service.run_once(
                enable_agv_navigation=enable_agv_navigation,
                agv_target=agv_target,
                plan=plan,
            )
            self._context.complete_execution()
            logger.info(
                "RecordReplay 后台执行成功 total_execution_count={}",
                self._context.snapshot().total_execution_count,
            )
        except Exception as error:
            snapshot = self._context.snapshot()
            if snapshot.error_code is None:
                self._context.set_state(
                    ReplayServiceState.RAPID_STOP,
                    error_code="internal_error",
                    error_text=f"后台执行发生未处理异常：{type(error).__name__}: {error}",
                )
            logger.exception(
                "RecordReplay 后台执行失败 type={} state={} error_code={}",
                type(error).__name__,
                self._context.snapshot().state,
                self._context.snapshot().error_code,
            )
        finally:
            final_state = self._context.snapshot().state
            try:
                self._state_store.save(final_state)
            except Exception as error:
                logger.exception(
                    "RecordReplay 后台结束状态持久化失败 state={} type={}",
                    final_state,
                    type(error).__name__,
                )
                if self._context.snapshot().error_code is None:
                    self._context.set_state(
                        ReplayServiceState.RAPID_STOP,
                        error_code="internal_error",
                        error_text=f"执行结果状态持久化失败：{type(error).__name__}: {error}",
                    )

    def _replace_prior(self, kind: PriorKind, payload: object) -> PriorUploadResponse:
        """按统一动作 JSON 的先验路径替换 JSON 并转换为 HTTP 响应对象。"""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RecordReplayApplicationError("busy", "回放正在执行，不能替换先验")
            if self._context.snapshot().state is not ReplayServiceState.IDLE:
                raise RecordReplayApplicationError(
                    "invalid_state",
                    "服务当前不在 idle 状态，不能替换先验",
                )
            deployment = load_replay_deployment_config(
                self._context.config.action_sequence_path
            )
            offset_config = deployment.offset_config
            if kind == "ball_pose":
                target_path = offset_config.prior_capture_path
            else:
                target_path = offset_config.charuco_prior_path
            if target_path is None:
                raise RuntimeError(f"统一 JSON 未配置 {kind} 先验路径")
            try:
                replacement: PriorReplacement = self._prior_store.replace_json(
                    kind,
                    payload,
                    target_path=target_path,
                )
            except ValueError as error:
                raise RecordReplayApplicationError(
                    "invalid_request",
                    f"{kind} 先验内容校验失败：{error}",
                ) from error
            logger.info(
                "RecordReplay 先验已替换 kind={} file={} backup={}",
                kind,
                replacement.file_name,
                replacement.backup_file,
            )
            return PriorUploadResponse(True, replacement.file_name, replacement.backup_file)


def _validate_start_parameters(
    old_tray_current_index: int,
    old_tray_put_index: int,
    new_tray_current_index: int,
    new_tray_put_index: int,
    enable_agv_navigation: bool,
    agv_target: str,
) -> None:
    """校验一次 start 的四个托盘位置、AGV 开关和目标。"""

    tray_indices = (
        ("old_tray_current_index", old_tray_current_index),
        ("old_tray_put_index", old_tray_put_index),
        ("new_tray_current_index", new_tray_current_index),
        ("new_tray_put_index", new_tray_put_index),
    )
    for name, value in tray_indices:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RecordReplayApplicationError(
                "invalid_index",
                f"{name} 必须是大于 0 的整数",
            )
    if not isinstance(enable_agv_navigation, bool):
        raise RecordReplayApplicationError(
            "invalid_request",
            "enable_agv_navigation 必须是 bool",
        )
    if not isinstance(agv_target, str) or not agv_target.strip():
        raise RecordReplayApplicationError("invalid_request", "agv_target 必须是非空字符串")
