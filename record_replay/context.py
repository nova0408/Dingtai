"""双臂自动回放的总运行上下文。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING

from .contracts import (
    ReplayCsvFileStatus,
    ReplayExecutionTaskStatus,
    ReplayExecutionCompletedEvent,
    ReplayErrorCode,
    ReplayOffsetStatus,
    ReplayServiceState,
    ReplayStatusSnapshot,
)
from .settings import ReplayCycleConfig
from .settings import ReplayOffsetSettings, ReplayServiceSettings

if TYPE_CHECKING:
    from .runtime import ReplayRuntime


@dataclass(slots=True)
class ReplayRuntimeResources:
    """一轮回放创建的设备资源。"""

    left_runtime: ReplayRuntime | None = None
    "左臂运行时资源，由设备网关创建。"
    right_runtime: ReplayRuntime | None = None
    "右臂运行时资源，仅同步 CSV 存在时创建。"


class ReplayContext:
    """自动回放的唯一跨模块资源与状态边界。

    该类管理配置、左右运行时资源、共享停止事件及状态快照；不解析 CSV、不做
    运动计算、不发 AGV 指令。执行器负责写入资源和状态，服务及外部查询方只读
    快照，从而避免 CSV、设备和服务模块相互穿透。
    """

    # region 初始化

    def __init__(self, config: ReplayCycleConfig) -> None:
        """创建服务进程内可复用的单次执行上下文。"""

        self.config = config
        self.resources = ReplayRuntimeResources()
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status_subscribers: list[
            Queue[ReplayStatusSnapshot | ReplayExecutionCompletedEvent]
        ] = []
        self._snapshot = ReplayStatusSnapshot(state=ReplayServiceState.IDLE)

    # endregion

    # region 状态

    def snapshot(self) -> ReplayStatusSnapshot:
        """返回当前不可变状态快照。"""

        with self._lock:
            return self._snapshot

    def set_state(
        self,
        state: ReplayServiceState,
        *,
        left_csv_state: str | None = None,
        plan_index: int | None = None,
        error_code: ReplayErrorCode | None = None,
        error_text: str | None = None,
    ) -> None:
        """原子更新服务阶段与当前左臂执行状态。"""

        with self._lock:
            if self._snapshot.state is ReplayServiceState.RAPID_STOP and state is not ReplayServiceState.RAPID_STOP:
                return
            clear_task = state is ReplayServiceState.IDLE and plan_index is None
            next_plan_index = (
                self._snapshot.plan_index if plan_index is None else plan_index
            )
            self._snapshot = replace(
                self._snapshot,
                state=state,
                left_csv_state=left_csv_state,
                plan_index=None if clear_task else next_plan_index,
                current_task_index=(
                    None
                    if clear_task
                    else next_plan_index
                    if plan_index is not None
                    else self._snapshot.current_task_index
                ),
                current_task_active=(
                    False
                    if clear_task
                    else True if plan_index is not None else self._snapshot.current_task_active
                ),
                error_code=error_code,
                error_text=error_text,
            )
            self._publish_locked()

    def apply_offset_settings(self, offset_settings: ReplayOffsetSettings) -> None:
        """在 start 前应用统一动作 JSON 中冻结的 offset 配置。"""

        with self._lock:
            self.config = replace(
                self.config,
                settings=replace(self.config.settings, offset=offset_settings),
            )
            self._publish_locked()

    def set_total_execution_count(self, count: int) -> None:
        """发布服务进程内已成功完成的回放轮次。"""

        if count < 0:
            raise ValueError("total_execution_count 不能为负数")
        with self._lock:
            self._snapshot = replace(self._snapshot, total_execution_count=count)
            self._publish_locked()

    def set_start_parameters(
        self,
        old_tray_current_index: int,
        old_tray_put_index: int,
        new_tray_current_index: int,
        new_tray_put_index: int,
        enable_agv_navigation: bool,
        agv_target: str,
    ) -> None:
        """发布本次 start 冻结的四个托盘位置与 AGV 选项。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                old_tray_current_index=old_tray_current_index,
                old_tray_put_index=old_tray_put_index,
                new_tray_current_index=new_tray_current_index,
                new_tray_put_index=new_tray_put_index,
                agv_navigation_enabled=enable_agv_navigation,
                agv_target=agv_target,
            )
            self._publish_locked()

    def complete_execution(self) -> None:
        """递增完成次数并向现有订阅者投递一次性完成事件。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                total_execution_count=self._snapshot.total_execution_count + 1,
            )
            event = ReplayExecutionCompletedEvent(self._snapshot)
            for subscriber in tuple(self._status_subscribers):
                try:
                    subscriber.get_nowait()
                except Empty:
                    pass
                try:
                    subscriber.put_nowait(event)
                except Full:
                    pass

    def subscribe_status(
        self,
    ) -> Queue[ReplayStatusSnapshot | ReplayExecutionCompletedEvent]:
        """订阅状态快照；建立连接后立即收到当前状态。"""

        subscriber: Queue[ReplayStatusSnapshot | ReplayExecutionCompletedEvent] = Queue(maxsize=1)
        with self._lock:
            self._status_subscribers.append(subscriber)
            subscriber.put_nowait(self._snapshot)
        return subscriber

    def unsubscribe_status(
        self,
        subscriber: Queue[ReplayStatusSnapshot | ReplayExecutionCompletedEvent],
    ) -> None:
        """解除一个状态订阅。"""

        with self._lock:
            if subscriber in self._status_subscribers:
                self._status_subscribers.remove(subscriber)

    def set_deployment_status(
        self,
        left_csv_files: tuple[ReplayCsvFileStatus, ...],
        right_csv_files: tuple[ReplayCsvFileStatus, ...],
        execution_tasks: tuple[ReplayExecutionTaskStatus, ...],
        action_sequence_sha256: str,
    ) -> None:
        """原子发布左右臂已部署 CSV 与对齐执行任务。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                left_csv_files=left_csv_files,
                right_csv_files=right_csv_files,
                execution_tasks=execution_tasks,
                action_sequence_sha256=action_sequence_sha256,
            )
            self._publish_locked()

    def reset_run_progress(self) -> None:
        """在下一次 start 前清除上一轮当前 CSV 与行进度。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                left_csv_state=None,
                plan_index=None,
                error_code=None,
                error_text=None,
                current_task_index=None,
                current_task_active=False,
                current_left_csv=None,
                current_left_action_name=None,
                current_left_action_index=None,
                current_right_csv=None,
                current_right_action_name=None,
                current_right_action_index=None,
                current_left_row=None,
                current_right_row=None,
                current_left_total_rows=None,
                current_right_total_rows=None,
                offset_statuses=(
                    ReplayOffsetStatus("head", False, False),
                    ReplayOffsetStatus("three_ball", False, False),
                ),
            )
            self._publish_locked()

    def refresh_offset_statuses(self) -> None:
        """从左右 runtime 汇总头部与三球 offset 状态。"""

        runtimes = tuple(
            runtime
            for runtime in (self.resources.left_runtime, self.resources.right_runtime)
            if runtime is not None
        )
        head_available = any(runtime.charuco_cartesian_offset is not None for runtime in runtimes)
        three_ball_available = any(runtime.global_cartesian_offset is not None for runtime in runtimes)
        head_applied = any(runtime.offset_source == "head" for runtime in runtimes)
        three_ball_applied = any(runtime.offset_source == "three_ball" for runtime in runtimes)
        statuses = (
            ReplayOffsetStatus("head", head_available, head_applied),
            ReplayOffsetStatus("three_ball", three_ball_available, three_ball_applied),
        )
        with self._lock:
            self._snapshot = replace(self._snapshot, offset_statuses=statuses)
            self._publish_locked()

    def advance_execution_task(
        self,
        left_csv: str | None,
        right_csv: str | None,
    ) -> None:
        """切换到下一个对齐任务，并校验执行顺序与已发布计划一致。"""

        with self._lock:
            next_index = (
                0
                if self._snapshot.current_task_index is None
                else self._snapshot.current_task_index + 1
            )
            if self._snapshot.current_task_active:
                raise RuntimeError("上一执行任务尚未完成，不能切换到下一任务")
            if next_index >= len(self._snapshot.execution_tasks):
                raise RuntimeError("实际执行任务超过已发布的任务清单")
            expected = self._snapshot.execution_tasks[next_index]
            if expected.left_csv != left_csv or expected.right_csv != right_csv:
                raise RuntimeError(
                    "实际执行任务与已发布清单不一致："
                    f"expected=({expected.left_csv}, {expected.right_csv}) "
                    f"actual=({left_csv}, {right_csv})"
                )
            self._snapshot = replace(
                self._snapshot,
                current_task_index=next_index,
                plan_index=next_index,
                current_task_active=True,
                current_left_csv=None,
                current_left_action_name=None,
                current_left_action_index=None,
                current_right_csv=None,
                current_right_action_name=None,
                current_right_action_index=None,
                current_left_row=None,
                current_right_row=None,
                current_left_total_rows=None,
                current_right_total_rows=None,
            )
            self._publish_locked()

    def complete_execution_task(self) -> None:
        """标记当前对齐任务完成并清除两侧实时行进度。"""

        with self._lock:
            if self._snapshot.current_task_index is None:
                raise RuntimeError("没有可以完成的当前执行任务")
            self._snapshot = replace(
                self._snapshot,
                current_task_active=False,
                current_left_csv=None,
                current_left_action_name=None,
                current_left_action_index=None,
                current_right_csv=None,
                current_right_action_name=None,
                current_right_action_index=None,
                current_left_row=None,
                current_right_row=None,
                current_left_total_rows=None,
                current_right_total_rows=None,
            )
            self._publish_locked()

    def set_csv_progress(
        self,
        arm_side: str,
        csv_name: str,
        row_index: int,
        total_rows: int,
        action_name: str | None = None,
        action_index: int | None = None,
    ) -> None:
        """原子发布一侧当前 CSV 与源数据行进度。"""

        with self._lock:
            if arm_side == "left":
                self._snapshot = replace(
                    self._snapshot,
                    current_left_csv=csv_name,
                    current_left_action_name=action_name,
                    current_left_action_index=action_index,
                    current_left_row=row_index,
                    current_left_total_rows=total_rows,
                )
                self._publish_locked()
                return
            if arm_side == "right":
                self._snapshot = replace(
                    self._snapshot,
                    current_right_csv=csv_name,
                    current_right_action_name=action_name,
                    current_right_action_index=action_index,
                    current_right_row=row_index,
                    current_right_total_rows=total_rows,
                )
                self._publish_locked()
                return
            raise ValueError(f"未知机械臂侧别：{arm_side}")

    def clear_csv_progress(self, arm_side: str) -> None:
        """一侧 CSV 完成后清除其当前运行标记。"""

        with self._lock:
            if arm_side == "left":
                self._snapshot = replace(
                    self._snapshot,
                    current_left_csv=None,
                    current_left_action_name=None,
                    current_left_action_index=None,
                    current_left_row=None,
                    current_left_total_rows=None,
                )
                self._publish_locked()
                return
            if arm_side == "right":
                self._snapshot = replace(
                    self._snapshot,
                    current_right_csv=None,
                    current_right_action_name=None,
                    current_right_action_index=None,
                    current_right_row=None,
                    current_right_total_rows=None,
                )
                self._publish_locked()
                return
            raise ValueError(f"未知机械臂侧别：{arm_side}")

    def reset_for_next_cycle(self) -> None:
        """清除停止信号和上一轮状态，准备接收下一轮指令。"""

        if self.snapshot().state is ReplayServiceState.RAPID_STOP:
            return
        self.stop_event.clear()
        self.reset_run_progress()
        self.set_state(ReplayServiceState.IDLE)

    def reset_after_manual_reset(self) -> None:
        """人工确认现场安全后清除停止锁存并恢复 idle。"""

        self.stop_event.clear()
        self.reset_run_progress()
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                state=ReplayServiceState.IDLE,
                error_code=None,
                error_text=None,
            )
            self._publish_locked()

    def update_settings(self, settings: ReplayServiceSettings) -> None:
        """仅在未执行硬件任务时替换后续轮次使用的运行参数。"""

        with self._lock:
            if self._snapshot.state is not ReplayServiceState.IDLE:
                raise RuntimeError("回放正在执行，不能修改运行参数")
            self.config = replace(self.config, settings=settings)
            self._publish_locked()

    def attach_runtimes(self, left_runtime: ReplayRuntime, right_runtime: ReplayRuntime | None) -> None:
        """登记当前轮次已创建的左右运行时资源。"""

        self.resources.left_runtime = left_runtime
        self.resources.right_runtime = right_runtime

    def detach_runtimes(self) -> tuple[ReplayRuntime | None, ReplayRuntime | None]:
        """取走当前 runtime 引用，避免下一轮复用已经关闭的资源。"""

        left_runtime = self.resources.left_runtime
        right_runtime = self.resources.right_runtime
        self.resources.left_runtime = None
        self.resources.right_runtime = None
        return left_runtime, right_runtime

    def _publish_locked(self) -> None:
        """向订阅者投递最新快照；慢客户端只保留最后一次变化。"""

        for subscriber in tuple(self._status_subscribers):
            try:
                subscriber.get_nowait()
            except Exception:
                pass
            try:
                subscriber.put_nowait(self._snapshot)
            except Full:
                pass

    # endregion
