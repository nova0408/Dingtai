"""双臂自动回放的总运行上下文。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .contracts import (
    ReplayCsvFileStatus,
    ReplayExecutionTaskStatus,
    ReplayServiceState,
    ReplayStatusSnapshot,
)
from .settings import ReplayCycleConfig
from .settings import ReplayServiceSettings

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
        """创建一轮或多轮可复用的执行上下文。"""

        self.config = config
        self.resources = ReplayRuntimeResources()
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = ReplayStatusSnapshot(state=ReplayServiceState.WAITING)

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
        error_text: str | None = None,
    ) -> None:
        """原子更新服务阶段与当前左臂执行状态。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                state=state,
                left_csv_state=left_csv_state,
                plan_index=plan_index,
                error_text=error_text,
            )

    def set_deployment_status(
        self,
        left_csv_files: tuple[ReplayCsvFileStatus, ...],
        right_csv_files: tuple[ReplayCsvFileStatus, ...],
        execution_tasks: tuple[ReplayExecutionTaskStatus, ...],
    ) -> None:
        """原子发布左右臂已部署 CSV 与对齐执行任务。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                left_csv_files=left_csv_files,
                right_csv_files=right_csv_files,
                execution_tasks=execution_tasks,
            )

    def reset_run_progress(self) -> None:
        """在新一轮开始前清除上一轮当前 CSV 与行进度。"""

        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                left_csv_state=None,
                plan_index=None,
                error_text=None,
                current_task_index=None,
                current_task_active=False,
                current_left_csv=None,
                current_right_csv=None,
                current_left_row=None,
                current_right_row=None,
                current_left_total_rows=None,
                current_right_total_rows=None,
            )

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
                current_task_active=True,
                current_left_csv=None,
                current_right_csv=None,
                current_left_row=None,
                current_right_row=None,
                current_left_total_rows=None,
                current_right_total_rows=None,
            )

    def complete_execution_task(self) -> None:
        """标记当前对齐任务完成并清除两侧实时行进度。"""

        with self._lock:
            if self._snapshot.current_task_index is None:
                raise RuntimeError("没有可以完成的当前执行任务")
            self._snapshot = replace(
                self._snapshot,
                current_task_active=False,
                current_left_csv=None,
                current_right_csv=None,
                current_left_row=None,
                current_right_row=None,
                current_left_total_rows=None,
                current_right_total_rows=None,
            )

    def set_csv_progress(
        self,
        arm_side: str,
        csv_name: str,
        row_index: int,
        total_rows: int,
    ) -> None:
        """原子发布一侧当前 CSV 与源数据行进度。"""

        with self._lock:
            if arm_side == "left":
                self._snapshot = replace(
                    self._snapshot,
                    current_left_csv=csv_name,
                    current_left_row=row_index,
                    current_left_total_rows=total_rows,
                )
                return
            if arm_side == "right":
                self._snapshot = replace(
                    self._snapshot,
                    current_right_csv=csv_name,
                    current_right_row=row_index,
                    current_right_total_rows=total_rows,
                )
                return
            raise ValueError(f"未知机械臂侧别：{arm_side}")

    def clear_csv_progress(self, arm_side: str) -> None:
        """一侧 CSV 完成后清除其当前运行标记。"""

        with self._lock:
            if arm_side == "left":
                self._snapshot = replace(
                    self._snapshot,
                    current_left_csv=None,
                    current_left_row=None,
                    current_left_total_rows=None,
                )
                return
            if arm_side == "right":
                self._snapshot = replace(
                    self._snapshot,
                    current_right_csv=None,
                    current_right_row=None,
                    current_right_total_rows=None,
                )
                return
            raise ValueError(f"未知机械臂侧别：{arm_side}")

    def reset_for_next_cycle(self) -> None:
        """清除停止信号和上一轮状态，准备接收下一轮指令。"""

        self.stop_event.clear()
        self.reset_run_progress()
        self.set_state(ReplayServiceState.WAITING)

    def update_settings(self, settings: ReplayServiceSettings) -> None:
        """仅在未执行硬件任务时替换后续轮次使用的运行参数。"""

        with self._lock:
            if self._snapshot.state not in (ReplayServiceState.WAITING, ReplayServiceState.FAILED):
                raise RuntimeError("回放正在执行，不能修改运行参数")
            self.config = replace(self.config, settings=settings)

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

    # endregion
