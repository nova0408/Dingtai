"""按双臂 CSV 执行计划驱动回放动作。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger

from .arm_actions import flush_pending_arm_segment
from .charuco_offset import CharucoOffsetInitializer
from .context import ReplayContext
from .contracts import CsvExecutionPlan, ReplayRow, ReplayServiceState
from .csv_repository import load_replay_rows, state_name_from_left_csv
from .hand_actions import execute_gripper_row, execute_lift_row, execute_m11_row
from .offset_updater import GlobalOffsetUpdater
from .runtime import ReplayRuntime, close_runtime, create_runtime, prepare_runtime


class DualArmExecutor:
    """双臂自动回放的阶段执行器。"""

    # region 主入口

    def execute(
        self,
        context: ReplayContext,
        plans: list[CsvExecutionPlan],
        offset_updater: GlobalOffsetUpdater | None = None,
        charuco_initializer: CharucoOffsetInitializer | None = None,
    ) -> None:
        """创建 runtime 并完整执行给定的双臂计划。"""

        stop_event = context.stop_event
        device_connection = context.config.device_connection
        service_settings = context.config.settings
        preloaded_rows_by_path = _preload_plan_rows(plans)
        left_runtime = create_runtime("left", stop_event, device_connection, service_settings)
        right_runtime = None
        try:
            left_runtime.preloaded_rows_by_path = preloaded_rows_by_path
            prepare_runtime(left_runtime)
            if offset_updater is not None:
                left_runtime.offset_target_sequences = service_settings.offset.target_sequences
            if _plans_require_right_runtime(plans):
                right_runtime = create_runtime("right", stop_event, device_connection, service_settings)
                right_runtime.preloaded_rows_by_path = preloaded_rows_by_path
                prepare_runtime(right_runtime)
            context.attach_runtimes(left_runtime, right_runtime)
            if charuco_initializer is not None:
                charuco_initializer.initialize(
                    [left_runtime] if right_runtime is None else [left_runtime, right_runtime]
                )
            for index, plan in enumerate(plans):
                context.set_state(
                    ReplayServiceState.BUSY,
                    left_csv_state=state_name_from_left_csv(plan.left_csv_path.name, context.config.state_prefix),
                    plan_index=index,
                )
                self._execute_plan(
                    context,
                    left_runtime,
                    right_runtime,
                    plan,
                    stop_event,
                    offset_updater,
                )
            flush_pending_arm_segment(left_runtime)
            if right_runtime is not None:
                flush_pending_arm_segment(right_runtime)
        finally:
            context.detach_runtimes()
            close_runtime(right_runtime)
            close_runtime(left_runtime)

    # endregion

    # region 阶段编排

    def _execute_plan(
        self,
        context: ReplayContext,
        left: ReplayRuntime,
        right: ReplayRuntime | None,
        plan: CsvExecutionPlan,
        stop_event: threading.Event,
        offset_updater: GlobalOffsetUpdater | None,
    ) -> None:
        left_executed = False
        if plan.start_together:
            if right is None or plan.right_start_csv_path is None:
                raise RuntimeError("启动并行阶段缺少右臂 runtime 或 CSV")
            context.advance_execution_task(
                plan.left_csv_path.name,
                plan.right_start_csv_path.name,
            )
            self._execute_parallel(
                context,
                left,
                plan.left_csv_path,
                right,
                plan.right_start_csv_path,
                stop_event,
                offset_updater,
            )
            context.complete_execution_task()
            left_executed = True
        if plan.right_pre_stage_csv_paths:
            flush_pending_arm_segment(left)
        for path in plan.right_pre_stage_csv_paths:
            if right is None:
                raise RuntimeError("右臂预阶段缺少 runtime")
            if stop_event.is_set():
                raise RuntimeError("检测到并行执行已请求停止，终止右臂预阶段")
            context.advance_execution_task(None, path.name)
            self._execute_csv(context, right, path)
            context.complete_execution_task()
        if plan.right_sync_csv_path is not None:
            if right is None:
                raise RuntimeError("同步阶段缺少右臂 runtime")
            context.advance_execution_task(
                plan.left_csv_path.name,
                plan.right_sync_csv_path.name,
            )
            self._execute_parallel(
                context,
                left,
                plan.left_csv_path,
                right,
                plan.right_sync_csv_path,
                stop_event,
                offset_updater,
            )
            context.complete_execution_task()
            left_executed = True
        if not left_executed:
            if right is not None:
                flush_pending_arm_segment(right)
            context.advance_execution_task(plan.left_csv_path.name, None)
            self._execute_csv(
                context,
                left,
                plan.left_csv_path,
                offset_updater=offset_updater,
            )
            context.complete_execution_task()
        if right is not None:
            if plan.right_post_stage_csv_paths:
                flush_pending_arm_segment(left)
            for path in plan.right_post_stage_csv_paths:
                if stop_event.is_set():
                    raise RuntimeError("检测到并行执行已请求停止，终止右臂后阶段")
                context.advance_execution_task(None, path.name)
                self._execute_csv(context, right, path)
                context.complete_execution_task()

    def _execute_parallel(
        self,
        context: ReplayContext,
        left: ReplayRuntime,
        left_path: Path,
        right: ReplayRuntime,
        right_path: Path,
        stop_event: threading.Event,
        offset_updater: GlobalOffsetUpdater | None,
    ) -> None:
        flush_pending_arm_segment(left)
        flush_pending_arm_segment(right)
        errors: list[BaseException] = []

        def worker(runtime: ReplayRuntime, csv_path: Path, is_left: bool) -> None:
            try:
                self._execute_csv(
                    context,
                    runtime,
                    csv_path,
                    flush_at_end=True,
                    offset_updater=offset_updater if is_left else None,
                )
                flush_pending_arm_segment(runtime)
            except BaseException as error:
                stop_event.set()
                errors.append(error)

        left_thread = threading.Thread(target=worker, args=(left, left_path, True), daemon=False)
        right_thread = threading.Thread(target=worker, args=(right, right_path, False), daemon=False)
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        if errors:
            raise RuntimeError("双臂同步阶段失败") from errors[0]

    # endregion

    # region CSV 与动作

    def _execute_csv(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        csv_path: Path,
        flush_at_end: bool = False,
        offset_updater: GlobalOffsetUpdater | None = None,
    ) -> None:
        """执行一个已在启动阶段预解析的 CSV。"""

        is_offset_trigger = offset_updater is not None and offset_updater.should_update_after(runtime, csv_path.name)
        rows = runtime.preloaded_rows_by_path.get(csv_path)
        if rows is None:
            raise RuntimeError(f"CSV 未在启动阶段预加载，拒绝在执行期读取文件：{csv_path}")
        if not rows:
            if flush_at_end:
                flush_pending_arm_segment(runtime)
            logger.warning(
                "CSV 是零字节或只有表头的占位文件，跳过执行 arm_side={} file={}",
                runtime.connected_arm.arm_side,
                csv_path.name,
            )
            return
        if is_offset_trigger:
            flush_pending_arm_segment(runtime)
        try:
            for row in rows:
                context.set_csv_progress(
                    runtime.connected_arm.arm_side,
                    csv_path.name,
                    row.row_index,
                    len(rows),
                )
                self._execute_row(runtime, row)
            if is_offset_trigger or flush_at_end:
                flush_pending_arm_segment(runtime)
            if is_offset_trigger and offset_updater is not None:
                time.sleep(runtime.settings.offset.capture_settle_delay_s)
                offset_updater.update(runtime)
        finally:
            context.clear_csv_progress(runtime.connected_arm.arm_side)

    def _execute_row(self, runtime: ReplayRuntime, row: ReplayRow) -> None:
        if runtime.stop_event.is_set():
            raise RuntimeError("检测到并行执行已请求停止")
        if row.action_type == "arm":
            runtime.pending_arm_rows.append(row)
            return
        flush_pending_arm_segment(runtime)
        if row.action_type == "gripper":
            execute_gripper_row(runtime, row)
            return
        if row.action_type == "m11":
            execute_m11_row(runtime, row)
            return
        if row.action_type == "lift":
            execute_lift_row(runtime, row)
            return
        raise ValueError(f"当前脚本暂不支持的记录类型: {row.action_type}")

    # endregion


def _plans_require_right_runtime(plans: list[CsvExecutionPlan]) -> bool:
    """判断执行计划是否真正引用右臂阶段。"""

    return any(
        plan.right_start_csv_path is not None
        or plan.right_pre_stage_csv_paths
        or plan.right_sync_csv_path is not None
        or plan.right_post_stage_csv_paths
        for plan in plans
    )


def _preload_plan_rows(plans: list[CsvExecutionPlan]) -> dict[Path, tuple[ReplayRow, ...]]:
    """按实际执行计划一次性解析所有 CSV，空占位文件保留为空元组。"""

    paths: list[Path] = []
    seen: set[Path] = set()
    for plan in plans:
        for path in (
            plan.left_csv_path,
            plan.right_start_csv_path,
            *plan.right_pre_stage_csv_paths,
            plan.right_sync_csv_path,
            *plan.right_post_stage_csv_paths,
        ):
            if path is not None and path not in seen:
                seen.add(path)
                paths.append(path)
    return {path: tuple(load_replay_rows(path)) for path in paths}
