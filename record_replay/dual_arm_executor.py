"""按命名动作顺序驱动双臂 CSV 回放。"""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from .action_sequence import (
    ActionSequencePlan,
    NamedActionPlan,
    SYNC_ACTION_ORDER,
)
from .arm_actions import flush_pending_arm_segment
from .charuco_offset import CharucoOffsetInitializer
from .context import ReplayContext
from .contracts import ReplayServiceState, ReplayRow
from .hand_actions import execute_gripper_row, execute_lift_row, execute_m11_row
from .offset_updater import GlobalOffsetUpdater
from .runtime import ReplayRuntime, close_runtime, create_runtime, prepare_runtime
from .settings import ReplayServiceSettings

SYNC_BARRIER_TIMEOUT_S = 60.0
"open_door/close_door 起点屏障最长等待时间。"


class DualArmExecutor:
    """双臂命名动作执行器；动作顺序来自 start 前冻结的 JSON。"""

    def execute(
        self,
        context: ReplayContext,
        plan: ActionSequencePlan,
        offset_updater: GlobalOffsetUpdater | None = None,
        charuco_initializer: CharucoOffsetInitializer | None = None,
    ) -> None:
        """创建 runtime 并按左右 JSON 列表执行有限轮次。"""

        stop_event = context.stop_event
        settings = context.config.settings
        _validate_offset_target_exclusivity(settings)
        preloaded_rows_by_path = _preload_action_rows(plan)
        left_runtime: ReplayRuntime | None = None
        right_runtime: ReplayRuntime | None = None
        open_door_start_barrier: threading.Barrier | None = None
        close_door_start_barrier: threading.Barrier | None = None
        errors: list[BaseException] = []
        try:
            _raise_if_stop_event(stop_event)
            left_runtime = create_runtime("left", stop_event, context.config.device_connection, settings)
            context.attach_runtimes(left_runtime, None)
            # 先登记已连接资源，再检查停止标志；这样 stop 与建连完成之间的竞态
            # 也能让 stop_devices 找到左臂并执行 robot.stop。
            _raise_if_stop_event(stop_event)
            left_runtime.preloaded_rows_by_path = preloaded_rows_by_path
            left_runtime.offset_target_action_names = settings.offset.target_action_names
            prepare_runtime(left_runtime)
            if plan.right_actions:
                _raise_if_stop_event(stop_event)
                right_runtime = create_runtime(
                    "right", stop_event, context.config.device_connection, settings
                )
                context.attach_runtimes(left_runtime, right_runtime)
                # 右臂同样必须先登记，再检查停止标志，避免 stop 请求遗漏刚建好的连接。
                _raise_if_stop_event(stop_event)
                right_runtime.preloaded_rows_by_path = preloaded_rows_by_path
                right_runtime.offset_target_action_names = settings.offset.target_action_names
                prepare_runtime(right_runtime)
            if charuco_initializer is not None and _plan_requires_charuco(plan, settings):
                charuco_initializer.initialize(
                    [left_runtime] if right_runtime is None else [left_runtime, right_runtime]
                )
            if right_runtime is not None:
                open_door_start_barrier = _barrier_for_action(plan, "open_door")
                close_door_start_barrier = _barrier_for_action(plan, "close_door")
                self._execute_both_sides(
                    context,
                    plan,
                    left_runtime,
                    right_runtime,
                    offset_updater,
                    open_door_start_barrier,
                    close_door_start_barrier,
                    errors,
                )
            else:
                self._execute_side(
                    context,
                    plan.left_actions,
                    (),
                    plan.loop_count,
                    left_runtime,
                    offset_updater,
                    None,
                    None,
                )
            flush_pending_arm_segment(left_runtime)
            if right_runtime is not None:
                flush_pending_arm_segment(right_runtime)
        finally:
            _break_barrier(open_door_start_barrier)
            _break_barrier(close_door_start_barrier)
            context.detach_runtimes()
            close_runtime(right_runtime)
            close_runtime(left_runtime)
        if errors:
            raise RuntimeError("命名动作执行失败") from errors[0]

    def _execute_both_sides(
        self,
        context: ReplayContext,
        plan: ActionSequencePlan,
        left_runtime: ReplayRuntime,
        right_runtime: ReplayRuntime,
        offset_updater: GlobalOffsetUpdater | None,
        open_door_start_barrier: threading.Barrier | None,
        close_door_start_barrier: threading.Barrier | None,
        errors: list[BaseException],
    ) -> None:
        """并行运行两侧列表；只在 open/close 动作起点使用显式屏障。"""

        def run_left() -> None:
            try:
                self._execute_side(
                    context,
                    plan.left_actions,
                    plan.right_actions,
                    plan.loop_count,
                    left_runtime,
                    offset_updater,
                    open_door_start_barrier,
                    close_door_start_barrier,
                )
            except BaseException as error:
                errors.append(error)
                context.stop_event.set()
                _break_barrier(open_door_start_barrier)
                _break_barrier(close_door_start_barrier)

        def run_right() -> None:
            try:
                self._execute_side(
                    context,
                    plan.right_actions,
                    (),
                    plan.loop_count,
                    right_runtime,
                    None,
                    open_door_start_barrier,
                    close_door_start_barrier,
                )
            except BaseException as error:
                errors.append(error)
                context.stop_event.set()
                _break_barrier(open_door_start_barrier)
                _break_barrier(close_door_start_barrier)

        left_thread = threading.Thread(target=run_left, name="record-replay-left", daemon=False)
        right_thread = threading.Thread(target=run_right, name="record-replay-right", daemon=False)
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()

    def _execute_side(
        self,
        context: ReplayContext,
        actions: tuple[NamedActionPlan, ...],
        synchronized_peer_actions: tuple[NamedActionPlan, ...],
        loop_count: int,
        runtime: ReplayRuntime,
        offset_updater: GlobalOffsetUpdater | None,
        open_door_start_barrier: threading.Barrier | None,
        close_door_start_barrier: threading.Barrier | None,
    ) -> None:
        """按一侧 JSON 顺序执行指定循环次数。"""

        synchronized_peer_csvs = _build_synchronized_peer_csvs(
            actions,
            synchronized_peer_actions,
        )
        for _ in range(loop_count):
            for action_position, action in enumerate(actions):
                if runtime.stop_event.is_set():
                    raise RuntimeError("检测到停止请求，终止后续命名动作")
                runtime.offset_source = "none"
                context.refresh_offset_statuses()
                if runtime.connected_arm.arm_side == "left":
                    context.advance_execution_task(
                        action.csv_asset.path.name,
                        synchronized_peer_csvs[action_position],
                    )
                context.set_state(
                    ReplayServiceState.BUSY,
                    left_csv_state=(
                        action.csv_asset.path.stem
                        if runtime.connected_arm.arm_side == "left"
                        else context.snapshot().left_csv_state
                    ),
                    plan_index=None,
                )
                barrier = _barrier_for_name(
                    action.item.function_name,
                    open_door_start_barrier,
                    close_door_start_barrier,
                )
                _wait_action_start(barrier, action.item.function_name, runtime)
                self._execute_action(context, runtime, action, offset_updater)
                context.refresh_offset_statuses()
                if runtime.connected_arm.arm_side == "left":
                    context.complete_execution_task()

    def _execute_action(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
        offset_updater: GlobalOffsetUpdater | None,
    ) -> None:
        """按 record_left 命名显式分发，不使用反射或动态注册表。"""

        _raise_if_stop_event(runtime.stop_event)
        match action.item.function_name:
            case "go_out":
                self._execute_go_out(context, runtime, action)
            case "open_door":
                self._execute_open_door(context, runtime, action)
            case "before_calibration":
                self._execute_before_calibration(context, runtime, action)
            case "calibration":
                self._execute_calibration(context, runtime, action)
            case "get_tray":
                self._execute_get_tray(context, runtime, action)
            case "after_get_tray":
                self._execute_after_get_tray(context, runtime, action)
            case "put_tray":
                self._execute_put_tray(context, runtime, action)
            case "before_get_new_tray":
                self._execute_before_get_new_tray(context, runtime, action)
            case "get_new_tray":
                self._execute_get_new_tray(context, runtime, action)
            case "before_put_new_tray":
                self._execute_before_put_new_tray(context, runtime, action)
            case "put_new_tray":
                self._execute_put_new_tray(context, runtime, action)
            case "calibration_new_tray":
                self._execute_calibration_new_tray(context, runtime, action)
            case "after_put_new_tray":
                self._execute_after_put_new_tray(context, runtime, action)
            case "close_door":
                self._execute_close_door(context, runtime, action)
            case "go_home":
                self._execute_go_home(context, runtime, action)
            case _:
                raise RuntimeError(f"未实现的命名动作：{action.item.function_name}")
        if not runtime.preloaded_rows_by_path.get(action.csv_asset.path):
            return
        self._update_offset_after_configured_action(runtime, action, offset_updater)

    def _execute_capture_action(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        """慢速到位后等待；具体算法由命名动作显式调用。"""

        runtime.current_action = action
        self._execute_csv(context, runtime, action)
        if not runtime.preloaded_rows_by_path.get(action.csv_asset.path):
            logger.info("拍摄动作 {} CSV 为空，按未实现动作留空", action.item.function_name)
            return
        _wait_settle_delay(runtime, action.item.settle_delay)

    def _execute_fast_action(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        """执行开门、转移和收尾等快速动作。"""

        runtime.current_action = action
        self._execute_csv(context, runtime, action)

    def _execute_precise_action(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        """执行精确动作；zone=0 由 arm_actions 再次强制。"""

        runtime.current_action = action
        self._execute_csv(context, runtime, action)

    def _execute_go_out(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_open_door(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_before_calibration(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_calibration(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_capture_action(context, runtime, action)

    def _update_offset_after_configured_action(
        self,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
        offset_updater: GlobalOffsetUpdater | None,
    ) -> None:
        """在统一 JSON 指定的动作完成后显式执行三球检测。"""

        if runtime.connected_arm.arm_side != "left":
            return
        if runtime.settings.offset.calculate_after_action_name != action.item.function_name:
            return
        if offset_updater is None:
            raise RuntimeError(f"{action.item.function_name} 动作缺少三球检测服务")
        offset_updater.update(runtime)

    def _execute_get_tray(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_precise_action(context, runtime, action)

    def _execute_after_get_tray(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_put_tray(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_precise_action(context, runtime, action)

    def _execute_before_get_new_tray(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_get_new_tray(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_precise_action(context, runtime, action)

    def _execute_before_put_new_tray(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_put_new_tray(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_precise_action(context, runtime, action)

    def _execute_calibration_new_tray(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_capture_action(context, runtime, action)

    def _execute_after_put_new_tray(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_close_door(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_go_home(self, context: ReplayContext, runtime: ReplayRuntime, action: NamedActionPlan) -> None:
        self._execute_fast_action(context, runtime, action)

    def _execute_csv(
        self,
        context: ReplayContext,
        runtime: ReplayRuntime,
        action: NamedActionPlan,
    ) -> None:
        """执行启动阶段预解析的一个命名动作 CSV。"""

        csv_path = action.csv_asset.path
        rows = runtime.preloaded_rows_by_path.get(csv_path)
        if rows is None:
            raise RuntimeError(f"CSV 未在启动阶段预加载，拒绝执行期读取：{csv_path}")
        if not rows:
            logger.warning("CSV 无数据，跳过命名动作 file={}", csv_path.name)
            return
        last_arm_position = max(
            (position for position, row in enumerate(rows) if row.action_type == "arm"),
            default=-1,
        )
        for position, row in enumerate(rows):
            context.set_csv_progress(
                runtime.connected_arm.arm_side,
                csv_path.name,
                row.row_index,
                len(rows),
                action.item.function_name,
                action.item.index,
            )
            self._execute_row(runtime, row, position > last_arm_position)
        flush_pending_arm_segment(runtime)
        context.clear_csv_progress(runtime.connected_arm.arm_side)

    def _execute_row(
        self,
        runtime: ReplayRuntime,
        row: ReplayRow,
        final_arm_segment: bool,
    ) -> None:
        """执行一条 CSV 数据行。"""

        if runtime.stop_event.is_set():
            raise RuntimeError("检测到停止请求")
        if row.action_type == "arm":
            runtime.pending_arm_rows.append(row)
            return
        flush_pending_arm_segment(runtime, final_arm_segment)
        if row.action_type == "gripper":
            execute_gripper_row(runtime, row)
            return
        if row.action_type == "m11":
            execute_m11_row(runtime, row)
            return
        if row.action_type == "lift":
            execute_lift_row(runtime, row)
            return
        raise ValueError(f"当前脚本暂不支持的记录类型：{row.action_type}")


def _preload_action_rows(plan: ActionSequencePlan) -> dict[Path, tuple[ReplayRow, ...]]:
    """返回 start 前已冻结的引用 CSV 行，不重新读取磁盘。"""

    return dict(plan.preloaded_rows_by_path)


def _raise_if_stop_event(stop_event: threading.Event) -> None:
    """在创建 runtime 和进入命名动作前阻止停止后的后续流程。"""

    if stop_event.is_set():
        raise RuntimeError("检测到停止请求，禁止继续执行回放流程")


def _validate_offset_target_exclusivity(settings: ReplayServiceSettings) -> None:
    """启动前拒绝同一命名动作同时配置两种 offset。"""

    global_targets = settings.offset.target_action_names
    left_overlap = global_targets.intersection(settings.offset.left_charuco_target_action_names)
    right_overlap = global_targets.intersection(settings.offset.right_charuco_target_action_names)
    overlap = sorted(left_overlap | right_overlap)
    if overlap:
        raise RuntimeError("命名动作不能同时配置头部 offset 与三球 offset：" + ", ".join(overlap))


def _plan_requires_charuco(plan: ActionSequencePlan, settings: ReplayServiceSettings) -> bool:
    """检查计划是否包含当前配置声明的 ChArUco 目标动作。"""

    return (
        any(
            action.item.function_name in settings.offset.left_charuco_target_action_names
            for action in plan.left_actions
        )
        or any(
            action.item.function_name in settings.offset.right_charuco_target_action_names
            for action in plan.right_actions
        )
    )


def _barrier_for_action(plan: ActionSequencePlan, action_name: str) -> threading.Barrier | None:
    """只为确实出现在左右两侧的命名同步动作创建屏障。"""

    left_count = sum(action.item.function_name == action_name for action in plan.left_actions)
    right_count = sum(action.item.function_name == action_name for action in plan.right_actions)
    return threading.Barrier(2) if left_count == right_count and left_count > 0 else None


def _build_synchronized_peer_csvs(
    actions: tuple[NamedActionPlan, ...],
    peer_actions: tuple[NamedActionPlan, ...],
) -> tuple[str | None, ...]:
    """按同步动作名和出现次序绑定一侧的对应 CSV。"""

    consumed_peer_positions: set[int] = set()
    peer_csvs: list[str | None] = []
    for action in actions:
        peer_csv: str | None = None
        if action.item.function_name in SYNC_ACTION_ORDER:
            for peer_position, peer_action in enumerate(peer_actions):
                if peer_position in consumed_peer_positions:
                    continue
                if peer_action.item.function_name != action.item.function_name:
                    continue
                consumed_peer_positions.add(peer_position)
                peer_csv = peer_action.csv_asset.path.name
                break
        peer_csvs.append(peer_csv)
    return tuple(peer_csvs)


def _barrier_for_name(
    action_name: str,
    open_door_start_barrier: threading.Barrier | None,
    close_door_start_barrier: threading.Barrier | None,
) -> threading.Barrier | None:
    """按两个固定同步动作名选择屏障。"""

    if action_name == "open_door":
        return open_door_start_barrier
    if action_name == "close_door":
        return close_door_start_barrier
    return None


def _wait_action_start(
    barrier: threading.Barrier | None,
    action_name: str,
    runtime: ReplayRuntime,
) -> None:
    """等待命名动作起点同步；不等待动作终点。"""

    if barrier is None:
        return
    try:
        barrier.wait(timeout=SYNC_BARRIER_TIMEOUT_S)
    except threading.BrokenBarrierError as error:
        raise RuntimeError(f"{action_name} 起点同步失败") from error
    if runtime.stop_event.is_set():
        raise RuntimeError(f"{action_name} 起点同步后收到停止请求")


def _break_barrier(barrier: threading.Barrier | None) -> None:
    """唤醒等待中的固定屏障。"""

    if barrier is not None:
        barrier.abort()


def _wait_settle_delay(runtime: ReplayRuntime, delay_s: float) -> None:
    """等待拍摄稳定时间，并允许停止事件立即打断等待。"""

    if runtime.stop_event.wait(timeout=delay_s):
        raise RuntimeError("检测到停止请求，终止拍摄稳定等待")
