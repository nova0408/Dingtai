"""命名动作计划的只读状态摘要。"""

from __future__ import annotations

from .action_sequence import ActionSequencePlan, SYNC_ACTION_ORDER
from .contracts import ReplayExecutionTaskStatus


def build_execution_task_statuses(
    plan: ActionSequencePlan,
) -> tuple[ReplayExecutionTaskStatus, ...]:
    """把左臂冻结动作列表展开为当前任务状态的静态摘要。

    执行器的两侧动作分别在线程中按各自 JSON 列表运行，右臂独立动作没有
    一个真实的全局先后位置。因此当前任务序号以左臂动作列表为基准；右臂
    独立动作仍通过 ``current_right_*`` 字段实时发布，不伪造一个执行器并不
    存在的“右臂追加阶段”。
    """

    tasks: list[tuple[str | None, str | None, bool]] = []
    right_by_function: dict[str, list[int]] = {}
    for right_position, action in enumerate(plan.right_actions):
        right_by_function.setdefault(action.item.function_name, []).append(right_position)
    consumed_right_positions: set[int] = set()
    for action in plan.left_actions:
        right_csv: str | None = None
        synchronized = action.item.function_name in SYNC_ACTION_ORDER
        if synchronized:
            candidates = right_by_function.get(action.item.function_name, [])
            for candidate_position in candidates:
                if candidate_position not in consumed_right_positions:
                    right_csv = plan.right_actions[candidate_position].csv_asset.path.name
                    consumed_right_positions.add(candidate_position)
                    break
        tasks.append((action.csv_asset.path.name, right_csv, synchronized))
    return tuple(
        ReplayExecutionTaskStatus(
            sequence=index,
            left_csv=left_csv,
            right_csv=right_csv,
            synchronized=synchronized,
        )
        for index, (left_csv, right_csv, synchronized) in enumerate(tasks, start=1)
    )
