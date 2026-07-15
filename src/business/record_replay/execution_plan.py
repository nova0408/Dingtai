"""左右臂 CSV 的确定性阶段编排。"""

from __future__ import annotations

from pathlib import Path

from .contracts import CsvExecutionPlan
from .csv_repository import extract_csv_sequence, extract_sync_csv_sequence


def build_execution_plans(left_csv_paths: list[Path], right_csv_paths: list[Path]) -> list[CsvExecutionPlan]:
    """按旧自动回放规则生成双臂执行计划。"""

    right_by_sequence = {extract_csv_sequence(path.name): path for path in right_csv_paths}
    right_sequences = sorted(right_by_sequence)
    consumed: set[int] = set()
    specs: list[tuple[Path, Path | None, tuple[Path, ...], Path | None, bool]] = []
    for left_index, left_path in enumerate(left_csv_paths):
        left_sequence = extract_csv_sequence(left_path.name)
        sync_sequence = extract_sync_csv_sequence(left_path.name)
        start_path = right_by_sequence.get(left_sequence) if left_index == 0 else None
        start_together = start_path is not None and left_index == 0
        if start_path is not None:
            consumed.add(left_sequence)
        pre_sequences: list[int] = []
        sync_path = None
        if sync_sequence is not None:
            sync_path = right_by_sequence.get(sync_sequence)
            if sync_path is None:
                raise RuntimeError(
                    f"左臂 CSV 声明同步右臂文件但不存在：left={left_path.name} right_seq={sync_sequence:02d}"
                )
            for sequence in right_sequences:
                if sequence not in consumed and sequence < sync_sequence:
                    pre_sequences.append(sequence)
                    consumed.add(sequence)
            consumed.add(sync_sequence)
        elif left_index > 0:
            future_sync = next(
                (
                    extract_sync_csv_sequence(path.name)
                    for path in left_csv_paths[left_index + 1 :]
                    if extract_sync_csv_sequence(path.name) is not None
                ),
                None,
            )
            upper_bound = left_sequence if future_sync is None else future_sync
            for sequence in right_sequences:
                if sequence not in consumed and sequence < upper_bound:
                    pre_sequences.append(sequence)
                    consumed.add(sequence)
        specs.append(
            (left_path, start_path, tuple(right_by_sequence[item] for item in pre_sequences), sync_path, start_together)
        )
    trailing = tuple(right_by_sequence[item] for item in right_sequences if item not in consumed)
    return [
        CsvExecutionPlan(left, start, pre, sync, trailing if index == len(specs) - 1 else (), together)
        for index, (left, start, pre, sync, together) in enumerate(specs)
    ]
