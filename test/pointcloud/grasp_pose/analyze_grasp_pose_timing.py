from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _setup_matplotlib_chinese_font() -> None:
    # 按常见平台顺序回退，避免中文乱码。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class StepStat:
    step_name: str
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    compute_seq: int
    elapsed_ms: float
    status: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析抓取位姿流水线耗时 CSV 并生成统计图。")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("logs/grasp_pose_pipeline_timing.csv"),
        help="输入耗时 CSV 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/grasp_pose_timing_analysis"),
        help="图表和统计 CSV 输出目录",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="可选 run_id 过滤。为空时默认使用 CSV 里的最新 run_id。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="交互显示图表（IDE 直接运行时可用）。",
    )
    parser.add_argument(
        "--skip-initial-compute-frames",
        type=int,
        default=-1,
        help="手动跳过起始计算帧数量。-1 表示自动估计启动阶段并跳过。",
    )
    return parser.parse_args()


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pick_run_id(rows: list[dict[str, str]], run_id: str) -> str:
    all_ids = [r["run_id"] for r in rows if r.get("run_id")]
    if len(all_ids) == 0:
        raise RuntimeError("CSV 不包含 run_id")
    if run_id:
        return run_id
    # 默认选择最后出现的 run_id（通常是最新一次运行）
    return all_ids[-1]


def _compute_step_stats(rows: list[dict[str, str]]) -> list[StepStat]:
    grouped: dict[str, list[float]] = {}
    for r in rows:
        step = r["step_name"]
        if step == "frame_total":
            continue
        if r.get("status", "ok") != "ok":
            continue
        grouped.setdefault(step, []).append(float(r["elapsed_ms"]))
    stats: list[StepStat] = []
    for step, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        stats.append(
            StepStat(
                step_name=step,
                count=int(arr.size),
                mean_ms=float(np.mean(arr)),
                p50_ms=float(np.percentile(arr, 50)),
                p90_ms=float(np.percentile(arr, 90)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                max_ms=float(np.max(arr)),
            )
        )
    stats.sort(key=lambda s: s.mean_ms, reverse=True)
    return stats


def _collect_frames(rows: list[dict[str, str]]) -> list[FrameInfo]:
    frame_rows = [r for r in rows if r["step_name"] == "frame_total"]
    frame_rows.sort(key=lambda r: int(r["frame_idx"]))
    out: list[FrameInfo] = []
    for i, r in enumerate(frame_rows):
        out.append(
            FrameInfo(
                frame_idx=int(r["frame_idx"]),
                compute_seq=i + 1,
                elapsed_ms=float(r["elapsed_ms"]),
                status=r.get("status", "ok"),
            )
        )
    return out


def _detect_warmup_skip_count(frames: list[FrameInfo]) -> int:
    if len(frames) <= 6:
        return 0
    arr = np.asarray([f.elapsed_ms for f in frames], dtype=np.float64)
    tail = arr[max(1, int(len(arr) * 0.4)) :]
    baseline = float(np.median(tail))
    threshold = baseline * 1.35
    # 连续 3 帧进入稳定区后，之前视为启动阶段
    for i in range(0, len(arr) - 2):
        if arr[i] <= threshold and arr[i + 1] <= threshold and arr[i + 2] <= threshold:
            return i
    return min(3, max(0, len(arr) // 5))


def _build_stage_map(frames: list[FrameInfo]) -> dict[int, str]:
    n = len(frames)
    if n == 0:
        return {}
    stage_map: dict[int, str] = {}
    s1 = int(np.ceil(n / 3.0))
    s2 = int(np.ceil(2 * n / 3.0))
    for i, f in enumerate(frames, start=1):
        if i <= s1:
            stage = "阶段1-前段"
        elif i <= s2:
            stage = "阶段2-中段"
        else:
            stage = "阶段3-后段"
        stage_map[f.frame_idx] = stage
    return stage_map


def _save_stage_step_summary_csv(rows: list[dict[str, str]], stage_map: dict[int, str], out_csv: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["step_name"] == "frame_total" or r.get("status", "ok") != "ok":
            continue
        frame_idx = int(r["frame_idx"])
        stage = stage_map.get(frame_idx, "未知阶段")
        key = (stage, r["step_name"])
        grouped.setdefault(key, []).append(float(r["elapsed_ms"]))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "step_name", "count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms"])
        keys = sorted(grouped.keys(), key=lambda k: (k[0], -float(np.mean(np.asarray(grouped[k], dtype=np.float64)))))
        for stage, step_name in keys:
            arr = np.asarray(grouped[(stage, step_name)], dtype=np.float64)
            writer.writerow(
                [
                    stage,
                    step_name,
                    int(arr.size),
                    f"{np.mean(arr):.3f}",
                    f"{np.percentile(arr, 50):.3f}",
                    f"{np.percentile(arr, 90):.3f}",
                    f"{np.percentile(arr, 95):.3f}",
                    f"{np.max(arr):.3f}",
                ]
            )


def _save_summary_csv(stats: list[StepStat], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step_name", "count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"])
        for s in stats:
            writer.writerow(
                [
                    s.step_name,
                    s.count,
                    f"{s.mean_ms:.3f}",
                    f"{s.p50_ms:.3f}",
                    f"{s.p90_ms:.3f}",
                    f"{s.p95_ms:.3f}",
                    f"{s.p99_ms:.3f}",
                    f"{s.max_ms:.3f}",
                ]
            )


def _plot_step_mean_bar(stats: list[StepStat], out_png: Path, show: bool) -> None:
    names = [s.step_name for s in stats]
    means = [s.mean_ms for s in stats]
    y = np.arange(len(names))
    plt.figure(figsize=(14, max(6, int(len(names) * 0.42))))
    plt.barh(y, means)
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.xlabel("平均耗时 (ms)")
    plt.title("各步骤平均耗时")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show(block=False)
    plt.close()


def _plot_frame_total_trend(frames: list[FrameInfo], out_png: Path, show: bool) -> None:
    frame_idx = np.asarray([f.compute_seq for f in frames], dtype=np.int32)
    elapsed = np.asarray([f.elapsed_ms for f in frames], dtype=np.float64)
    status = [f.status for f in frames]
    plt.figure(figsize=(14, 6))
    plt.plot(frame_idx, elapsed, linewidth=1.2, label="计算帧总耗时")
    bad = np.asarray([i for i, s in enumerate(status) if s != "ok"], dtype=np.int32)
    if bad.size > 0:
        plt.scatter(frame_idx[bad], elapsed[bad], color="red", s=30, label="异常帧")
    p95 = float(np.percentile(elapsed, 95))
    plt.axhline(p95, linestyle="--", linewidth=1.0, label=f"P95={p95:.1f}ms")
    plt.xlabel("计算帧序号")
    plt.ylabel("耗时 (ms)")
    plt.title("计算帧总耗时趋势（已排除启动阶段）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show(block=False)
    plt.close()


def _plot_stage_step_mean(rows: list[dict[str, str]], stage_map: dict[int, str], out_png: Path, show: bool) -> None:
    stage_order = ["阶段1-前段", "阶段2-中段", "阶段3-后段"]
    grouped: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["step_name"] == "frame_total" or r.get("status", "ok") != "ok":
            continue
        stage = stage_map.get(int(r["frame_idx"]), "未知阶段")
        grouped.setdefault((stage, r["step_name"]), []).append(float(r["elapsed_ms"]))
    all_steps = sorted({k[1] for k in grouped.keys()})
    # 只画平均耗时前 8 步，突出重点
    global_means = []
    for step in all_steps:
        vals = []
        for stg in stage_order:
            vals.extend(grouped.get((stg, step), []))
        if len(vals) > 0:
            global_means.append((step, float(np.mean(np.asarray(vals, dtype=np.float64)))))
    top_steps = [x[0] for x in sorted(global_means, key=lambda x: x[1], reverse=True)[:8]]
    x = np.arange(len(top_steps))
    w = 0.24
    plt.figure(figsize=(14, 6))
    for i, stg in enumerate(stage_order):
        y = []
        for step in top_steps:
            arr = np.asarray(grouped.get((stg, step), []), dtype=np.float64)
            y.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
        plt.bar(x + (i - 1) * w, y, width=w, label=stg)
    plt.xticks(x, top_steps, rotation=25, ha="right")
    plt.ylabel("平均耗时 (ms)")
    plt.title("分阶段步骤耗时对比（计算帧）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show(block=False)
    plt.close()


def _plot_top_step_box(
    rows: list[dict[str, str]], stats: list[StepStat], out_png: Path, show: bool, topn: int = 8
) -> None:
    top_steps = [s.step_name for s in stats[: max(1, topn)]]
    series: list[np.ndarray] = []
    labels: list[str] = []
    for step in top_steps:
        vals = [
            float(r["elapsed_ms"])
            for r in rows
            if r["step_name"] == step and r.get("status", "ok") == "ok"
        ]
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        # 用 p99 截断，减少单点极端值遮挡整体分布
        hi = float(np.percentile(arr, 99))
        arr = arr[arr <= hi]
        if arr.size == 0:
            continue
        series.append(arr)
        labels.append(step)
    plt.figure(figsize=(14, max(6, int(len(labels) * 0.7))))
    plt.boxplot(series, tick_labels=labels, showfliers=False, vert=False)
    plt.xlabel("耗时 (ms)")
    plt.title("主要步骤耗时分布（隐藏离群点，按 P99 截断）")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show(block=False)
    plt.close()


def main() -> None:
    args = _parse_args()
    _setup_matplotlib_chinese_font()
    rows = _load_rows(args.csv)
    run_id = _pick_run_id(rows, args.run_id)
    filtered = [r for r in rows if r.get("run_id", "") == run_id]
    if len(filtered) == 0:
        raise RuntimeError(f"run_id={run_id} 在 CSV 中无记录")

    frames_all = _collect_frames(filtered)
    if len(frames_all) == 0:
        raise RuntimeError("当前 run_id 不包含 frame_total 记录，无法分析计算帧")
    if int(args.skip_initial_compute_frames) >= 0:
        skip_n = int(args.skip_initial_compute_frames)
    else:
        skip_n = _detect_warmup_skip_count(frames_all)
    skip_n = int(np.clip(skip_n, 0, max(0, len(frames_all) - 1)))
    keep_frames = frames_all[skip_n:]
    keep_frame_idx = {f.frame_idx for f in keep_frames}
    filtered_kept = [r for r in filtered if int(r["frame_idx"]) in keep_frame_idx]
    stage_map = _build_stage_map(keep_frames)

    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = _compute_step_stats(filtered_kept)
    _save_summary_csv(stats, out_dir / "step_summary.csv")
    _save_stage_step_summary_csv(filtered_kept, stage_map, out_dir / "stage_step_summary.csv")
    _plot_step_mean_bar(stats, out_dir / "step_mean_bar.png", show=bool(args.show))
    _plot_frame_total_trend(keep_frames, out_dir / "frame_total_trend.png", show=bool(args.show))
    _plot_stage_step_mean(filtered_kept, stage_map, out_dir / "stage_step_mean_by_phase.png", show=bool(args.show))
    _plot_top_step_box(filtered_kept, stats, out_dir / "top_steps_boxplot.png", show=bool(args.show), topn=8)

    frame_ms = np.asarray([f.elapsed_ms for f in keep_frames], dtype=np.float64)
    err_count = int(np.count_nonzero(np.asarray([f.status != "ok" for f in keep_frames], dtype=bool)))
    print(f"run_id: {run_id}")
    print(f"总计算帧: {len(frames_all)}，已跳过启动帧: {skip_n}，纳入分析帧: {len(keep_frames)}")
    print(f"纳入分析异常帧: {err_count}")
    print(
        "计算帧总耗时(ms): "
        f"mean={np.mean(frame_ms):.2f}, p90={np.percentile(frame_ms,90):.2f}, "
        f"p95={np.percentile(frame_ms,95):.2f}, p99={np.percentile(frame_ms,99):.2f}, max={np.max(frame_ms):.2f}"
    )
    print(f"outputs: {out_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
