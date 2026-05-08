from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "runtime" / "three_plane"

TIMING_COLUMNS = (
    "prepare_xyz_rgb",
    "project_points",
    "prepare_preview",
    "tray",
    "tray_detect_async",
    "tray_snapshot_age",
    "tray_predict_shift",
    "tray_shift_dx",
    "tray_shift_dy",
    "tray_shift_score",
    "pose",
    "draw_overlay",
    "total",
)
TIME_COLUMNS = (
    "prepare_xyz_rgb",
    "project_points",
    "prepare_preview",
    "tray",
    "tray_detect_async",
    "pose",
    "draw_overlay",
    "total",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 three-plane pipeline 性能 CSV（含绘制耗时）")
    parser.add_argument("--csv", type=str, default="", help="指定 CSV 文件路径；不传则自动取 runtime/three_plane 下最新文件")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示图表")
    return parser.parse_args()


def _resolve_csv_path(csv_arg: str) -> Path:
    if csv_arg.strip():
        path = Path(csv_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV 不存在：{path}")
        return path
    candidates = sorted(DEFAULT_RUNTIME_DIR.glob("three_plane_timing_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"未在 {DEFAULT_RUNTIME_DIR} 找到 three_plane_timing_*.csv")
    return candidates[0]


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _stats(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    return (
        f"n={arr.size}, mean={float(np.mean(arr)):.2f}ms, "
        f"p50={float(p50):.2f}ms, p90={float(p90):.2f}ms, "
        f"p99={float(p99):.2f}ms, max={float(np.max(arr)):.2f}ms"
    )


def _build_frame_indices(rows: list[dict[str, str]]) -> np.ndarray:
    idx_list: list[int] = []
    for i, row in enumerate(rows):
        try:
            idx_list.append(int(row.get("frame_idx", str(i + 1))))
        except ValueError:
            idx_list.append(i + 1)
    return np.asarray(idx_list, dtype=np.int32)


def _save_plots(csv_path: Path, frames: np.ndarray, numeric: dict[str, np.ndarray]) -> list[Path]:
    output_dir = csv_path.parent
    stem = csv_path.stem
    method_cols = [c for c in TIME_COLUMNS if c != "total"]
    out_paths: list[Path] = []

    # 图1：每个计算帧的总耗时
    fig1 = plt.figure(figsize=(14, 5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(frames, numeric["total"], color="#1f77b4", linewidth=1.5, label="total")
    ax1.set_title("Three-plane 每帧总耗时")
    ax1.set_xlabel("frame_idx")
    ax1.set_ylabel("time (ms)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")
    p1 = output_dir / f"{stem}_frame_total.png"
    fig1.tight_layout()
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)
    out_paths.append(p1)

    # 图2：各方法每帧耗时曲线（不含 total）
    fig2 = plt.figure(figsize=(14, 7))
    ax2 = fig2.add_subplot(111)
    for name in method_cols:
        ax2.plot(frames, numeric[name], linewidth=1.2, label=name)
    ax2.set_title("Three-plane 各方法每帧耗时")
    ax2.set_xlabel("frame_idx")
    ax2.set_ylabel("time (ms)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", ncol=2)
    p2 = output_dir / f"{stem}_frame_methods.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    out_paths.append(p2)

    # 图3：不同方法耗时对比（mean / p90 / p99）
    means = np.asarray([float(np.mean(numeric[name])) for name in method_cols], dtype=np.float64)
    p90 = np.asarray([float(np.percentile(numeric[name], 90)) for name in method_cols], dtype=np.float64)
    p99 = np.asarray([float(np.percentile(numeric[name], 99)) for name in method_cols], dtype=np.float64)
    x = np.arange(len(method_cols), dtype=np.float64)
    width = 0.25
    fig3 = plt.figure(figsize=(14, 6))
    ax3 = fig3.add_subplot(111)
    ax3.bar(x - width, means, width=width, label="mean")
    ax3.bar(x, p90, width=width, label="p90")
    ax3.bar(x + width, p99, width=width, label="p99")
    ax3.set_title("Three-plane 方法耗时对比")
    ax3.set_xlabel("method")
    ax3.set_ylabel("time (ms)")
    ax3.set_xticks(x, method_cols, rotation=20, ha="right")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(loc="upper right")
    p3 = output_dir / f"{stem}_method_compare.png"
    fig3.tight_layout()
    fig3.savefig(p3, dpi=140)
    plt.close(fig3)
    out_paths.append(p3)
    return out_paths


def _show_plots(frames: np.ndarray, numeric: dict[str, np.ndarray]) -> None:
    method_cols = [c for c in TIME_COLUMNS if c != "total"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].plot(frames, numeric["total"], color="#1f77b4", linewidth=1.5)
    axes[0].set_title("Three-plane 每帧总耗时")
    axes[0].set_xlabel("frame_idx")
    axes[0].set_ylabel("time (ms)")
    axes[0].grid(True, alpha=0.25)

    for name in method_cols:
        axes[1].plot(frames, numeric[name], linewidth=1.2, label=name)
    axes[1].set_title("Three-plane 各方法每帧耗时")
    axes[1].set_xlabel("frame_idx")
    axes[1].set_ylabel("time (ms)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", ncol=2)

    means = np.asarray([float(np.mean(numeric[name])) for name in method_cols], dtype=np.float64)
    p90 = np.asarray([float(np.percentile(numeric[name], 90)) for name in method_cols], dtype=np.float64)
    p99 = np.asarray([float(np.percentile(numeric[name], 99)) for name in method_cols], dtype=np.float64)
    x = np.arange(len(method_cols), dtype=np.float64)
    width = 0.25
    axes[2].bar(x - width, means, width=width, label="mean")
    axes[2].bar(x, p90, width=width, label="p90")
    axes[2].bar(x + width, p99, width=width, label="p99")
    axes[2].set_title("Three-plane 方法耗时对比")
    axes[2].set_xlabel("method")
    axes[2].set_ylabel("time (ms)")
    axes[2].set_xticks(x, method_cols, rotation=20, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = _parse_args()
    csv_path = _resolve_csv_path(args.csv)
    rows = _read_rows(csv_path)
    print(f"CSV: {csv_path}")
    if not rows:
        print("CSV 为空")
        return

    numeric: dict[str, np.ndarray] = {}
    frames = _build_frame_indices(rows)
    for col in TIMING_COLUMNS:
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row.get(col, "0") or 0.0))
            except ValueError:
                values.append(0.0)
        numeric[col] = np.asarray(values, dtype=np.float64)

    print("\n逐项耗时统计：")
    for col in TIMING_COLUMNS:
        print(f"- {col}: {_stats(numeric[col])}")

    total_mean = float(np.mean(numeric["total"])) if numeric["total"].size else 0.0
    print("\n平均耗时占比（相对 total mean）：")
    for col in TIMING_COLUMNS:
        if col == "total":
            continue
        ratio = 0.0 if total_mean <= 1e-9 else float(np.mean(numeric[col])) / total_mean * 100.0
        print(f"- {col}: {ratio:.1f}%")

    print("\n最慢帧（按 total）Top 10：")
    totals = numeric["total"]
    top_idx = np.argsort(-totals)[: min(10, totals.size)]
    for rank, idx in enumerate(top_idx, start=1):
        row = rows[int(idx)]
        print(
            f"{rank}. frame={row.get('frame_idx', '?')}, total={totals[int(idx)]:.2f}ms, "
            f"draw_overlay={numeric['draw_overlay'][int(idx)]:.2f}ms, "
            f"pose={numeric['pose'][int(idx)]:.2f}ms, tray={numeric['tray'][int(idx)]:.2f}ms"
        )

    plot_paths = _save_plots(csv_path=csv_path, frames=frames, numeric=numeric)
    print("\n图表输出：")
    for p in plot_paths:
        print(f"- {p}")
    if args.show:
        _show_plots(frames=frames, numeric=numeric)


if __name__ == "__main__":
    main()
