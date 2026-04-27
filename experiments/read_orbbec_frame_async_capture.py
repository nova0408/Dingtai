from __future__ import annotations

"""Orbbec 异步采集结果读取脚本（配套 test_orbbec_frame_async_capture.py）。"""

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# region 默认参数（优先在这里直接改）

# =======================默认输入目录=====================================

DEFAULT_INPUT_DIR = Path(".") / Path(
    "260427"
)  # 填入 Path 格式的输入目录，类似于 DEFAULT_INPUT_DIR=Path(".") / Path("260427")

# =======================默认输入目录=====================================

DEFAULT_PREVIEW = True  # 是否开启预览窗口（默认开启）
DEFAULT_MAX_FRAMES = 0  # 最多读取帧数，0 表示全部
DEFAULT_SELECTED_ROOT_NAME = "_selected_frames"  # 预览另存根目录名（位于输入目录下）
DEFAULT_SELECTED_RUN_PREFIX = "pick_"  # 每次运行的另存目录前缀
DEFAULT_WINDOW_NAME = "Orbbec frame reader"  # 预览窗口名（ASCII）
DEFAULT_WINDOW_WIDTH = 1400  # 预览窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 预览窗口高度，单位 像素
# endregion

# region 用法说明（单列）
USAGE_GUIDE = """
[数据格式]
1. 目录名：yymmdd，例如 260427
2. 每帧文件名：
   frame_{frame_index}_rgb_{ss}.png
   frame_{frame_index}_depth_{ss}.png
   frame_{frame_index}_pcd_{ss}.pcd
3. depth 为灰度 png；pcd 为 ASCII（x y z，且可选 r g b）

[运行方式]
1. IDE 直跑：使用 DEFAULT_* 默认参数
2. CLI 覆盖：
   python experiments/read_orbbec_frame_async_capture.py --input-dir 260427 --preview

[预览键位]
1. Space / N：下一帧
2. P：上一帧
3. S：另存当前帧（复制 rgb/depth/pcd）
4. Q / ESC：退出

[另存目录]
1. 输入目录下自动创建：_selected_frames/
2. 每次运行新建唯一目录：pick_YYYYmmdd_HHMMSS（重名自动加后缀）
""".strip()
# endregion


_FRAME_FILE_PATTERN = re.compile(
    r"^frame_(?P<idx>\d+)_(?P<typ>rgb|depth|pcd)_(?P<sec>\d{2})\.(?P<ext>png|pcd)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    second_tag: str
    rgb_path: Path
    depth_path: Path
    pcd_path: Path


class FrameDataReader:
    """面向调用方的静态读取接口集合。"""

    @staticmethod
    def collect_records(input_dir: Path, max_frames: int = 0) -> list[FrameRecord]:
        records = _collect_frame_records(input_dir=Path(input_dir))
        if max_frames > 0:
            records = records[: int(max_frames)]
        return records

    @staticmethod
    def find_record(
        records: list[FrameRecord],
        frame_index: int,
        second_tag: str | None = None,
    ) -> FrameRecord:
        candidates = [r for r in records if r.frame_index == int(frame_index)]
        if second_tag is not None:
            candidates = [r for r in candidates if r.second_tag == str(second_tag)]
        if not candidates:
            raise KeyError(f"未找到 frame_index={frame_index}, second_tag={second_tag} 的记录")
        return candidates[0]

    @staticmethod
    def read_rgb(record: FrameRecord) -> np.ndarray:
        return _read_rgb(record.rgb_path)

    @staticmethod
    def read_depth(record: FrameRecord) -> np.ndarray:
        return _read_depth_gray(record.depth_path)

    @staticmethod
    def read_pcd(record: FrameRecord) -> tuple[np.ndarray, np.ndarray | None]:
        return _read_pcd_ascii(record.pcd_path)

    @staticmethod
    def read_frame_bundle(record: FrameRecord) -> dict[str, object]:
        rgb = FrameDataReader.read_rgb(record)
        depth = FrameDataReader.read_depth(record)
        xyz, rgb_pts = FrameDataReader.read_pcd(record)
        return {
            "frame_index": record.frame_index,
            "second_tag": record.second_tag,
            "rgb": rgb,
            "depth": depth,
            "pcd_xyz": xyz,
            "pcd_rgb": rgb_pts,
        }

    @staticmethod
    def read_frame_bundle_by_index(
        input_dir: Path,
        frame_index: int,
        second_tag: str | None = None,
    ) -> dict[str, object]:
        records = FrameDataReader.collect_records(input_dir=input_dir)
        record = FrameDataReader.find_record(records=records, frame_index=frame_index, second_tag=second_tag)
        return FrameDataReader.read_frame_bundle(record)

    @staticmethod
    def read_type_across_frames(records: list[FrameRecord], data_type: str) -> list[object]:
        typ = str(data_type).lower().strip()
        if typ not in {"rgb", "depth", "pcd"}:
            raise ValueError(f"data_type 仅支持 rgb/depth/pcd，当前为 {data_type}")

        outputs: list[object] = []
        for rec in records:
            if typ == "rgb":
                outputs.append(FrameDataReader.read_rgb(rec))
            elif typ == "depth":
                outputs.append(FrameDataReader.read_depth(rec))
            else:
                outputs.append(FrameDataReader.read_pcd(rec))
        return outputs

    @staticmethod
    def read_type_across_frames_by_dir(input_dir: Path, data_type: str, max_frames: int = 0) -> list[object]:
        records = FrameDataReader.collect_records(input_dir=input_dir, max_frames=max_frames)
        return FrameDataReader.read_type_across_frames(records=records, data_type=data_type)

    @staticmethod
    def create_unique_selected_run_dir(
        input_dir: Path,
        root_name: str = DEFAULT_SELECTED_ROOT_NAME,
        run_prefix: str = DEFAULT_SELECTED_RUN_PREFIX,
    ) -> Path:
        root_dir = Path(input_dir) / str(root_name)
        root_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{run_prefix}{time.strftime('%Y%m%d_%H%M%S')}"
        candidate = root_dir / base_name
        suffix = 1
        while candidate.exists():
            candidate = root_dir / f"{base_name}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    @staticmethod
    def save_frame_assets(record: FrameRecord, output_dir: Path) -> list[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for src in (record.rgb_path, record.depth_path, record.pcd_path):
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
        return copied


# region 主流程
def main(
    input_dir: Path = DEFAULT_INPUT_DIR,
    preview: bool = DEFAULT_PREVIEW,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    _log_usage_guide()
    logger.info(f"读取目录：{input_dir.resolve()}")
    logger.info("文件命名格式：frame_{frame_index}_{type}_{ss}")
    logger.info("type=rgb/depth/pcd, depth 为灰度 png, pcd 为 ASCII 点云")
    logger.info("预览操作：Space/N 下一帧，P 上一帧，S 另存当前帧，Q/ESC 退出")

    records = FrameDataReader.collect_records(input_dir=input_dir, max_frames=max_frames)

    if not records:
        logger.warning("未发现完整帧（需同时存在 rgb/depth/pcd 三个文件）")
        return

    logger.success(f"发现完整帧数量：{len(records)}")
    _print_quick_summary(records)

    if preview:
        selected_run_dir = FrameDataReader.create_unique_selected_run_dir(input_dir=input_dir)
        logger.success(f"本次预览另存目录：{selected_run_dir}")
        _run_preview(records=records, selected_run_dir=selected_run_dir)
    else:
        _read_without_preview(records=records)


# endregion


# region 读取与解析
def _collect_frame_records(input_dir: Path) -> list[FrameRecord]:
    grouped: dict[tuple[int, str], dict[str, Path]] = {}

    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        m = _FRAME_FILE_PATTERN.match(p.name)
        if m is None:
            continue

        idx = int(m.group("idx"))
        typ = m.group("typ").lower()
        sec = m.group("sec")
        key = (idx, sec)

        bucket = grouped.setdefault(key, {})
        bucket[typ] = p

    records: list[FrameRecord] = []
    for (idx, sec), bucket in grouped.items():
        if {"rgb", "depth", "pcd"} - set(bucket.keys()):
            continue
        records.append(
            FrameRecord(
                frame_index=idx,
                second_tag=sec,
                rgb_path=bucket["rgb"],
                depth_path=bucket["depth"],
                pcd_path=bucket["pcd"],
            )
        )

    records.sort(key=lambda r: (r.frame_index, r.second_tag))
    return records


def _read_rgb(path: Path) -> np.ndarray:
    _require_cv2_for_image_io()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取 RGB 失败：{path}")
    return img


def _read_depth_gray(path: Path) -> np.ndarray:
    _require_cv2_for_image_io()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"读取 depth 灰度图失败：{path}")
    return img


def _read_pcd_ascii(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise RuntimeError(f"空 PCD 文件：{path}")

    fields: list[str] = []
    data_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        upper = s.upper()
        if upper.startswith("FIELDS "):
            fields = s.split()[1:]
        elif upper == "DATA ASCII":
            data_idx = i + 1
            break

    if data_idx < 0:
        raise RuntimeError(f"PCD 非 ASCII 或缺少 DATA ascii: {path}")

    if not fields:
        raise RuntimeError(f"PCD 缺少 FIELDS: {path}")

    numeric_rows: list[list[float]] = []
    for line in lines[data_idx:]:
        s = line.strip()
        if not s:
            continue
        numeric_rows.append([float(x) for x in s.split()])

    if not numeric_rows:
        return np.empty((0, 3), dtype=np.float32), None

    data = np.asarray(numeric_rows, dtype=np.float32)
    field_to_idx = {name.lower(): i for i, name in enumerate(fields)}

    if not {"x", "y", "z"}.issubset(field_to_idx):
        raise RuntimeError(f"PCD 缺少 x/y/z 字段：{path}")

    xyz = data[:, [field_to_idx["x"], field_to_idx["y"], field_to_idx["z"]]].astype(np.float32, copy=False)

    rgb: np.ndarray | None = None
    if {"r", "g", "b"}.issubset(field_to_idx):
        rgb = data[:, [field_to_idx["r"], field_to_idx["g"], field_to_idx["b"]]]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)

    return xyz, rgb


def _require_cv2_for_image_io() -> None:
    if cv2 is None:
        raise RuntimeError("当前环境缺少 opencv-python，无法读取 png 图像。")


# endregion


# region 无预览读取
def _read_without_preview(records: list[FrameRecord]) -> None:
    logger.info("预览关闭：执行顺序读取与统计")
    for i, rec in enumerate(records, start=1):
        bundle = FrameDataReader.read_frame_bundle(rec)
        rgb = bundle["rgb"]
        depth = bundle["depth"]
        xyz = bundle["pcd_xyz"]
        rgb_pts = bundle["pcd_rgb"]
        logger.info(
            f"[{i}/{len(records)}] frame={rec.frame_index}, ss={rec.second_tag}, "
            f"rgb={rgb.shape[1]}x{rgb.shape[0]}, depth={depth.shape[1]}x{depth.shape[0]}, "
            f"pcd_points={len(xyz)}, pcd_color={'yes' if rgb_pts is not None else 'no'}"
        )


# endregion


# region 预览读取
def _run_preview(records: list[FrameRecord], selected_run_dir: Path) -> None:
    _require_cv2_for_image_io()
    logger.info("预览已开启：Space/N 下一帧，P 上一帧，S 另存当前帧，Q/ESC 退出")
    logger.info(f"另存目标目录：{selected_run_dir}")

    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    idx = 0
    saved_keys: set[tuple[int, str]] = set()
    try:
        while True:
            rec = records[idx]
            rgb = FrameDataReader.read_rgb(rec)
            depth = FrameDataReader.read_depth(rec)

            panel = _compose_preview_panel(
                rgb=rgb,
                depth_gray=depth,
                frame_index=rec.frame_index,
                second_tag=rec.second_tag,
                cursor_text=f"{idx + 1}/{len(records)}",
            )

            cv2.imshow(DEFAULT_WINDOW_NAME, panel)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                logger.warning("收到退出指令，结束预览")
                break
            if key in (32, ord("n"), ord("N")):
                idx = min(len(records) - 1, idx + 1)
                continue
            if key in (ord("p"), ord("P")):
                idx = max(0, idx - 1)
                continue
            if key in (ord("s"), ord("S")):
                rec_key = (rec.frame_index, rec.second_tag)
                if rec_key in saved_keys:
                    logger.warning(f"当前帧已另存过：frame={rec.frame_index}, ss={rec.second_tag}")
                    continue
                copied_files = FrameDataReader.save_frame_assets(record=rec, output_dir=selected_run_dir)
                saved_keys.add(rec_key)
                logger.success(
                    f"另存完成：frame={rec.frame_index}, ss={rec.second_tag}, "
                    f"文件数 {len(copied_files)}，目录 {selected_run_dir}"
                )
                continue
    finally:
        _safe_destroy_window(DEFAULT_WINDOW_NAME)


def _compose_preview_panel(
    rgb: np.ndarray,
    depth_gray: np.ndarray,
    frame_index: int,
    second_tag: str,
    cursor_text: str,
) -> np.ndarray:
    depth_vis = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

    h = max(rgb.shape[0], depth_vis.shape[0])
    w = max(rgb.shape[1], depth_vis.shape[1])
    rgb_r = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    canvas = np.hstack([rgb_r, depth_r])
    cv2.putText(canvas, "RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Depth(gray)", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    info = f"frame={frame_index}, ss={second_tag}, {cursor_text}"
    cv2.putText(canvas, info, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Keys: [S] Save current frame | [Space/N] Next | [P] Prev | [Q/ESC] Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _safe_destroy_window(window_name: str) -> None:
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except Exception:
        return


# endregion


# region 输出摘要
def _print_quick_summary(records: list[FrameRecord]) -> None:
    first = records[0]
    last = records[-1]
    logger.info(
        f"帧范围：{first.frame_index} -> {last.frame_index}, "
        f"首帧秒标签 {first.second_tag}, 末帧秒标签 {last.second_tag}"
    )

    try:
        xyz0, rgb0 = _read_pcd_ascii(first.pcd_path)
        logger.info(
            f"首帧 pcd 结构：points={len(xyz0)}, "
            f"xyz_dtype={xyz0.dtype}, rgb={'uint8' if rgb0 is not None else 'none'}"
        )
    except Exception as exc:
        logger.warning(f"首帧 pcd 解析失败：{exc}")


# endregion


# region CLI（仅用于覆盖默认参数）
def _parse_cli(argv: list[str] | None = None) -> tuple[Path, bool, int]:
    parser = argparse.ArgumentParser(
        description=(
            "读取 Orbbec 异步采集结果（frame_{index}_{type}_{ss}），"
            "支持顺序读取和可开关预览；预览时按 S 可另存当前帧到输入目录/_selected_frames/本次运行目录。"
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="输入目录（默认 yymmdd）")
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PREVIEW,
        help="是否开启预览窗口",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最多读取帧数，0 表示全部")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning(f"检测到未识别参数，已忽略：{unknown}")
    return Path(args.input_dir), bool(args.preview), int(args.max_frames)


def _log_usage_guide() -> None:
    for line in USAGE_GUIDE.splitlines():
        logger.info(line)


# endregion


if __name__ == "__main__":
    try:
        in_dir_arg, preview_arg, max_frames_arg = _parse_cli(sys.argv[1:])
        main(input_dir=in_dir_arg, preview=preview_arg, max_frames=max_frames_arg)
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
