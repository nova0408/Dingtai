# 该方案不可行，显存不足（测试平台为 8G）

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.pointcloud.test_orbbec_realtime_plane_segmentation_zero_shot import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_CAPTURE_FPS,
    DEFAULT_COMBINE_PROMPTS_FORWARD,
    DEFAULT_DETECT_MAX_SIDE,
    DEFAULT_DEVICE,
    DEFAULT_GD_MODEL_ID,
    DEFAULT_MASK_IOU_SUPPRESS,
    DEFAULT_MAX_TARGETS,
    DEFAULT_MIN_MASK_PIXELS,
    DEFAULT_MIN_TARGET_CONF,
    DEFAULT_PROMPT,
    DEFAULT_PROXY_URL,
    DEFAULT_SAM_MAX_BOXES,
    DEFAULT_SAM_MODEL_ID,
    DEFAULT_STRICT_TARGET_FILTER,
    DEFAULT_TARGET_KEYWORDS,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOPK_OBJECTS,
    DEFAULT_USE_SAM,
    ZeroShotObjectPartitionDetector,
    _capture_preview_with_color_once,
    _project_points_to_image,
)

from src.rgbd_camera import OrbbecSession, SessionOptions

DEFAULT_TEMPLATE_CANDIDATES = (
    Path(r"C:\Project Documents\鼎泰项目\GeoTransformer\template\tray_template.npy"),
    Path(r"C:\Project Documents\鼎泰项目\GeoTransformer\template\tray_template.ply"),
    Path(r"C:\Project Documents\鼎泰项目\GeoTransformer\template\tray_template.pcd"),
)


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        pts = np.load(path)
    elif suffix in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported template format: {path}")
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Invalid point array shape: {pts.shape}")
    return np.asarray(pts[:, :3], dtype=np.float32)


def _save_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, points.astype(np.float32))
        return
    if suffix in {".ply", ".pcd"}:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        ok = o3d.io.write_point_cloud(str(path), pcd)
        if not ok:
            raise RuntimeError(f"Failed to write point cloud: {path}")
        return
    raise ValueError(f"Unsupported template output format: {path}")


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=float(max(1e-4, voxel_size_m)))
    out = np.asarray(pcd.points, dtype=np.float32)
    return out if out.size > 0 else points


def _random_downsample_cap(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if int(max_points) <= 0 or points.shape[0] <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return np.asarray(points[ids], dtype=np.float32)


def _find_existing_template(user_path: Path | None) -> Path | None:
    if user_path is not None:
        p = Path(user_path)
        return p if p.exists() else None
    for p in DEFAULT_TEMPLATE_CANDIDATES:
        if p.exists():
            return p
    return None


def _select_template_output_path(user_path: Path | None, template_output: Path) -> Path:
    if user_path is not None:
        return Path(user_path)
    return Path(template_output)


def _select_tray_points(
    points_xyz: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    detector: ZeroShotObjectPartitionDetector,
    min_points: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    detections = detector.detect(color_bgr)
    if len(detections) == 0:
        return None, None

    det = sorted(detections, key=lambda x: float(x.confidence_2d), reverse=True)[0]
    mask = np.asarray(det.mask > 0, dtype=bool)

    uv, valid_proj = _project_points_to_image(
        xyz=points_xyz,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        w=img_w,
        h=img_h,
    )
    keep = np.zeros(points_xyz.shape[0], dtype=bool)
    valid_ids = np.where(valid_proj)[0]
    if valid_ids.size == 0:
        return None, None

    uu = uv[valid_ids, 0].astype(np.int32)
    vv = uv[valid_ids, 1].astype(np.int32)
    inside = mask[vv, uu]
    keep[valid_ids] = inside

    selected = points_xyz[keep]
    if selected.shape[0] < int(min_points):
        return None, mask

    return selected, mask


def _capture_tray_roi_once(
    session: OrbbecSession,
    point_filter,
    detector: ZeroShotObjectPartitionDetector,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    min_points: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    points, color_bgr = _capture_preview_with_color_once(session=session, point_filter=point_filter)
    if points is None or color_bgr is None:
        return None, None, None

    xyz = np.asarray(points[:, :3], dtype=np.float32)
    picked, mask = _select_tray_points(
        points_xyz=xyz,
        color_bgr=color_bgr,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_w=img_w,
        img_h=img_h,
        detector=detector,
        min_points=int(min_points),
    )
    if picked is None:
        return None, color_bgr, mask
    return picked, color_bgr, mask


def _build_template_from_camera(
    session: OrbbecSession,
    point_filter,
    detector: ZeroShotObjectPartitionDetector,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    min_points: int,
    voxel_size_m: float,
    target_frames: int,
    max_attempts: int,
    min_accepted_frames: int,
    sleep_ms: int,
    out_file: Path,
) -> np.ndarray:
    accepted: list[np.ndarray] = []
    attempts = 0

    while attempts < int(max_attempts) and len(accepted) < int(target_frames):
        attempts += 1
        picked, _, _ = _capture_tray_roi_once(
            session=session,
            point_filter=point_filter,
            detector=detector,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_w=img_w,
            img_h=img_h,
            min_points=min_points,
        )
        if picked is None:
            logger.info("Template build attempt {}: no valid ROI", attempts)
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)
            continue

        accepted.append(_voxel_downsample(picked, voxel_size_m))
        logger.info(
            "Template build attempt {}: accepted frame {}/{} (points={})",
            attempts,
            len(accepted),
            target_frames,
            accepted[-1].shape[0],
        )
        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    if len(accepted) < int(min_accepted_frames):
        raise RuntimeError(
            f"Auto template build failed: accepted={len(accepted)} < min_accepted_frames={min_accepted_frames}"
        )

    merged = np.vstack(accepted).astype(np.float32)
    template_points = _voxel_downsample(merged, voxel_size_m)
    _save_points(out_file, template_points)
    logger.info("Auto template saved: {} (points={})", out_file, template_points.shape[0])
    return template_points


def _save_overlay(color_bgr: np.ndarray, mask: np.ndarray | None, out_png: Path) -> None:
    if mask is None:
        cv2.imwrite(str(out_png), color_bgr)
        return
    out = color_bgr.copy()
    red = np.zeros_like(out)
    red[:, :, 2] = 255
    alpha = 0.35
    out[mask] = cv2.addWeighted(out[mask], 1.0 - alpha, red[mask], alpha, 0.0)
    cv2.imwrite(str(out_png), out)


def _compose_preview_frame(
    color_bgr: np.ndarray,
    mask: np.ndarray | None,
    stage_name: str,
    roi_ok: bool,
    roi_points: int,
    confirm_key: str,
    quit_key: str,
) -> np.ndarray:
    frame = color_bgr.copy()
    if mask is not None:
        red = np.zeros_like(frame)
        red[:, :, 2] = 255
        m = np.asarray(mask, dtype=bool)
        frame[m] = cv2.addWeighted(frame[m], 0.65, red[m], 0.35, 0.0)

    lines = [
        f"[{stage_name}] ROI {'OK' if roi_ok else 'NOT READY'}  points={roi_points}",
        f"Press '{confirm_key.upper()}' to confirm current frame",
        f"Press '{quit_key.upper()}' to quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 240, 20), 2, cv2.LINE_AA)
        y += 30
    return frame


def _wait_user_confirmed_roi_frame(
    session: OrbbecSession,
    point_filter,
    detector: ZeroShotObjectPartitionDetector,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    min_points: int,
    stage_name: str,
    window_name: str,
    confirm_key: str,
    quit_key: str,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    confirm_ch = str(confirm_key).strip().lower()[:1] or "s"
    quit_ch = str(quit_key).strip().lower()[:1] or "q"
    confirm_ord = ord(confirm_ch)
    quit_ord = ord(quit_ch)

    attempts = 0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while attempts < int(max_attempts):
            attempts += 1
            picked, color_bgr, mask = _capture_tray_roi_once(
                session=session,
                point_filter=point_filter,
                detector=detector,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
                min_points=min_points,
            )
            if color_bgr is None:
                key = cv2.waitKey(1) & 0xFF
                if key == quit_ord:
                    raise KeyboardInterrupt("User aborted by quit key.")
                continue

            roi_ok = picked is not None
            roi_points = int(picked.shape[0]) if roi_ok else 0
            preview = _compose_preview_frame(
                color_bgr=color_bgr,
                mask=mask,
                stage_name=stage_name,
                roi_ok=roi_ok,
                roi_points=roi_points,
                confirm_key=confirm_ch,
                quit_key=quit_ch,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == quit_ord:
                raise KeyboardInterrupt("User aborted by quit key.")
            if key == confirm_ord:
                if roi_ok:
                    logger.info("{} confirmed by user, points={}", stage_name, roi_points)
                    return picked, color_bgr, mask
                logger.warning("{}: ROI not ready, ignore confirm", stage_name)
    finally:
        cv2.destroyWindow(window_name)

    raise RuntimeError(f"{stage_name}: exceed max preview attempts ({max_attempts}) without user confirmation.")


def _run_geotransformer_demo(
    geot_python: str,
    geot_exp_dir: Path,
    src_file: Path,
    ref_file: Path,
    gt_file: Path,
    weights_file: Path,
    geot_device: str,
    cuda_alloc_conf: str,
) -> int:
    demo_py = geot_exp_dir / "demo.py"
    if not weights_file.exists():
        raise FileNotFoundError(f"GeoTransformer weights not found: {weights_file}")
    geot_repo_root = geot_exp_dir.parent.parent
    env = os.environ.copy()
    old_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(geot_repo_root) if len(old_py) == 0 else str(geot_repo_root) + os.pathsep + old_py
    if cuda_alloc_conf.strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(cuda_alloc_conf).strip()

    if demo_py.exists():
        cmd = [
            geot_python,
            str(demo_py),
            f"--src_file={src_file}",
            f"--ref_file={ref_file}",
            f"--gt_file={gt_file}",
            f"--weights={weights_file}",
            f"--device={geot_device}",
        ]
        logger.info("Run GeoTransformer demo.py: {}", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(geot_exp_dir), env=env, check=False)
        return int(proc.returncode)

    # Fallback for release packages (e.g. v1.0.0) where demo.py is absent.
    logger.warning("demo.py not found, use internal single-pair fallback.")
    fallback_code = textwrap.dedent(
        """
        import argparse
        import os
        import os.path as osp
        import sys
        import numpy as np
        import torch

        from geotransformer.utils.data import registration_collate_fn_stack_mode
        from geotransformer.utils.torch import to_cuda, release_cuda
        from geotransformer.utils.registration import compute_registration_error

        from config import make_cfg
        from model import create_model

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--src_file", required=True)
            parser.add_argument("--ref_file", required=True)
            parser.add_argument("--gt_file", required=True)
            parser.add_argument("--weights", required=True)
            parser.add_argument("--device", default="cuda")
            args = parser.parse_args()

            src_points = np.load(args.src_file).astype(np.float32)
            ref_points = np.load(args.ref_file).astype(np.float32)
            transform = np.load(args.gt_file).astype(np.float32)
            src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
            ref_feats = np.ones_like(ref_points[:, :1], dtype=np.float32)

            cfg = make_cfg()
            data_dict = {
                "ref_points": ref_points,
                "src_points": src_points,
                "ref_feats": ref_feats,
                "src_feats": src_feats,
                "transform": transform,
            }
            neighbor_limits = [38, 36, 36, 38]
            data_dict = registration_collate_fn_stack_mode(
                [data_dict],
                cfg.backbone.num_stages,
                cfg.backbone.init_voxel_size,
                cfg.backbone.init_radius,
                neighbor_limits,
            )

            req = str(args.device).strip().lower()
            use_cuda = req.startswith("cuda") and torch.cuda.is_available()
            device = req if use_cuda else "cpu"
            model = create_model(cfg).to(device)
            state_dict = torch.load(args.weights, map_location=device)
            model.load_state_dict(state_dict["model"])
            model.eval()

            if device == "cuda":
                data_dict = to_cuda(data_dict)

            with torch.no_grad():
                output_dict = model(data_dict)

            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)
            estimated_transform = output_dict["estimated_transform"]
            rre, rte = compute_registration_error(transform, estimated_transform)
            print(f"[fallback] RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")

        if __name__ == "__main__":
            main()
        """
    ).strip()

    cmd = [
        geot_python,
        "-c",
        fallback_code,
        f"--src_file={src_file}",
        f"--ref_file={ref_file}",
        f"--gt_file={gt_file}",
        f"--weights={weights_file}",
        f"--device={geot_device}",
    ]
    logger.info("Run GeoTransformer fallback: {} -c <inline_script> ...", geot_python)
    proc = subprocess.run(cmd, cwd=str(geot_exp_dir), env=env, check=False)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orbbec tray ROI + GeoTransformer demo for manual qualitative comparison"
    )
    parser.add_argument(
        "--template-file",
        type=Path,
        default=None,
        help="Existing template point cloud (.npy/.ply/.pcd). If omitted, try default candidates.",
    )
    parser.add_argument(
        "--auto-build-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-build template from camera if template file is missing.",
    )
    parser.add_argument(
        "--force-rebuild-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuilding template from camera even if a template already exists.",
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=DEFAULT_TEMPLATE_CANDIDATES[0],
        help="Output path for auto-built template (.npy/.ply/.pcd).",
    )
    parser.add_argument(
        "--template-build-frames", type=int, default=8, help="Target accepted ROI frames for template build."
    )
    parser.add_argument("--template-build-attempts", type=int, default=80, help="Max attempts for auto template build.")
    parser.add_argument(
        "--template-build-min-accepted", type=int, default=4, help="Min accepted ROI frames to allow template save."
    )
    parser.add_argument(
        "--template-build-sleep-ms", type=int, default=60, help="Sleep milliseconds between template build attempts."
    )
    parser.add_argument(
        "--interactive-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live preview window and require manual confirmation before capture stages.",
    )
    parser.add_argument(
        "--require-confirm-before-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require manual confirmation before starting auto template build.",
    )
    parser.add_argument(
        "--require-confirm-before-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require manual confirmation before final source ROI capture.",
    )
    parser.add_argument("--preview-window-name", type=str, default="Tray ROI Preview")
    parser.add_argument("--confirm-key", type=str, default="s", help="Confirm key in preview window.")
    parser.add_argument("--quit-key", type=str, default="q", help="Quit key in preview window.")
    parser.add_argument("--preview-max-attempts", type=int, default=1200, help="Max preview loops for manual confirm.")

    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(r"C:\Project Documents\鼎泰项目\GeoTransformer\runs\orbbec_compare"),
        help="Output root for captured src/ref/gt and overlay image.",
    )
    parser.add_argument(
        "--geot-exp-dir",
        type=Path,
        default=Path(
            r"C:\Project Documents\鼎泰项目\GeoTransformer\GeoTransformer\experiments\geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
        ),
        help="GeoTransformer experiment folder containing demo.py.",
    )
    parser.add_argument(
        "--geot-weights",
        type=Path,
        default=Path(r"C:\Project Documents\鼎泰项目\GeoTransformer\weights\geotransformer-3dmatch.pth.tar"),
        help="GeoTransformer pretrained weight path.",
    )
    parser.add_argument("--geot-python", type=str, default="python", help="Python executable for GeoTransformer env.")

    parser.add_argument("--gd-model-id", type=str, default=DEFAULT_GD_MODEL_ID)
    parser.add_argument("--sam-model-id", type=str, default=DEFAULT_SAM_MODEL_ID)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--proxy-url", type=str, default=DEFAULT_PROXY_URL)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--target-keywords", type=str, default=DEFAULT_TARGET_KEYWORDS)
    parser.add_argument(
        "--strict-target-filter", action=argparse.BooleanOptionalAction, default=DEFAULT_STRICT_TARGET_FILTER
    )
    parser.add_argument("--max-targets", type=int, default=DEFAULT_MAX_TARGETS)
    parser.add_argument("--use-sam", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_SAM)
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--min-target-conf", type=float, default=DEFAULT_MIN_TARGET_CONF)
    parser.add_argument("--topk-objects", type=int, default=DEFAULT_TOPK_OBJECTS)
    parser.add_argument("--sam-max-boxes", type=int, default=DEFAULT_SAM_MAX_BOXES)
    parser.add_argument("--min-mask-pixels", type=int, default=DEFAULT_MIN_MASK_PIXELS)
    parser.add_argument("--mask-iou-suppress", type=float, default=DEFAULT_MASK_IOU_SUPPRESS)
    parser.add_argument("--detect-max-side", type=int, default=DEFAULT_DETECT_MAX_SIDE)
    parser.add_argument(
        "--combine-prompts-forward", action=argparse.BooleanOptionalAction, default=DEFAULT_COMBINE_PROMPTS_FORWARD
    )

    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS)
    parser.add_argument("--max-attempts", type=int, default=30, help="Max attempts for final source ROI capture.")
    parser.add_argument("--min-tray-points", type=int, default=600, help="Min ROI point count after segmentation.")
    parser.add_argument("--voxel-size-m", type=float, default=0.01, help="Downsample voxel size in meters.")
    parser.add_argument(
        "--geot-max-src-points", type=int, default=8000, help="Max source points sent to GeoTransformer."
    )
    parser.add_argument(
        "--geot-max-ref-points", type=int, default=8000, help="Max template points sent to GeoTransformer."
    )
    parser.add_argument(
        "--release-gpu-before-geot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Release detector models and clear CUDA cache before GeoTransformer run.",
    )
    parser.add_argument("--geot-device", type=str, default="cpu", help="GeoTransformer device: cuda / cpu.")
    parser.add_argument(
        "--cuda-alloc-conf",
        type=str,
        default="expandable_segments:True",
        help="PYTORCH_CUDA_ALLOC_CONF for GeoTransformer subprocess.",
    )
    args = parser.parse_args()

    detector = ZeroShotObjectPartitionDetector(
        gd_model_id=args.gd_model_id,
        sam_model_id=args.sam_model_id,
        device=args.device,
        proxy_url=args.proxy_url,
        prompt=args.prompt,
        target_keywords=args.target_keywords,
        strict_target_filter=bool(args.strict_target_filter),
        max_targets=int(args.max_targets),
        use_sam=bool(args.use_sam),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        min_target_conf=float(args.min_target_conf),
        topk_objects=int(args.topk_objects),
        sam_max_boxes=int(args.sam_max_boxes),
        combine_prompts_forward=bool(args.combine_prompts_forward),
        min_mask_pixels=int(args.min_mask_pixels),
        mask_iou_suppress=float(args.mask_iou_suppress),
        detect_max_side=int(args.detect_max_side),
    )

    options = SessionOptions(timeout_ms=int(args.timeout_ms), preferred_capture_fps=max(1, int(args.capture_fps)))

    template_points: np.ndarray | None = None
    template_file: Path | None = None
    chosen_points: np.ndarray | None = None
    chosen_bgr: np.ndarray | None = None
    chosen_mask: np.ndarray | None = None

    with OrbbecSession(options=options) as session:
        cam = session.get_camera_param()
        ci = cam.rgb_intrinsic if session.has_color_sensor else cam.depth_intrinsic
        img_w = int(max(32, ci.width))
        img_h = int(max(32, ci.height))
        fx = float(ci.fx)
        fy = float(ci.fy)
        cx = float(ci.cx)
        cy = float(ci.cy)

        point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())

        existing_template = _find_existing_template(args.template_file)
        need_rebuild = bool(args.force_rebuild_template) or (existing_template is None)

        if need_rebuild:
            if not bool(args.auto_build_template):
                if args.template_file is not None and not Path(args.template_file).exists():
                    raise FileNotFoundError(f"--template-file not found: {args.template_file}")
                raise FileNotFoundError(
                    "Template not found and auto build disabled. Provide --template-file or enable --auto-build-template."
                )

            if bool(args.interactive_preview) and bool(args.require_confirm_before_template):
                _wait_user_confirmed_roi_frame(
                    session=session,
                    point_filter=point_filter,
                    detector=detector,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    img_w=img_w,
                    img_h=img_h,
                    min_points=int(args.min_tray_points),
                    stage_name="TemplateStartCheck",
                    window_name=f"{args.preview_window_name} - Template",
                    confirm_key=str(args.confirm_key),
                    quit_key=str(args.quit_key),
                    max_attempts=int(args.preview_max_attempts),
                )

            template_file = _select_template_output_path(args.template_file, args.template_output)
            logger.info("Start auto template build -> {}", template_file)
            template_points = _build_template_from_camera(
                session=session,
                point_filter=point_filter,
                detector=detector,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
                min_points=int(args.min_tray_points),
                voxel_size_m=float(args.voxel_size_m),
                target_frames=int(args.template_build_frames),
                max_attempts=int(args.template_build_attempts),
                min_accepted_frames=int(args.template_build_min_accepted),
                sleep_ms=int(args.template_build_sleep_ms),
                out_file=template_file,
            )
        else:
            template_file = existing_template
            template_points = _voxel_downsample(_load_points(template_file), float(args.voxel_size_m))

        logger.info("Template file: {}", template_file)
        logger.info("Template points: {}", template_points.shape[0])

        if bool(args.interactive_preview) and bool(args.require_confirm_before_source):
            picked, color_bgr, mask = _wait_user_confirmed_roi_frame(
                session=session,
                point_filter=point_filter,
                detector=detector,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
                min_points=int(args.min_tray_points),
                stage_name="SourceCapture",
                window_name=f"{args.preview_window_name} - Source",
                confirm_key=str(args.confirm_key),
                quit_key=str(args.quit_key),
                max_attempts=int(args.preview_max_attempts),
            )
            chosen_points = _voxel_downsample(picked, float(args.voxel_size_m))
            chosen_bgr = color_bgr
            chosen_mask = mask
        else:
            for i in range(1, int(args.max_attempts) + 1):
                picked, color_bgr, mask = _capture_tray_roi_once(
                    session=session,
                    point_filter=point_filter,
                    detector=detector,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    img_w=img_w,
                    img_h=img_h,
                    min_points=int(args.min_tray_points),
                )
                if picked is None:
                    logger.info("Source capture attempt {}: tray ROI not found / too few points", i)
                    continue

                chosen_points = _voxel_downsample(picked, float(args.voxel_size_m))
                chosen_bgr = color_bgr
                chosen_mask = mask
                logger.info("Source capture attempt {}: selected tray points={}", i, chosen_points.shape[0])
                break

    if chosen_points is None or chosen_bgr is None or template_points is None:
        raise RuntimeError("Failed to prepare source/template points.")

    run_dir = args.save_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    src_file = run_dir / "src.npy"
    ref_file = run_dir / "ref.npy"
    gt_file = run_dir / "gt.npy"
    overlay_file = run_dir / "seg_overlay.png"

    np.save(src_file, chosen_points.astype(np.float32))
    np.save(ref_file, template_points.astype(np.float32))
    np.save(gt_file, np.eye(4, dtype=np.float32))
    _save_overlay(chosen_bgr, chosen_mask, overlay_file)

    src_limited = _random_downsample_cap(chosen_points, int(args.geot_max_src_points), seed=7)
    ref_limited = _random_downsample_cap(template_points, int(args.geot_max_ref_points), seed=11)
    np.save(src_file, src_limited.astype(np.float32))
    np.save(ref_file, ref_limited.astype(np.float32))
    logger.info("GeoTransformer input points: src={} ref={}", src_limited.shape[0], ref_limited.shape[0])

    if bool(args.release_gpu_before_geot):
        detector = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Skip CUDA cache clear: {}", e)

    logger.info("Saved run files: {}", run_dir)

    code = _run_geotransformer_demo(
        geot_python=args.geot_python,
        geot_exp_dir=args.geot_exp_dir,
        src_file=src_file,
        ref_file=ref_file,
        gt_file=gt_file,
        weights_file=args.geot_weights,
        geot_device=str(args.geot_device),
        cuda_alloc_conf=str(args.cuda_alloc_conf),
    )
    if code != 0:
        raise RuntimeError(f"GeoTransformer demo exit code: {code}")


if __name__ == "__main__":
    main()
