from __future__ import annotations

"""
采集 AprilTag board 与 opening pose 的相对位姿先验。

工作流程：
1. 复用 `test/wuji/apriltag_detect.py` 的检测链路，稳定识别先验 tag 0、1。
2. 直接使用检测结果中已经解出的 tag 位姿做逐帧缓存与平均。
3. 调用 opening detection 得到开口 pose。
4. 记录第一个 tag 与 opening pose 的直接相对关系，并将最终图、JSON、逐帧明细落盘到 `.archive`。

说明：
该脚本只依赖检测结果中的单 tag 位姿，不再使用联合 PnP。
"""

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_WUJI_ROOT = PROJECT_ROOT / "test" / "wuji"
if str(TEST_WUJI_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_WUJI_ROOT))

import apriltag_detect as apriltag_eval

from camera_pipeline.client import CameraPipelineClient
from camera_pipeline.opening_detection.protocol import OpeningDetectionPipelineRequest

DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_apriltag_relative_pose"
DEFAULT_TARGET_TAG_IDS = (0, 1)
DEFAULT_STABLE_WINDOW_S = 1.0
DEFAULT_STABLE_MIN_SUPPORT = 3
DEFAULT_MAX_FRAMES = 20
DEFAULT_TAG_SIZE_MM = 40.0
DEFAULT_WAIT_AFTER_SUCCESS_MS = 5000


@dataclass(frozen=True)
class TagLayoutEntry:
    """单个 tag 在 board 坐标系下的布局。"""

    tag_id: int
    translation_mm: np.ndarray
    rotation_matrix: np.ndarray


@dataclass(frozen=True)
class CollectResult:
    """一次采集结果。"""

    frame_index: int
    tag_poses_camera_frame: dict[int, np.ndarray]
    opening_pose_camera_frame: np.ndarray | None
    opening_T_tag0: np.ndarray | None
    relative_transform: np.ndarray | None
    relative_base_pose: np.ndarray | None
    opening_raw_result: dict[str, Any] | None
    last_frame_bgr: np.ndarray | None
    camera_intrinsics: np.ndarray | None
    detected_tag_ids: list[int]
    layout_path: Path
    overlay_image_path: Path | None
    error: str | None


@dataclass(frozen=True)
class FrameTask:
    """后台处理任务。"""

    frame_id: int
    color_bgr: np.ndarray


@dataclass(frozen=True)
class FrameProcessResult:
    """后台处理结果。"""

    frame_id: int
    preview: np.ndarray
    capture_rows: list[apriltag_eval.CaptureRow]
    detected_tag_ids: list[int]
    tag_pose_samples: dict[int, list[np.ndarray]]


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> int:
    apriltag_eval._validate_runtime_requirements()
    session_dir = apriltag_eval._create_session_dir(Path(output_root))
    logger.info("开始采集 AprilTag relative pose，目标 tag：{}", DEFAULT_TARGET_TAG_IDS)

    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=30_000)
    try:
        summary_response = client.get_camera_summary(timeout_s=30.0)
        status_response = client.get_camera_status(timeout_s=30.0)
        intrinsics_response = client.get_camera_intrinsics(timeout_s=30.0)
        calibration = apriltag_eval._read_camera_calibration(intrinsics_response)
        dictionary = apriltag_eval._get_apriltag_dictionary(apriltag_eval.DEFAULT_DICTIONARY_NAME)
        template_bank = apriltag_eval._build_template_bank(
            dictionary,
            {},
            allowed_tag_ids=DEFAULT_TARGET_TAG_IDS,
        )
        logger.info(
            "相机状态 camera={} model={} 分辨率={}x{}",
            status_response.camera_name,
            status_response.camera_model,
            status_response.width,
            status_response.height,
        )
        logger.info("相机摘要 source_meta={}", summary_response.source_meta)

        result = _collect_once(
            client=client,
            camera_name=str(camera_name),
            calibration=calibration,
            dictionary=dictionary,
            template_bank=template_bank,
            session_dir=session_dir,
            max_frames=int(max_frames),
        )
    finally:
        client.close()

    overlay_path = _save_overlay(session_dir, result)
    payload = _serialize_result(result, overlay_path)
    (session_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("结果已写入 {}", session_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.error is not None:
        raise RuntimeError(result.error)
    return 0


def _collect_once(
    client: CameraPipelineClient,
    camera_name: str,
    calibration: apriltag_eval.CameraCalibration,
    dictionary: Any,
    template_bank: apriltag_eval.TemplateBank,
    session_dir: Path,
    max_frames: int,
) -> CollectResult:
    capture_rows: list[apriltag_eval.CaptureRow] = []
    latest_preview: np.ndarray | None = None
    frame_index = 0
    last_frame_id = 0
    last_frame_bgr: np.ndarray | None = None
    detected_tag_ids: list[int] = []
    tag_pose_samples: dict[int, list[np.ndarray]] = {tag_id: [] for tag_id in DEFAULT_TARGET_TAG_IDS}
    opening_pose_samples: list[np.ndarray] = []
    opening_raw_result: dict[str, Any] | None = None

    frame_queue: Queue[FrameTask | None] = Queue(maxsize=1)
    result_queue: Queue[FrameProcessResult] = Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_frame_worker,
        args=(
            frame_queue,
            result_queue,
            stop_event,
            calibration,
            dictionary,
            template_bank,
            session_dir,
        ),
        daemon=True,
    )
    worker.start()
    cv2.namedWindow(apriltag_eval.DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for frame in client.subscribe_camera_frames(camera_name):
            color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
            if color_bgr.size == 0:
                continue
            frame_index += 1
            last_frame_id = int(frame.frame_id)
            last_frame_bgr = cv2.undistort(color_bgr, calibration.camera_matrix, calibration.dist_coeffs)
            cv2.imshow(apriltag_eval.DEFAULT_WINDOW_NAME, last_frame_bgr)
            cv2.waitKey(1)
            latest_preview = last_frame_bgr
            try:
                frame_queue.put_nowait(FrameTask(frame_id=int(frame.frame_id), color_bgr=color_bgr))
            except Exception:
                pass
            _drain_worker_results(
                result_queue=result_queue,
                capture_rows=capture_rows,
                tag_pose_samples=tag_pose_samples,
                opening_pose_samples=opening_pose_samples,
                detected_tag_ids=detected_tag_ids,
                latest_preview_ref=lambda value: value,
            )
            if frame_index >= int(max_frames):
                break
    finally:
        stop_event.set()
        try:
            frame_queue.put_nowait(None)
        except Exception:
            pass
        worker.join(timeout=10.0)
        cv2.destroyAllWindows()

    _drain_worker_results(
        result_queue=result_queue,
        capture_rows=capture_rows,
        tag_pose_samples=tag_pose_samples,
        opening_pose_samples=opening_pose_samples,
        detected_tag_ids=detected_tag_ids,
        latest_preview_ref=lambda value: value,
    )

    if last_frame_id > 0:
        opening_response = _request_opening_pose(client, camera_name, last_frame_id)
        opening_raw_result = _serialize_opening_response(opening_response)
        opening_pose_sample = _opening_pose_to_transform(opening_response)
        if opening_pose_sample is not None:
            opening_pose_samples.append(opening_pose_sample)

    averaged_tag_poses = {
        tag_id: _average_transform(samples) for tag_id, samples in tag_pose_samples.items() if samples
    }
    averaged_opening_pose = _average_transform(opening_pose_samples) if opening_pose_samples else None
    opening_T_tag0 = None
    base_tag_pose = averaged_tag_poses.get(DEFAULT_TARGET_TAG_IDS[0])
    if base_tag_pose is not None and averaged_opening_pose is not None:
        opening_T_tag0 = np.linalg.inv(base_tag_pose) @ averaged_opening_pose
    relative_transform = opening_T_tag0
    relative_base_pose = base_tag_pose

    final_error: str | None = None
    if len(averaged_tag_poses) < len(DEFAULT_TARGET_TAG_IDS):
        final_error = f"有效样本不足，当前仅获得 {len(averaged_tag_poses)} 个 tag 的平均位姿"
    if averaged_opening_pose is None:
        final_error = final_error or "未获得有效 opening pose 样本"

    return CollectResult(
        frame_index=int(frame_index),
        tag_poses_camera_frame=averaged_tag_poses,
        opening_pose_camera_frame=averaged_opening_pose,
        opening_T_tag0=opening_T_tag0,
        relative_transform=relative_transform,
        relative_base_pose=relative_base_pose,
        opening_raw_result=opening_raw_result,
        last_frame_bgr=last_frame_bgr,
        camera_intrinsics=calibration.camera_matrix,
        detected_tag_ids=detected_tag_ids,
        layout_path=Path(""),
        overlay_image_path=None,
        error=final_error,
    )


def _prune_history(
    temporal_fusion_history: list[tuple[float, list[apriltag_eval.DetectionResult]]],
    window_s: float,
) -> None:
    if not temporal_fusion_history:
        return
    latest_ts = temporal_fusion_history[-1][0]
    while temporal_fusion_history and latest_ts - temporal_fusion_history[0][0] > float(window_s):
        temporal_fusion_history.pop(0)


def _request_opening_pose(client: CameraPipelineClient, camera_name: str, frame_id: int) -> Any:
    response = client.request_opening_detection(
        OpeningDetectionPipelineRequest(
            request_id=int(frame_id),
            camera_name=str(camera_name),
            frame_id=int(frame_id),
            target_tray_index=0,
            enable_debug=True,
        )
    )
    if response.error is not None:
        raise RuntimeError(str(response.error))
    return response


def _opening_pose_to_transform(response: Any) -> np.ndarray | None:
    if response is None or response.selected_result is None or response.selected_result.pose is None:
        return None
    pose = response.selected_result.pose
    if pose.rotation is None or pose.grasp_point_mm is None:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(pose.rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(pose.grasp_point_mm, dtype=np.float64)
    return transform


def _collect_detection_tag_poses(detections: list[apriltag_eval.DetectionResult]) -> dict[int, np.ndarray]:
    tag_poses: dict[int, np.ndarray] = {}
    for detection in detections:
        pose = _detection_to_transform(detection)
        if pose is None:
            continue
        tag_poses[int(detection.tag_id)] = pose
    return tag_poses


def _frame_worker(
    frame_queue: Queue[FrameTask | None],
    result_queue: Queue[FrameProcessResult],
    stop_event: threading.Event,
    calibration: apriltag_eval.CameraCalibration,
    dictionary: Any,
    template_bank: apriltag_eval.TemplateBank,
    session_dir: Path,
) -> None:
    temporal_fusion_history: list[tuple[float, list[apriltag_eval.DetectionResult]]] = []
    while not stop_event.is_set():
        try:
            task = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        if task is None:
            break
        processed = _process_frame_task(
            task=task,
            calibration=calibration,
            dictionary=dictionary,
            template_bank=template_bank,
            session_dir=session_dir,
            temporal_fusion_history=temporal_fusion_history,
        )
        result_queue.put(processed)


def _process_frame_task(
    task: FrameTask,
    calibration: apriltag_eval.CameraCalibration,
    dictionary: Any,
    template_bank: apriltag_eval.TemplateBank,
    session_dir: Path,
    temporal_fusion_history: list[tuple[float, list[apriltag_eval.DetectionResult]]],
) -> FrameProcessResult:
    undistorted_bgr = cv2.undistort(task.color_bgr, calibration.camera_matrix, calibration.dist_coeffs)
    variant_frames = apriltag_eval._build_variant_frames(
        undistorted_bgr=undistorted_bgr,
        clip_limit=apriltag_eval.DEFAULT_CLAHE_CLIP_LIMIT,
        clahe_grid=apriltag_eval.DEFAULT_CLAHE_GRID,
    )
    started = cv2.getTickCount()
    frame_results = apriltag_eval._evaluate_frame(
        variant_frames=variant_frames,
        calibration=calibration,
        dictionary=dictionary,
        template_bank=template_bank,
        tag_specs={},
        tag_size_mm=DEFAULT_TAG_SIZE_MM,
    )
    elapsed_ms = (cv2.getTickCount() - started) * 1000.0 / cv2.getTickFrequency()
    temporal_fusion_history.append(
        (time.monotonic(), list(frame_results.get("Fusion", apriltag_eval.VariantDetections([], [])).results))
    )
    _prune_history(temporal_fusion_history, DEFAULT_STABLE_WINDOW_S)
    frame_results["TemporalFusion"] = apriltag_eval._fuse_temporal_detections(
        fusion_history=list(temporal_fusion_history),
        window_s=DEFAULT_STABLE_WINDOW_S,
        min_support=DEFAULT_STABLE_MIN_SUPPORT,
    )
    preview = apriltag_eval._compose_preview(
        variant_frames=variant_frames,
        frame_results=frame_results,
        frame_index=task.frame_id,
        elapsed_ms=elapsed_ms,
        session_dir=session_dir,
    )
    capture_rows: list[apriltag_eval.CaptureRow] = []
    _append_capture_rows(
        capture_rows=capture_rows,
        frame_index=task.frame_id,
        frame_results=frame_results,
    )
    temporal_results = frame_results["TemporalFusion"].results
    frame_tag_poses = _collect_detection_tag_poses(temporal_results)
    return FrameProcessResult(
        frame_id=task.frame_id,
        preview=preview,
        capture_rows=capture_rows,
        detected_tag_ids=sorted(frame_tag_poses.keys()),
        tag_pose_samples={tag_id: [pose] for tag_id, pose in frame_tag_poses.items()},
    )


def _drain_worker_results(
    result_queue: Queue[FrameProcessResult],
    capture_rows: list[apriltag_eval.CaptureRow],
    tag_pose_samples: dict[int, list[np.ndarray]],
    opening_pose_samples: list[np.ndarray],
    detected_tag_ids: list[int],
    latest_preview_ref,
) -> None:
    while True:
        try:
            item = result_queue.get_nowait()
        except Empty:
            break
        capture_rows.extend(item.capture_rows)
        for tag_id, samples in item.tag_pose_samples.items():
            tag_pose_samples.setdefault(int(tag_id), []).extend(samples)
        if item.detected_tag_ids:
            detected_tag_ids[:] = sorted(set(detected_tag_ids).union(item.detected_tag_ids))


def _detection_to_transform(detection: apriltag_eval.DetectionResult) -> np.ndarray | None:
    if detection.rvec is None or detection.tvec_mm is None:
        return None
    rot_mat, _ = cv2.Rodrigues(np.asarray(detection.rvec, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = np.asarray(detection.tvec_mm, dtype=np.float64).reshape(3)
    return transform


def _average_transform(transforms: list[np.ndarray]) -> np.ndarray | None:
    if not transforms:
        return None
    mats = [
        np.asarray(transform, dtype=np.float64) for transform in transforms if np.asarray(transform).shape == (4, 4)
    ]
    if not mats:
        return None
    translations = np.stack([mat[:3, 3] for mat in mats], axis=0)
    rotations = np.stack([mat[:3, :3] for mat in mats], axis=0)
    mean_translation = np.mean(translations, axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = mean_translation
    return transform


def _append_capture_rows(
    capture_rows: list[apriltag_eval.CaptureRow],
    frame_index: int,
    frame_results: dict[str, apriltag_eval.VariantDetections],
) -> None:
    temporal = frame_results.get("TemporalFusion", apriltag_eval.VariantDetections(results=[], rejected_corners=[]))
    for result in temporal.results:
        capture_rows.append(
            apriltag_eval.CaptureRow(
                pose_index=frame_index,
                frame_index=frame_index,
                timestamp_s=float(time.monotonic()),
                variant_name=result.variant_name,
                detection_index=result.detection_index,
                tag_id=result.tag_id,
                label=result.label,
                color_signature=result.color_signature,
                detected=result.detected,
                score=result.score,
                template_score=result.template_score,
                reprojection_error_px=result.reprojection_error_px,
                tx_mm=None if result.tvec_mm is None else float(result.tvec_mm[0]),
                ty_mm=None if result.tvec_mm is None else float(result.tvec_mm[1]),
                tz_mm=None if result.tvec_mm is None else float(result.tvec_mm[2]),
                roll_deg=None if result.rpy_deg is None else float(result.rpy_deg[0]),
                pitch_deg=None if result.rpy_deg is None else float(result.rpy_deg[1]),
                yaw_deg=None if result.rpy_deg is None else float(result.rpy_deg[2]),
            )
        )


def _load_layout(path: Path) -> dict[int, TagLayoutEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"布局文件不存在：{path}\n"
            "请创建 JSON，格式示例：\n"
            "{\n"
            '  "3": {"translation_mm": [0, 0, 0], "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]]},\n'
            '  "4": {"translation_mm": [40, 0, 0], "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]]},\n'
            '  "5": {"translation_mm": [0, 40, 0], "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]]}\n'
            "}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    layout: dict[int, TagLayoutEntry] = {}
    for key, value in payload.items():
        tag_id = int(key)
        translation_mm = np.asarray(value["translation_mm"], dtype=np.float64)
        rotation_matrix = np.asarray(value.get("rotation_matrix", np.eye(3)), dtype=np.float64)
        if translation_mm.shape != (3,) or rotation_matrix.shape != (3, 3):
            raise ValueError(f"tag {tag_id} 布局格式错误")
        layout[tag_id] = TagLayoutEntry(
            tag_id=tag_id,
            translation_mm=translation_mm,
            rotation_matrix=rotation_matrix,
        )
    return layout


def _save_overlay(session_dir: Path, result: CollectResult) -> Path | None:
    if not result.tag_poses_camera_frame:
        return None
    overlay = None if result.last_frame_bgr is None else np.asarray(result.last_frame_bgr, dtype=np.uint8).copy()
    if overlay is None:
        overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
    title_color = (0, 255, 255)
    cv2.putText(overlay, "apriltag relative pose", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"frame={result.frame_index}",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"tags={result.detected_tag_ids}",
        (40, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for tag_id in DEFAULT_TARGET_TAG_IDS:
        pose = result.tag_poses_camera_frame.get(int(tag_id))
        if pose is None:
            continue
        _draw_axis_on_image(overlay, pose, result.camera_intrinsics, (0, 220, 0), f"tag {tag_id}")
    if result.opening_pose_camera_frame is not None:
        _draw_axis_on_image(
            overlay, result.opening_pose_camera_frame, result.camera_intrinsics, (0, 180, 255), "opening"
        )
    if result.relative_base_pose is not None and result.relative_transform is not None:
        final_pose = np.asarray(result.relative_base_pose, dtype=np.float64) @ np.asarray(
            result.relative_transform, dtype=np.float64
        )
        _draw_axis_on_image(overlay, final_pose, result.camera_intrinsics, (255, 180, 0), "final")
    overlay_path = session_dir / "final_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    return overlay_path


def _draw_axis_on_image(
    canvas: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if camera_matrix is None:
        return
    pose = np.asarray(transform, dtype=np.float64)
    if pose.shape != (4, 4):
        return
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3].reshape(3, 1)
    cv2.drawFrameAxes(
        canvas, np.asarray(camera_matrix, dtype=np.float64), np.zeros((8,), dtype=np.float64), rvec, tvec, 40.0, 3
    )
    origin = tuple(int(v) for v in np.round(_project_point(canvas, camera_matrix, rvec, tvec)))
    cv2.putText(canvas, label, (origin[0] + 8, origin[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _project_point(
    canvas: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float]:
    point_3d = np.zeros((1, 3), dtype=np.float64)
    point_2d, _ = cv2.projectPoints(
        point_3d,
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        np.zeros((8,), dtype=np.float64),
    )
    xy = point_2d.reshape(2)
    return float(xy[0]), float(xy[1])


def _serialize_result(result: CollectResult, overlay_path: Path | None) -> dict[str, Any]:
    return {
        "frame_index": result.frame_index,
        "detected_tag_ids": result.detected_tag_ids,
        "camera": {
            "intrinsics": (
                None
                if result.camera_intrinsics is None
                else np.asarray(result.camera_intrinsics, dtype=np.float64).tolist()
            ),
        },
        "pose": {
            "tag_poses_camera_frame": {
                str(tag_id): np.asarray(pose, dtype=np.float64).tolist()
                for tag_id, pose in sorted(result.tag_poses_camera_frame.items())
            },
            "opening_pose_camera_frame": (
                None
                if result.opening_pose_camera_frame is None
                else np.asarray(result.opening_pose_camera_frame, dtype=np.float64).tolist()
            ),
            "opening_T_tag0": (
                None if result.opening_T_tag0 is None else np.asarray(result.opening_T_tag0, dtype=np.float64).tolist()
            ),
            "relative_transform": (
                None
                if result.relative_transform is None
                else np.asarray(result.relative_transform, dtype=np.float64).tolist()
            ),
            "relative_base_pose": (
                None
                if result.relative_base_pose is None
                else np.asarray(result.relative_base_pose, dtype=np.float64).tolist()
            ),
        },
        "opening_raw_result": result.opening_raw_result,
        "layout_path": str(result.layout_path),
        "overlay_image_path": None if overlay_path is None else str(overlay_path),
        "error": result.error,
    }


def _serialize_opening_response(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None
    selected = None
    if response.selected_result is not None:
        selected = {
            "tray_index": response.selected_result.tray_index,
            "tray_bbox_xywh": list(response.selected_result.tray_bbox_xywh),
            "tray_center_uv": list(response.selected_result.tray_center_uv),
            "opening_center_uv": (
                None
                if response.selected_result.opening_center_uv is None
                else list(response.selected_result.opening_center_uv)
            ),
            "opening_quad_uv": (
                None
                if response.selected_result.opening_quad_uv is None
                else [list(item) for item in response.selected_result.opening_quad_uv]
            ),
            "top_quad_uv": (
                None
                if response.selected_result.top_quad_uv is None
                else [list(item) for item in response.selected_result.top_quad_uv]
            ),
            "pose": (
                None
                if response.selected_result.pose is None
                else {
                    "grasp_point_mm": list(response.selected_result.pose.grasp_point_mm),
                    "pre_grasp_point_mm": list(response.selected_result.pose.pre_grasp_point_mm),
                    "rotation": (
                        None
                        if response.selected_result.pose.rotation is None
                        else [list(row) for row in response.selected_result.pose.rotation]
                    ),
                    "rpy_deg": (
                        None
                        if response.selected_result.pose.rpy_deg is None
                        else list(response.selected_result.pose.rpy_deg)
                    ),
                }
            ),
        }
    return {
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "tray_count": response.tray_count,
        "selected_tray_index": response.selected_tray_index,
        "elapsed_ms": response.elapsed_ms,
        "error": response.error,
        "selected_result": selected,
    }


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 AprilTag 相对 opening pose 的先验变换")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            output_root=Path(args.output_root),
            max_frames=int(args.max_frames),
        )
    )
