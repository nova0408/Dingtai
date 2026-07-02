from __future__ import annotations

"""
采集 AprilTag board 与 opening pose 的相对位姿先验。

工作流程：
1. 复用 `test/wuji/apriltag_detect.py` 的检测链路，稳定识别先验 tag 3、4、5。
2. 使用多 tag 联合 PnP，将 3/4/5 的观测角点解算到同一个 board 坐标系。
3. 调用 opening detection 得到开口 pose。
4. 记录 `board_T_opening`，并将最终图、JSON、逐帧明细落盘到 `.archive`。

说明：
该脚本假设三个 tag 在同一 rigid board 上，且通过外部布局文件给出每个 tag 在 board 坐标系下的中心位姿。
"""

import argparse
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
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

import apriltag_detect as apriltag_eval  # noqa: E402
from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from camera_pipeline.opening_detection.protocol import OpeningDetectionPipelineRequest  # noqa: E402

DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_apriltag_relative_pose"
DEFAULT_LAYOUT_JSON = PROJECT_ROOT / "test" / "wuji" / "apriltag_board_layout.json"
DEFAULT_TARGET_TAG_IDS = (3, 4, 5)
DEFAULT_STABLE_WINDOW_S = 1.0
DEFAULT_STABLE_MIN_SUPPORT = 3
DEFAULT_MAX_FRAMES = 200
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
    board_pose_camera_frame: np.ndarray | None
    opening_pose_camera_frame: np.ndarray | None
    board_T_opening: np.ndarray | None
    camera_intrinsics: np.ndarray | None
    detected_tag_ids: list[int]
    layout_path: Path
    overlay_image_path: Path | None
    error: str | None


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    layout_json: Path = DEFAULT_LAYOUT_JSON,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> int:
    apriltag_eval._validate_runtime_requirements()
    session_dir = apriltag_eval._create_session_dir(Path(output_root))
    layout = _load_layout(Path(layout_json))
    logger.info("开始采集 AprilTag relative pose，布局文件：{}", layout_json)

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
            layout=layout,
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
    layout: dict[int, TagLayoutEntry],
    session_dir: Path,
    max_frames: int,
) -> CollectResult:
    capture_rows: list[apriltag_eval.CaptureRow] = []
    temporal_fusion_history: list[tuple[float, list[apriltag_eval.DetectionResult]]] = []
    latest_preview: np.ndarray | None = None
    last_opening_response: Any | None = None
    frame_index = 0
    final_error: str | None = None
    board_pose_camera_frame: np.ndarray | None = None
    opening_pose_camera_frame: np.ndarray | None = None
    board_T_opening: np.ndarray | None = None
    detected_tag_ids: list[int] = []
    cv2.namedWindow(apriltag_eval.DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for frame in client.subscribe_camera_frames(camera_name):
            color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
            if color_bgr.size == 0:
                continue
            frame_index += 1
            undistorted_bgr = cv2.undistort(color_bgr, calibration.camera_matrix, calibration.dist_coeffs)
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
            temporal_fusion_history.append((time.monotonic(), list(frame_results.get("Fusion", apriltag_eval.VariantDetections([], [])).results)))
            _prune_history(temporal_fusion_history, DEFAULT_STABLE_WINDOW_S)
            frame_results["TemporalFusion"] = apriltag_eval._fuse_temporal_detections(
                fusion_history=list(temporal_fusion_history),
                window_s=DEFAULT_STABLE_WINDOW_S,
                min_support=DEFAULT_STABLE_MIN_SUPPORT,
            )
            preview = apriltag_eval._compose_preview(
                variant_frames=variant_frames,
                frame_results=frame_results,
                frame_index=frame_index,
                elapsed_ms=elapsed_ms,
                session_dir=session_dir,
            )
            cv2.imshow(apriltag_eval.DEFAULT_WINDOW_NAME, preview)
            cv2.waitKey(1)
            latest_preview = preview
            _append_capture_rows(
                capture_rows=capture_rows,
                frame_index=frame_index,
                frame_results=frame_results,
            )
            temporal_results = frame_results["TemporalFusion"].results
            if apriltag_eval._has_stable_target_tags(temporal_results, DEFAULT_TARGET_TAG_IDS):
                detected_tag_ids = [int(item.tag_id) for item in temporal_results]
                board_pose_camera_frame = _estimate_board_pose(
                    detections=temporal_results,
                    calibration=calibration,
                    layout=layout,
                    tag_size_mm=DEFAULT_TAG_SIZE_MM,
                )
                last_opening_response = _request_opening_pose(client, camera_name, frame.frame_id)
                opening_pose_camera_frame = _opening_pose_to_transform(last_opening_response)
                if board_pose_camera_frame is not None and opening_pose_camera_frame is not None:
                    board_T_opening = np.linalg.inv(board_pose_camera_frame) @ opening_pose_camera_frame
                break
            if int(max_frames) > 0 and frame_index >= int(max_frames):
                final_error = f"达到最大帧数 {int(max_frames)} 仍未稳定识别到目标 tag"
                break
    finally:
        cv2.destroyAllWindows()

    if latest_preview is not None:
        cv2.imwrite(str(session_dir / "final_preview.png"), latest_preview)
    if latest_preview is not None:
        cv2.imshow(apriltag_eval.DEFAULT_WINDOW_NAME, latest_preview)
        cv2.waitKey(DEFAULT_WAIT_AFTER_SUCCESS_MS)
        cv2.destroyAllWindows()

    if board_pose_camera_frame is None and final_error is None:
        final_error = "未获得稳定的 target tag 3/4/5 结果"

    return CollectResult(
        frame_index=int(frame_index),
        board_pose_camera_frame=board_pose_camera_frame,
        opening_pose_camera_frame=opening_pose_camera_frame,
        board_T_opening=board_T_opening,
        camera_intrinsics=calibration.camera_matrix,
        detected_tag_ids=detected_tag_ids,
        layout_path=DEFAULT_LAYOUT_JSON,
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


def _estimate_board_pose(
    detections: list[apriltag_eval.DetectionResult],
    calibration: apriltag_eval.CameraCalibration,
    layout: dict[int, TagLayoutEntry],
    tag_size_mm: float,
) -> np.ndarray | None:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for detection in detections:
        if detection.corners_px is None or detection.tag_id not in layout:
            continue
        entry = layout[int(detection.tag_id)]
        corner_object_points = _build_tag_corner_points(entry.translation_mm, entry.rotation_matrix, tag_size_mm)
        object_points.append(corner_object_points)
        image_points.append(np.asarray(detection.corners_px, dtype=np.float64).reshape(4, 2))
    if len(object_points) < 3:
        return None
    obj = np.concatenate(object_points, axis=0).astype(np.float64)
    img = np.concatenate(image_points, axis=0).astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rot_mat, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = tvec.reshape(3)
    return transform


def _build_tag_corner_points(
    translation_mm: np.ndarray,
    rotation_matrix: np.ndarray,
    tag_size_mm: float,
) -> np.ndarray:
    half = float(tag_size_mm) * 0.5
    local_corners = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    return (np.asarray(rotation_matrix, dtype=np.float64) @ local_corners.T).T + np.asarray(translation_mm, dtype=np.float64)


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
    if result.board_pose_camera_frame is None:
        return None
    overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(overlay, "apriltag relative pose", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"frame={result.frame_index}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"tags={result.detected_tag_ids}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if result.board_T_opening is not None:
        for idx, row in enumerate(np.asarray(result.board_T_opening, dtype=np.float64)):
            cv2.putText(overlay, " ".join(f"{value: .3f}" for value in row), (40, 260 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
    overlay_path = session_dir / "final_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    return overlay_path


def _serialize_result(result: CollectResult, overlay_path: Path | None) -> dict[str, Any]:
    return {
        "frame_index": result.frame_index,
        "detected_tag_ids": result.detected_tag_ids,
        "camera": {
            "intrinsics": None if result.camera_intrinsics is None else np.asarray(result.camera_intrinsics, dtype=np.float64).tolist(),
        },
        "pose": {
            "board_pose_camera_frame": None if result.board_pose_camera_frame is None else np.asarray(result.board_pose_camera_frame, dtype=np.float64).tolist(),
            "opening_pose_camera_frame": None if result.opening_pose_camera_frame is None else np.asarray(result.opening_pose_camera_frame, dtype=np.float64).tolist(),
            "board_T_opening": None if result.board_T_opening is None else np.asarray(result.board_T_opening, dtype=np.float64).tolist(),
        },
        "layout_path": str(result.layout_path),
        "overlay_image_path": None if overlay_path is None else str(overlay_path),
        "error": result.error,
    }


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 AprilTag 相对 opening pose 的先验变换")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            output_root=Path(args.output_root),
            layout_json=Path(args.layout_json),
            max_frames=int(args.max_frames),
        )
    )
