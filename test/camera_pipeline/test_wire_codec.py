from __future__ import annotations

import sys
import threading
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPosePriorInfo,
)
from camera_pipeline.protocol import CameraFramePacket, CameraName
from camera_pipeline.service.protocol import (
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CameraStatusResponse,
    CharucoDetectionRequest,
)
from camera_pipeline.service.transport import CameraPipelineRpcClient, CameraPipelineRpcServer
from camera_pipeline.service.wire_codec import WireCodecError, decode_wire, encode_wire


def test_request_round_trip_preserves_nested_protocol_types() -> None:
    request = CameraPipelineServiceRequest(
        operation="detect_ball",
        camera_name=CameraName.LEFT_ARM,
        detect_ball=BallPoseDetectionRequest(
            request_id=7,
            camera_name=CameraName.LEFT_ARM,
            priors=(
                BallPosePriorInfo(
                    color_hex="#ffff00",
                    radius_mm=20.0,
                    model_center_mm=(1.0, 2.0, 3.0),
                ),
            ),
        ),
    )

    decoded = decode_wire(encode_wire(request), CameraPipelineServiceRequest)

    assert decoded == request
    assert isinstance(decoded.camera_name, CameraName)
    assert decoded.detect_ball is not None
    assert isinstance(decoded.detect_ball.camera_name, CameraName)


def test_charuco_request_round_trip_preserves_explicit_board_parameters() -> None:
    request = CameraPipelineServiceRequest(
        operation="detect_charuco",
        camera_name=CameraName.HEAD,
        detect_charuco=CharucoDetectionRequest(
            camera_name=CameraName.HEAD,
            dictionary_name="DICT_APRILTAG_16H5",
            squares_x=4,
            squares_y=4,
            square_length_mm=20.0,
            marker_length_mm=14.0,
            min_charuco_corners=6,
            max_frames=300,
            stable_timeout_s=10.0,
        ),
    )

    decoded = decode_wire(encode_wire(request), CameraPipelineServiceRequest)

    assert decoded == request
    assert isinstance(decoded.camera_name, CameraName)
    assert decoded.detect_charuco is not None
    assert isinstance(decoded.detect_charuco.camera_name, CameraName)


def test_frame_round_trip_preserves_numpy_dtype_and_values() -> None:
    frame = CameraFramePacket(
        frame_id=42,
        camera_name="test_camera",
        timestamp_ms=1234.5,
        color_bgr=np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        depth_mm=np.arange(12, dtype=np.uint16).reshape(3, 4),
        fx=600.0,
        fy=601.0,
        cx=2.0,
        cy=1.5,
        distortion=(0.1, -0.2, 0.001, -0.002, 0.03),
    )

    decoded = decode_wire(encode_wire(frame), CameraFramePacket)

    assert decoded.frame_id == frame.frame_id
    assert decoded.color_bgr.dtype == np.uint8
    assert decoded.depth_mm.dtype == np.uint16
    assert np.array_equal(decoded.color_bgr, frame.color_bgr)
    assert np.array_equal(decoded.depth_mm, frame.depth_mm)
    assert decoded.distortion == frame.distortion


def test_decoder_rejects_non_wire_payload() -> None:
    try:
        decode_wire(b"legacy-binary-payload", CameraPipelineServiceRequest)
    except WireCodecError as exc:
        assert "magic" in str(exc)
    else:
        raise AssertionError("non-wire payload was not rejected")


def test_transport_round_trip_uses_explicit_wire_protocol() -> None:
    address = f"inproc://wire-codec-{uuid.uuid4().hex}"
    server = CameraPipelineRpcServer(address)
    client = CameraPipelineRpcClient(address)
    errors: list[Exception] = []

    def _serve_once() -> None:
        try:
            request = server.receive()
            server.send(
                CameraPipelineServiceResponse(
                    operation=request.operation,
                    camera_status=CameraStatusResponse(
                        camera_name="test_camera",
                        camera_id="TEST",
                        camera_model="test",
                        width=640,
                        height=480,
                        color_enabled=True,
                        depth_enabled=True,
                        online=True,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=_serve_once, daemon=True)
    thread.start()
    try:
        response = client.call(
            CameraPipelineServiceRequest(
                operation="camera_status",
                camera_name=CameraName.LEFT_ARM,
                timeout_s=1.0,
            )
        )
        assert response.camera_status is not None
        assert response.camera_status.camera_name == "test_camera"
    finally:
        client.close()
        server.close()
        thread.join(timeout=1.0)
    assert not errors


def main() -> None:
    test_request_round_trip_preserves_nested_protocol_types()
    test_charuco_request_round_trip_preserves_explicit_board_parameters()
    test_frame_round_trip_preserves_numpy_dtype_and_values()
    test_decoder_rejects_non_wire_payload()
    test_transport_round_trip_uses_explicit_wire_protocol()


if __name__ == "__main__":
    main()
