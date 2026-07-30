from __future__ import annotations

import struct
import time

import cv2
import numpy as np
import pytest

from camera_pipeline.camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
from camera_pipeline.camera_stream.runtime import _calculate_retry_interval_s
from camera_pipeline.protocol import CameraColorFramePacket

_FRAME_HEADER = struct.Struct("<4sBBBBIIIIIIIQI")


def _runtime_without_connections() -> CameraStreamRuntime:
    """构造仅用于纯解码测试、不会建立 ZMQ 连接的运行时。"""

    return CameraStreamRuntime.__new__(CameraStreamRuntime)


def test_decode_returns_color_packet_when_upstream_frame_has_no_depth() -> None:
    runtime = _runtime_without_connections()
    runtime._config = CameraStreamRuntimeConfig(camera_name="head_camera")  # noqa: SLF001
    runtime._cached_intrinsics = (900.0, 901.0, 320.0, 240.0, ())  # noqa: SLF001
    encoded, color_buffer = cv2.imencode(
        ".jpg",
        np.zeros((8, 12, 3), dtype=np.uint8),
    )
    assert encoded
    color_payload = color_buffer.tobytes()
    raw_message = _FRAME_HEADER.pack(
        b"ZCAM",
        1,
        4,
        0,
        0,
        12,
        8,
        len(color_payload),
        0,
        0,
        0,
        0,
        123_000,
        42,
    ) + color_payload

    color_frame, rgbd_frame = runtime._decode_frame(raw_message)  # noqa: SLF001

    assert color_frame.frame_id == 42
    assert color_frame.color_bgr.shape == (8, 12, 3)
    assert rgbd_frame is None


def test_decode_rejects_message_length_mismatch_before_image_decode() -> None:
    runtime = _runtime_without_connections()
    raw_message = _FRAME_HEADER.pack(
        b"ZCAM",
        1,
        4,
        0,
        1,
        1280,
        720,
        100,
        1280,
        720,
        200,
        1280 * 720 * 2,
        123_000,
        43,
    )

    with pytest.raises(RuntimeError, match=r"ZMQ camera frame size mismatch"):
        runtime._decode_frame(raw_message)  # noqa: SLF001


def test_control_maintenance_recovers_after_late_upstream_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_without_connections()
    runtime._config = CameraStreamRuntimeConfig()  # noqa: SLF001
    runtime._cached_intrinsics = None  # noqa: SLF001
    runtime._depth_stream_confirmed = False  # noqa: SLF001
    runtime._depth_reset_required = False  # noqa: SLF001
    runtime._control_retry_failures = 2  # noqa: SLF001
    runtime._next_control_retry_at = 0.0  # noqa: SLF001
    commands: list[str] = []

    def _send_control(
        _runtime: CameraStreamRuntime,
        command_name: str,
        _params: object = None,
    ) -> dict[str, object]:
        commands.append(command_name)
        return {}

    monkeypatch.setattr(CameraStreamRuntime, "_send_control_command", _send_control)
    monkeypatch.setattr(
        CameraStreamRuntime,
        "_get_intrinsics_from_control",
        lambda _runtime: (900.0, 901.0, 320.0, 240.0, ()),
    )

    assert runtime._maintain_upstream_control()  # noqa: SLF001
    assert commands == ["set_depth_enabled"]
    assert runtime._cached_intrinsics == (  # noqa: SLF001
        900.0,
        901.0,
        320.0,
        240.0,
        (),
    )
    assert runtime._control_retry_failures == 0  # noqa: SLF001


def test_continuous_color_only_frames_reenter_depth_recovery() -> None:
    runtime = _runtime_without_connections()
    runtime._config = CameraStreamRuntimeConfig(  # noqa: SLF001
        max_consecutive_color_only_frames=3
    )
    runtime._depth_stream_confirmed = True  # noqa: SLF001
    runtime._depth_reset_required = False  # noqa: SLF001
    runtime._consecutive_color_only_frames = 0  # noqa: SLF001
    runtime._next_control_retry_at = 100.0  # noqa: SLF001

    runtime._record_color_only_frame()  # noqa: SLF001
    runtime._record_color_only_frame()  # noqa: SLF001
    assert runtime._depth_stream_confirmed  # noqa: SLF001

    runtime._record_color_only_frame()  # noqa: SLF001
    assert not runtime._depth_stream_confirmed  # noqa: SLF001
    assert runtime._depth_reset_required  # noqa: SLF001
    assert runtime._next_control_retry_at == 0.0  # noqa: SLF001


def test_depth_recovery_resets_mismatched_upstream_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_without_connections()
    runtime._config = CameraStreamRuntimeConfig()  # noqa: SLF001
    runtime._cached_intrinsics = (900.0, 901.0, 320.0, 240.0, ())  # noqa: SLF001
    runtime._depth_stream_confirmed = False  # noqa: SLF001
    runtime._depth_reset_required = True  # noqa: SLF001
    runtime._control_retry_failures = 0  # noqa: SLF001
    runtime._next_control_retry_at = 0.0  # noqa: SLF001
    depth_states: list[bool] = []

    def _send_control(
        _runtime: CameraStreamRuntime,
        command_name: str,
        params: object = None,
    ) -> dict[str, object]:
        assert command_name == "set_depth_enabled"
        assert isinstance(params, dict)
        enabled = params.get("enable")
        assert isinstance(enabled, bool)
        depth_states.append(enabled)
        return {}

    monkeypatch.setattr(CameraStreamRuntime, "_send_control_command", _send_control)
    monkeypatch.setattr(
        CameraStreamRuntime,
        "_get_intrinsics_from_control",
        lambda _runtime: (610.0, 611.0, 630.0, 360.0, ()),
    )

    assert runtime._maintain_upstream_control()  # noqa: SLF001
    assert depth_states == [False, True]
    assert not runtime._depth_reset_required  # noqa: SLF001
    assert runtime._cached_intrinsics == (  # noqa: SLF001
        610.0,
        611.0,
        630.0,
        360.0,
        (),
    )


@pytest.mark.parametrize(
    ("fx", "fy"),
    [
        (0.0, 610.0),
        (610.0, 0.0),
        (-1.0, 610.0),
    ],
)
def test_control_intrinsics_reject_non_positive_focal_length(
    monkeypatch: pytest.MonkeyPatch,
    fx: float,
    fy: float,
) -> None:
    runtime = _runtime_without_connections()
    payload = {
        "data": {
            "fx": fx,
            "fy": fy,
            "cx": 630.0,
            "cy": 360.0,
            "dist": [0.0] * 8,
        }
    }
    monkeypatch.setattr(
        CameraStreamRuntime,
        "_send_control_command",
        lambda _runtime, _command_name, _params=None: payload,
    )

    with pytest.raises(RuntimeError, match="focal length"):
        runtime._get_intrinsics_from_control()  # noqa: SLF001


def test_control_intrinsics_reject_non_finite_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_without_connections()
    payload = {
        "data": {
            "fx": float("nan"),
            "fy": 610.0,
            "cx": 630.0,
            "cy": 360.0,
            "dist": [0.0] * 8,
        }
    }
    monkeypatch.setattr(
        CameraStreamRuntime,
        "_send_control_command",
        lambda _runtime, _command_name, _params=None: payload,
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        runtime._get_intrinsics_from_control()  # noqa: SLF001


def test_control_retry_uses_bounded_exponential_backoff() -> None:
    assert _calculate_retry_interval_s(1, 2.0, 30.0) == 2.0
    assert _calculate_retry_interval_s(2, 2.0, 30.0) == 4.0
    assert _calculate_retry_interval_s(10, 2.0, 30.0) == 30.0


def test_stale_frame_identity_triggers_stream_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_without_connections()
    runtime._config = CameraStreamRuntimeConfig(stale_frame_timeout_s=0.1)  # noqa: SLF001
    runtime._last_stream_frame_id = 3850  # noqa: SLF001
    runtime._last_stream_timestamp_ms = 3_196_881.786  # noqa: SLF001
    runtime._last_stream_progress_at = time.perf_counter() - 1.0  # noqa: SLF001
    recovery_reasons: list[str] = []

    monkeypatch.setattr(
        CameraStreamRuntime,
        "_recover_stream_runtime",
        lambda _runtime, reason: recovery_reasons.append(reason),
    )

    stale_frame = CameraColorFramePacket(
        frame_id=3850,
        camera_name="left_hand_camera",
        timestamp_ms=3_196_881.786,
        color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        distortion=(),
    )

    assert not runtime._accept_stream_progress(stale_frame)  # noqa: SLF001
    assert len(recovery_reasons) == 1
    assert "previous_frame_id=3850 frame_id=3850" in recovery_reasons[0]
