from __future__ import annotations

import pytest

from camera_pipeline.camera_stream.runtime import JsonValue, _read_zmq_distortion


def test_read_zmq_distortion_converts_sdk_order_to_opencv_order() -> None:
    distortion = _read_zmq_distortion(
        {
            "dist": [
                -1.1831429,
                0.8150869,
                -0.22013795,
                -1.1651548,
                0.7908634,
                -0.21106473,
                0.00002755,
                -0.00024799,
            ]
        }
    )

    assert distortion == (
        -1.1831429,
        0.8150869,
        0.00002755,
        -0.00024799,
        -0.22013795,
        -1.1651548,
        0.7908634,
        -0.21106473,
    )


@pytest.mark.parametrize(
    "dist",
    (
        [],
        [0.0] * 7,
        [0.0] * 9,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True],
    ),
)
def test_read_zmq_distortion_rejects_invalid_coefficients(
    dist: list[JsonValue],
) -> None:
    with pytest.raises(RuntimeError):
        _read_zmq_distortion({"dist": dist})
