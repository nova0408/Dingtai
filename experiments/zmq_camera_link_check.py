#!/usr/bin/env python3
from __future__ import annotations

import argparse

import zmq


DEFAULT_HOST = "192.168.10.37"
CONTROL_PORT = 5570
CAMERAS: tuple[tuple[str, str, int], ...] = (
    ("HEAD", "head_camera", 5560),
    ("CHEST", "chest_camera", 5561),
    ("LEFT", "left_hand_camera", 5562),
    ("RIGHT", "right_hand_camera", 5563),
)

def _send_control_request(ctx: zmq.Context, host: str, camera_id: str, command: str) -> dict[str, object]:
    socket_obj = ctx.socket(zmq.REQ)
    socket_obj.setsockopt(zmq.RCVTIMEO, 3000)
    socket_obj.setsockopt(zmq.SNDTIMEO, 3000)
    socket_obj.connect(f"tcp://{host}:{CONTROL_PORT}")
    try:
        socket_obj.send_json({"cmd": command, "camera": camera_id, "params": {}})
        response = socket_obj.recv_json()
    finally:
        socket_obj.close(linger=0)
    if not isinstance(response, dict):
        raise RuntimeError(f"invalid response: {response!r}")
    return response


def _parse_intrinsics(payload: dict[str, object]) -> tuple[float, float, float, float, list[float]]:
    data = payload.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid intrinsics payload: {payload!r}")
    fx = float(data.get("fx", 0.0))
    fy = float(data.get("fy", 0.0))
    cx = float(data.get("cx", 0.0))
    cy = float(data.get("cy", 0.0))
    dist = [float(value) for value in data.get("dist", [])]
    return fx, fy, cx, cy, dist


def _parse_status(payload: dict[str, object]) -> tuple[bool, bool, bool]:
    data = payload.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid status payload: {payload!r}")
    return (
        bool(data.get("online", False)),
        bool(data.get("color_enabled", False)),
        bool(data.get("depth_enabled", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal ZMQ link checker for wuyou left/right cameras.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="ZMQ camera host")
    args = parser.parse_args()

    ctx = zmq.Context()
    exit_code = 0

    try:
        print(f"host={args.host}")
        for camera_id, camera_name, stream_port in CAMERAS:
            print(f"\n[{camera_id}] {camera_name} tcp://{args.host}:{stream_port}")
            try:
                status_payload = _send_control_request(ctx, args.host, camera_id, "get_status")
                online, color_enabled, depth_enabled = _parse_status(status_payload)
                print(
                    f"  status online={online} "
                    f"color_enabled={color_enabled} depth_enabled={depth_enabled}"
                )
            except Exception as exc:
                exit_code = 1
                print(f"  status_error={exc}")
                continue

            try:
                intrinsics_payload = _send_control_request(ctx, args.host, camera_id, "get_intrinsics")
                fx, fy, cx, cy, dist = _parse_intrinsics(intrinsics_payload)
                print(f"  intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
                print(f"  dist={dist}")
                if fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
                    exit_code = 1
                    print("  intrinsics_issue=zero_field_detected")
            except Exception as exc:
                exit_code = 1
                print(f"  intrinsics_error={exc}")

        return exit_code
    finally:
        ctx.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
