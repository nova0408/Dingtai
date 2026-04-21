from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imu.imu760 import IMU760, IMU760EulerAngles, decode_output_payload
from src.imu.ttl import TTLSerialConfig, TTLSerialTransport


def _find_euler(payload: bytes) -> tuple[float, float, float] | None:
    decoded = decode_output_payload(payload)
    for item in decoded:
        if isinstance(item, IMU760EulerAngles):
            return float(item.pitch), float(item.roll), float(item.yaw)
    return None


def run_realtime_monitor(
    port: str = "COM5",
    baudrate: int = 460800,
    timeout_s: float = 0.2,
    history_size: int = 600,
) -> None:
    config = TTLSerialConfig(port=port, baudrate=baudrate, timeout_s=timeout_s, write_timeout_s=timeout_s)
    transport = TTLSerialTransport(config=config)
    imu = IMU760(transport=transport, debug_enabled=False)

    times: deque[float] = deque(maxlen=history_size)
    pitch_hist: deque[float] = deque(maxlen=history_size)
    roll_hist: deque[float] = deque(maxlen=history_size)
    yaw_hist: deque[float] = deque(maxlen=history_size)
    frame_timestamps: deque[float] = deque(maxlen=500)

    start_time = time.perf_counter()
    frame_count = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    (line_pitch,) = ax.plot([], [], label="Pitch (deg)")
    (line_roll,) = ax.plot([], [], label="Roll (deg)")
    (line_yaw,) = ax.plot([], [], label="Yaw (deg)")

    ax.set_title("IMU760 Realtime Attitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    imu.open()
    imu.clear_input_buffer()
    logger.info(f"开始读取 {port} @ {baudrate}，关闭图窗或 Ctrl+C 结束。")

    try:
        while plt.fignum_exists(fig.number):
            frame = imu.read_output_frame(timeout_s=max(timeout_s, 0.05))
            now = time.perf_counter()
            elapsed = now - start_time
            frame_count += 1
            frame_timestamps.append(now)

            euler = _find_euler(frame.payload)
            if euler is None:
                plt.pause(0.001)
                continue

            pitch, roll, yaw = euler
            times.append(elapsed)
            pitch_hist.append(pitch)
            roll_hist.append(roll)
            yaw_hist.append(yaw)

            line_pitch.set_data(times, pitch_hist)
            line_roll.set_data(times, roll_hist)
            line_yaw.set_data(times, yaw_hist)

            if len(times) >= 2:
                ax.set_xlim(times[0], times[-1] + 1e-6)
            else:
                ax.set_xlim(0.0, max(1.0, elapsed))

            all_values = list(pitch_hist) + list(roll_hist) + list(yaw_hist)
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                if math.isclose(y_min, y_max, rel_tol=1e-9, abs_tol=1e-9):
                    y_min -= 1.0
                    y_max += 1.0
                margin = max((y_max - y_min) * 0.1, 1.0)
                ax.set_ylim(y_min - margin, y_max + margin)

            total_fps = frame_count / max(elapsed, 1e-6)
            if len(frame_timestamps) >= 2:
                win_dt = frame_timestamps[-1] - frame_timestamps[0]
                realtime_fps = (len(frame_timestamps) - 1) / max(win_dt, 1e-6)
            else:
                realtime_fps = 0.0

            fig.suptitle(
                f"Port={port}  Baud={baudrate}  FPS(avg)={total_fps:.1f}  FPS(rt)={realtime_fps:.1f}\n"
                f"Pitch={pitch:.3f}  Roll={roll:.3f}  Yaw={yaw:.3f}",
                fontsize=11,
            )
            plt.pause(0.001)

    except KeyboardInterrupt:
        logger.info("手动终止。")
    finally:
        imu.close()
        plt.ioff()
        plt.show()

    total_elapsed = time.perf_counter() - start_time
    logger.info(
        f"总帧数={frame_count}, 总时长={total_elapsed:.3f}s, 平均速度={frame_count / max(total_elapsed, 1e-6):.2f} FPS"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="IMU760 读取速度与实时姿态绘图")
    parser.add_argument("--port", type=str, default="COM5", help="串口号，默认 COM5")
    parser.add_argument("--baudrate", type=int, default=460800, help="波特率，默认 460800")
    parser.add_argument("--timeout", type=float, default=0.2, help="串口读超时 (秒)")
    parser.add_argument("--history-size", type=int, default=600, help="图上保留的历史点数量")
    args = parser.parse_args()

    try:
        run_realtime_monitor(
            port=args.port,
            baudrate=args.baudrate,
            timeout_s=args.timeout,
            history_size=args.history_size,
        )
        return 0
    except Exception as exc:
        logger.error(f"运行失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
