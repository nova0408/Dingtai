from __future__ import annotations

import argparse
import csv
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

from src.imu.imu760 import (
    IMU760,
    IMU760AlgorithmMode,
    IMU760EulerAngles,
    IMU760MagneticNormalized,
    decode_output_payload,
)
from src.imu.ttl import TTLSerialConfig, TTLSerialTransport

try:
    from serial import SerialException
except Exception:  # pragma: no cover - pyserial 缺失时兜底

    class SerialException(Exception):
        pass


def _extract_observation(payload: bytes) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    decoded = decode_output_payload(payload)
    euler: tuple[float, float, float] | None = None
    magnetic_norm: tuple[float, float, float] | None = None
    for item in decoded:
        if isinstance(item, IMU760EulerAngles):
            euler = (float(item.pitch), float(item.roll), float(item.yaw))
        elif isinstance(item, IMU760MagneticNormalized):
            magnetic_norm = (item.mx, item.my, item.mz)
    return euler, magnetic_norm


def _decode_algorithm_mode(data: bytes) -> IMU760AlgorithmMode | None:
    candidates = [value for value in data if value in (0x01, 0x02, 0x03)]
    if not candidates:
        return None
    return IMU760AlgorithmMode(candidates[-1])


def _ensure_ahrs_mode(imu: IMU760, timeout_s: float = 0.5) -> IMU760AlgorithmMode:
    """启动前强制并确认算法模式为 AHRS。"""
    imu.set_algorithm_mode(IMU760AlgorithmMode.AHRS, save_to_flash=False, timeout_s=timeout_s)
    response = imu.query(data_class=0x4D, query_data=bytes([0x02]), timeout_s=timeout_s)
    mode = _decode_algorithm_mode(response.data)
    if mode != IMU760AlgorithmMode.AHRS:
        raise RuntimeError(f"算法模式确认失败，期望 AHRS，实际={mode}")
    return mode


def _write_hotplug_csv(events: list[dict[str, float]], output_csv: str) -> None:
    if not events:
        return
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(events[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def _plot_hotplug_events(events: list[dict[str, float]]) -> None:
    if not events:
        logger.info("没有记录到插拔事件，跳过插拔统计图。")
        return

    x_center = [int(e["plug_index"]) for e in events]
    x_before = [x - 0.3 for x in x_center]
    x_after = [x + 0.3 for x in x_center]
    before_mx = [e["before_mx"] for e in events]
    before_my = [e["before_my"] for e in events]
    before_mz = [e["before_mz"] for e in events]
    before_mnorm = [e["before_m_norm"] for e in events]
    after_mx = [e["after_mx"] for e in events]
    after_my = [e["after_my"] for e in events]
    after_mz = [e["after_mz"] for e in events]
    after_mnorm = [e["after_m_norm"] for e in events]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(x_before, before_mx, marker="o", label="before_mx")
    ax2.plot(x_before, before_my, marker="o", label="before_my")
    ax2.plot(x_before, before_mz, marker="o", label="before_mz")
    ax2.plot(x_before, before_mnorm, marker="o", linewidth=2.0, label="before_|m|")
    ax2.plot(x_after, after_mx, marker="x", linestyle="--", label="after_mx")
    ax2.plot(x_after, after_my, marker="x", linestyle="--", label="after_my")
    ax2.plot(x_after, after_mz, marker="x", linestyle="--", label="after_mz")
    ax2.plot(x_after, after_mnorm, marker="x", linestyle="--", linewidth=2.0, label="after_|m|")
    for idx, xc in enumerate(x_center):
        ax2.plot(
            [x_before[idx], x_after[idx]], [before_mnorm[idx], after_mnorm[idx]], color="gray", alpha=0.5, linewidth=1.0
        )

    ax2.set_title("Hot-plug Interval Values (n-0.3: before, n+0.3: after)")
    ax2.set_xlabel("Plug Count")
    ax2.set_ylabel("Normalized Magnetic Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_xticks(x_center)
    fig2.tight_layout()


def _mean_sample(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        raise ValueError("samples 不能为空")
    keys = ("pitch", "roll", "yaw", "mx", "my", "mz", "m_norm")
    n = float(len(samples))
    return {k: sum(s[k] for s in samples) / n for k in keys}


def run_realtime_monitor(
    port: str = "COM5",
    baudrate: int = 460800,
    timeout_s: float = 0.2,
    history_size: int = 600,
    reconnect_interval_s: float = 1.0,
    mag_norm_abs_tol: float = 0.15,
    mag_jump_tol: float = 0.08,
    hotplug_csv: str = "test/imu/imu_hotplug_changes.csv",
    plug_stable_frames: int = 10,
    plug_settle_frames: int = 5,
) -> None:
    times: deque[float] = deque(maxlen=history_size)
    pitch_hist: deque[float] = deque(maxlen=history_size)
    roll_hist: deque[float] = deque(maxlen=history_size)
    yaw_hist: deque[float] = deque(maxlen=history_size)
    magx_hist: deque[float] = deque(maxlen=history_size)
    magy_hist: deque[float] = deque(maxlen=history_size)
    magz_hist: deque[float] = deque(maxlen=history_size)
    mag_norm_hist: deque[float] = deque(maxlen=history_size)
    frame_timestamps: deque[float] = deque(maxlen=500)

    start_time = time.perf_counter()
    frame_count = 0

    plt.ion()
    fig, (ax_att, ax_mag) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    (line_pitch,) = ax_att.plot([], [], label="Pitch (deg)")
    (line_roll,) = ax_att.plot([], [], label="Roll (deg)")
    (line_yaw,) = ax_att.plot([], [], label="Yaw (deg)")

    (line_magx,) = ax_mag.plot([], [], label="mx (normalized)")
    (line_magy,) = ax_mag.plot([], [], label="my (normalized)")
    (line_magz,) = ax_mag.plot([], [], label="mz (normalized)")
    (line_mag_norm,) = ax_mag.plot([], [], label="|m|", linewidth=2.0)

    ax_att.set_ylabel("Angle (deg)")
    ax_mag.set_ylabel("Normalized Magnetic")
    ax_mag.set_xlabel("Time (s)")
    ax_att.grid(True, alpha=0.3)
    ax_mag.grid(True, alpha=0.3)
    ax_att.legend(loc="upper left")
    ax_mag.legend(loc="upper left")

    imu: IMU760 | None = None
    connected = False
    last_reconnect_try = 0.0
    algo_mode_text = "UNKNOWN"
    mag_quality_status = "UNKNOWN"
    mag_warn_count = 0
    plug_count = 0
    recent_samples: deque[dict[str, float]] = deque(maxlen=max(plug_stable_frames, 3))
    pending_before_sample: dict[str, float] | None = None
    waiting_after_sample = False
    post_reconnect_seen = 0
    after_samples: list[dict[str, float]] = []
    hotplug_events: list[dict[str, float]] = []

    logger.info(f"开始读取 {port} @ {baudrate}，支持热插拔，关闭图窗或 Ctrl+C 结束。")

    try:
        while plt.fignum_exists(fig.number):
            now = time.perf_counter()

            if not connected:
                if now - last_reconnect_try >= reconnect_interval_s:
                    last_reconnect_try = now
                    try:
                        config = TTLSerialConfig(
                            port=port, baudrate=baudrate, timeout_s=timeout_s, write_timeout_s=timeout_s
                        )
                        transport = TTLSerialTransport(config=config)
                        imu = IMU760(transport=transport, debug_enabled=False)
                        imu.open()
                        imu.clear_input_buffer()
                        confirmed_mode = _ensure_ahrs_mode(imu, timeout_s=max(timeout_s, 0.2))
                        algo_mode_text = confirmed_mode.name
                        connected = True
                        logger.info(f"串口已连接：{port} @ {baudrate}, 算法模式={algo_mode_text}")
                        if waiting_after_sample:
                            logger.info(
                                "检测到重连，先丢弃前{}帧，再采集{}帧均值作为插后值。",
                                plug_settle_frames,
                                plug_stable_frames,
                            )
                    except Exception as exc:
                        connected = False
                        imu = None
                        algo_mode_text = "UNKNOWN"
                        logger.warning(f"等待设备连接中 ({port})：{exc}")

                elapsed_wait = now - start_time
                att_status = f"Attitude | Mode={algo_mode_text} | STATUS=DISCONNECTED"
                mag_status = (
                    f"Magnetic Normalized | Mode={algo_mode_text} | MAG={mag_quality_status} | STATUS=DISCONNECTED"
                )
                ax_att.set_title(att_status)
                ax_mag.set_title(mag_status)
                fig.suptitle(
                    f"Port={port}  Baud={baudrate}  STATUS=DISCONNECTED (auto-reconnect {reconnect_interval_s:.1f}s)\n"
                    f"Elapsed={elapsed_wait:.1f}s  Frames={frame_count}",
                    fontsize=11,
                )
                plt.pause(0.05)
                continue

            assert imu is not None
            try:
                frame = imu.read_output_frame(timeout_s=max(timeout_s, 0.05))
            except TimeoutError:
                plt.pause(0.001)
                continue
            except (OSError, SerialException) as exc:
                logger.warning(f"串口断开，准备重连：{exc}")
                if len(recent_samples) > 0 and not waiting_after_sample:
                    before_pool = list(recent_samples)[-plug_stable_frames:]
                    pending_before_sample = _mean_sample(before_pool)
                    waiting_after_sample = True
                    post_reconnect_seen = 0
                    after_samples.clear()
                    logger.info("已记录插拔前稳定样本 (均值帧数={})，等待重连后稳定样本。", len(before_pool))
                try:
                    imu.close()
                except Exception:
                    pass
                connected = False
                imu = None
                plt.pause(0.05)
                continue
            except Exception as exc:
                logger.warning(f"读取异常，尝试重连：{exc}")
                if len(recent_samples) > 0 and not waiting_after_sample:
                    before_pool = list(recent_samples)[-plug_stable_frames:]
                    pending_before_sample = _mean_sample(before_pool)
                    waiting_after_sample = True
                    post_reconnect_seen = 0
                    after_samples.clear()
                    logger.info("已记录异常前稳定样本 (均值帧数={})，等待重连后稳定样本。", len(before_pool))
                try:
                    imu.close()
                except Exception:
                    pass
                connected = False
                imu = None
                plt.pause(0.05)
                continue

            now = time.perf_counter()
            elapsed = now - start_time
            frame_count += 1
            frame_timestamps.append(now)

            euler, magnetic_norm = _extract_observation(frame.payload)
            if euler is None and magnetic_norm is None:
                plt.pause(0.001)
                continue

            times.append(elapsed)
            if euler is not None:
                pitch, roll, yaw = euler
                pitch_hist.append(pitch)
                roll_hist.append(roll)
                yaw_hist.append(yaw)
            else:
                pitch = pitch_hist[-1] if pitch_hist else 0.0
                roll = roll_hist[-1] if roll_hist else 0.0
                yaw = yaw_hist[-1] if yaw_hist else 0.0
                pitch_hist.append(pitch)
                roll_hist.append(roll)
                yaw_hist.append(yaw)

            if magnetic_norm is not None:
                mx, my, mz = magnetic_norm
                magx_hist.append(mx)
                magy_hist.append(my)
                magz_hist.append(mz)
                mag_norm_hist.append(math.sqrt(mx * mx + my * my + mz * mz))
            else:
                mx = magx_hist[-1] if magx_hist else 0.0
                my = magy_hist[-1] if magy_hist else 0.0
                mz = magz_hist[-1] if magz_hist else 0.0
                magx_hist.append(mx)
                magy_hist.append(my)
                magz_hist.append(mz)
                mag_norm_hist.append(math.sqrt(mx * mx + my * my + mz * mz))

            m_norm = mag_norm_hist[-1]
            jump = 0.0
            if len(magx_hist) >= 2:
                dx = magx_hist[-1] - magx_hist[-2]
                dy = magy_hist[-1] - magy_hist[-2]
                dz = magz_hist[-1] - magz_hist[-2]
                jump = math.sqrt(dx * dx + dy * dy + dz * dz)

            mag_ok = abs(m_norm - 1.0) <= mag_norm_abs_tol and jump <= mag_jump_tol
            if mag_ok:
                mag_quality_status = "OK"
                line_mag_norm.set_color("tab:blue")
            else:
                mag_quality_status = "WARN"
                mag_warn_count += 1
                line_mag_norm.set_color("tab:red")

            current_sample = {
                "pitch": pitch,
                "roll": roll,
                "yaw": yaw,
                "mx": mx,
                "my": my,
                "mz": mz,
                "m_norm": m_norm,
            }
            recent_samples.append(current_sample)

            if waiting_after_sample and pending_before_sample is not None:
                post_reconnect_seen += 1
                if post_reconnect_seen > plug_settle_frames:
                    after_samples.append(current_sample)
                    if len(after_samples) >= plug_stable_frames:
                        after_mean = _mean_sample(after_samples[-plug_stable_frames:])
                        plug_count += 1
                        event = {
                            "plug_index": float(plug_count),
                            "before_pitch": pending_before_sample["pitch"],
                            "before_roll": pending_before_sample["roll"],
                            "before_yaw": pending_before_sample["yaw"],
                            "before_mx": pending_before_sample["mx"],
                            "before_my": pending_before_sample["my"],
                            "before_mz": pending_before_sample["mz"],
                            "before_m_norm": pending_before_sample["m_norm"],
                            "after_pitch": after_mean["pitch"],
                            "after_roll": after_mean["roll"],
                            "after_yaw": after_mean["yaw"],
                            "after_mx": after_mean["mx"],
                            "after_my": after_mean["my"],
                            "after_mz": after_mean["mz"],
                            "after_m_norm": after_mean["m_norm"],
                            "event_time_s": elapsed,
                            "before_window_frames": float(min(plug_stable_frames, len(recent_samples))),
                            "after_settle_frames": float(plug_settle_frames),
                            "after_window_frames": float(plug_stable_frames),
                        }
                        hotplug_events.append(event)
                        logger.info(
                            "插拔#{}(稳定均值) 记录：before(|m|={:.4f}, mx={:.4f}, my={:.4f}, mz={:.4f}) "
                            "after(|m|={:.4f}, mx={:.4f}, my={:.4f}, mz={:.4f})",
                            plug_count,
                            event["before_m_norm"],
                            event["before_mx"],
                            event["before_my"],
                            event["before_mz"],
                            event["after_m_norm"],
                            event["after_mx"],
                            event["after_my"],
                            event["after_mz"],
                        )
                        waiting_after_sample = False
                        pending_before_sample = None
                        post_reconnect_seen = 0
                        after_samples.clear()

            line_pitch.set_data(times, pitch_hist)
            line_roll.set_data(times, roll_hist)
            line_yaw.set_data(times, yaw_hist)
            line_magx.set_data(times, magx_hist)
            line_magy.set_data(times, magy_hist)
            line_magz.set_data(times, magz_hist)
            line_mag_norm.set_data(times, mag_norm_hist)

            if len(times) >= 2:
                ax_att.set_xlim(times[0], times[-1] + 1e-6)
            else:
                ax_att.set_xlim(0.0, max(1.0, elapsed))

            att_values = list(pitch_hist) + list(roll_hist) + list(yaw_hist)
            if att_values:
                y_min = min(att_values)
                y_max = max(att_values)
                if math.isclose(y_min, y_max, rel_tol=1e-9, abs_tol=1e-9):
                    y_min -= 1.0
                    y_max += 1.0
                margin = max((y_max - y_min) * 0.1, 1.0)
                ax_att.set_ylim(y_min - margin, y_max + margin)

            mag_values = list(magx_hist) + list(magy_hist) + list(magz_hist) + list(mag_norm_hist)
            if mag_values:
                m_min = min(mag_values)
                m_max = max(mag_values)
                if math.isclose(m_min, m_max, rel_tol=1e-9, abs_tol=1e-9):
                    m_min -= 1.0
                    m_max += 1.0
                m_margin = max((m_max - m_min) * 0.1, 1.0)
                ax_mag.set_ylim(m_min - m_margin, m_max + m_margin)

            total_fps = frame_count / max(elapsed, 1e-6)
            if len(frame_timestamps) >= 2:
                win_dt = frame_timestamps[-1] - frame_timestamps[0]
                realtime_fps = (len(frame_timestamps) - 1) / max(win_dt, 1e-6)
            else:
                realtime_fps = 0.0

            ax_att.set_title(f"Attitude | Mode={algo_mode_text}")
            ax_mag.set_title(f"Magnetic Normalized | Mode={algo_mode_text} | MAG={mag_quality_status}")
            fig.suptitle(
                f"Port={port}  Baud={baudrate}  STATUS=RUNNING  FPS(avg)={total_fps:.1f}  FPS(rt)={realtime_fps:.1f}\n"
                f"Pitch={pitch:.3f}  Roll={roll:.3f}  Yaw={yaw:.3f} | "
                f"mx={mx:.3f} my={my:.3f} mz={mz:.3f} | |m|={m_norm:.3f} jump={jump:.3f} | MAG={mag_quality_status} | PLUG={plug_count}",
                fontsize=11,
            )
            plt.pause(0.001)

    except KeyboardInterrupt:
        logger.info("手动终止。")
    finally:
        if imu is not None:
            try:
                imu.close()
            except Exception:
                pass
        plt.ioff()
        plt.show()

    total_elapsed = time.perf_counter() - start_time
    logger.info(
        f"总帧数={frame_count}, 总时长={total_elapsed:.3f}s, 平均速度={frame_count / max(total_elapsed, 1e-6):.2f} FPS"
    )
    logger.info(f"磁场质量告警次数={mag_warn_count}")
    _write_hotplug_csv(hotplug_events, hotplug_csv)
    if hotplug_events:
        logger.info(f"插拔统计 CSV 已写入：{hotplug_csv} (事件数={len(hotplug_events)})")
    _plot_hotplug_events(hotplug_events)
    if hotplug_events:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="IMU760 读取速度与实时姿态绘图")
    parser.add_argument("--port", type=str, default="COM5", help="串口号，默认 COM5")
    parser.add_argument("--baudrate", type=int, default=460800, help="波特率，默认 460800")
    parser.add_argument("--timeout", type=float, default=0.2, help="串口读超时 (秒)")
    parser.add_argument("--history-size", type=int, default=600, help="图上保留的历史点数量")
    parser.add_argument("--reconnect-interval", type=float, default=1.0, help="断开后重连间隔 (秒)")
    parser.add_argument("--mag-norm-abs-tol", type=float, default=0.15, help="|m|-1 告警阈值")
    parser.add_argument("--mag-jump-tol", type=float, default=0.08, help="归一化磁场相邻采样突变阈值")
    parser.add_argument(
        "--hotplug-csv", type=str, default="test/imu/imu_hotplug_changes.csv", help="插拔变化统计 CSV 输出路径"
    )
    parser.add_argument("--plug-stable-frames", type=int, default=10, help="插拔前后用于均值统计的稳定帧数")
    parser.add_argument("--plug-settle-frames", type=int, default=5, help="重连后先丢弃的帧数")
    args = parser.parse_args()

    try:
        run_realtime_monitor(
            port=args.port,
            baudrate=args.baudrate,
            timeout_s=args.timeout,
            history_size=args.history_size,
            reconnect_interval_s=args.reconnect_interval,
            mag_norm_abs_tol=args.mag_norm_abs_tol,
            mag_jump_tol=args.mag_jump_tol,
            hotplug_csv=args.hotplug_csv,
            plug_stable_frames=args.plug_stable_frames,
            plug_settle_frames=args.plug_settle_frames,
        )
        return 0
    except Exception as exc:
        logger.error(f"运行失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
