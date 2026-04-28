from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imu.imu760 import IMU760, IMU760AlgorithmMode, IMU760EulerAngles
from src.rgbd_camera import Gemini305, SessionOptions

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - 绘图为可选能力
    plt = None


# region 默认参数（优先在这里直接改）
DEFAULT_IMU760_BAUDRATE = 460_800  # IMU760 波特率，单位 bps
DEFAULT_IMU760_TIMEOUT_S = 0.05  # IMU760 单次读取超时，单位 s
DEFAULT_IMU760_PORT_PROBE_TIMEOUT_S = 0.8  # 自动探测单个串口超时，单位 s
DEFAULT_ORBBEC_TIMEOUT_MS = 120  # Orbbec 等待帧超时，单位 ms
DEFAULT_ORBBEC_CAPTURE_FPS = 30  # Orbbec 请求采集帧率，单位 fps
DEFAULT_DURATION_S = 30.0  # 对比采集时长，单位 s
DEFAULT_OUTPUT_CSV = Path(
    "test/rgbd_camera/orbbec_imu_vs_imu760_rpy.csv"
)  # 对比结果 CSV 输出路径
DEFAULT_PLOT = True  # 是否显示 RPY 对比图
DEFAULT_ORBBEC_GYRO_AXIS_MAP = (
    "x,y,z"  # Orbbec gyro 到安装坐标映射，可写成 x,y,z 或 -x,y,z
)
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class RpyDeg:
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True, slots=True)
class CompareSample:
    time_s: float
    orbbec_roll_delta_deg: float
    orbbec_pitch_delta_deg: float
    orbbec_yaw_delta_deg: float
    imu760_roll_delta_deg: float
    imu760_pitch_delta_deg: float
    imu760_yaw_delta_deg: float
    roll_error_deg: float
    pitch_error_deg: float
    yaw_error_deg: float
    orbbec_gyro_x_rad_s: float
    orbbec_gyro_y_rad_s: float
    orbbec_gyro_z_rad_s: float
    imu760_roll_deg: float
    imu760_pitch_deg: float
    imu760_yaw_deg: float


# endregion


# region 主流程
def main(
    imu760_port: str | None = None,
    imu760_baudrate: int = DEFAULT_IMU760_BAUDRATE,
    imu760_timeout_s: float = DEFAULT_IMU760_TIMEOUT_S,
    imu760_port_probe_timeout_s: float = DEFAULT_IMU760_PORT_PROBE_TIMEOUT_S,
    orbbec_timeout_ms: int = DEFAULT_ORBBEC_TIMEOUT_MS,
    orbbec_capture_fps: int = DEFAULT_ORBBEC_CAPTURE_FPS,
    duration_s: float = DEFAULT_DURATION_S,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    plot: bool = DEFAULT_PLOT,
    orbbec_gyro_axis_map: str = DEFAULT_ORBBEC_GYRO_AXIS_MAP,
) -> None:
    logger.info("硬件测试脚本：需要同时连接 Orbbec 相机和 IMU760，未连接时会失败。")
    axis_mapper = _parse_axis_map(orbbec_gyro_axis_map)
    imu760 = IMU760.create_with_auto_port(
        port=imu760_port,
        baudrate=int(imu760_baudrate),
        timeout_s=float(imu760_timeout_s),
        probe_timeout_s=float(imu760_port_probe_timeout_s),
        debug_enabled=False,
    )

    samples: list[CompareSample] = []
    imu760_initial: RpyDeg | None = None
    imu760_current: RpyDeg | None = None
    orbbec_rotation = R.identity()
    last_orbbec_ts_s: float | None = None
    start_time = time.monotonic()

    options = SessionOptions(
        timeout_ms=int(orbbec_timeout_ms),
        preferred_capture_fps=max(1, int(orbbec_capture_fps)),
        enable_imu=True,
    )

    imu760.open()
    try:
        imu760.clear_input_buffer()
        _configure_imu760_for_ahrs(
            imu760=imu760, timeout_s=max(0.2, float(imu760_timeout_s))
        )
        logger.info("IMU760 已切换 AHRS，并设置输出内容为欧拉角。")

        with Gemini305(options=options) as session:
            logger.info(
                "Orbbec IMU 支持状态 accel={} gyro={}",
                session.has_accel_sensor,
                session.has_gyro_sensor,
            )
            if not session.has_gyro_sensor:
                raise RuntimeError("Orbbec 未启用 gyro，无法积分计算 RPY 变化。")

            while time.monotonic() - start_time < float(duration_s):
                frames = session.wait_for_frames(timeout_ms=int(orbbec_timeout_ms))
                if frames is None:
                    continue

                orbbec_sample = session.get_imu_sample_from_frames(frames)
                if orbbec_sample.gyro_rad_s is None:
                    continue

                latest_imu760 = _read_latest_imu760_rpy(
                    imu760=imu760, timeout_s=float(imu760_timeout_s)
                )
                if latest_imu760 is not None:
                    imu760_current = latest_imu760

                if imu760_current is None:
                    continue

                if imu760_initial is None:
                    imu760_initial = imu760_current
                    last_orbbec_ts_s = _orbbec_sample_time_s(
                        orbbec_sample.gyro_timestamp_us
                    )
                    logger.info(
                        "初始姿态对齐：roll={:.3f} deg pitch={:.3f} deg yaw={:.3f} deg",
                        imu760_initial.roll,
                        imu760_initial.pitch,
                        imu760_initial.yaw,
                    )
                    continue

                now_ts_s = _orbbec_sample_time_s(orbbec_sample.gyro_timestamp_us)
                if last_orbbec_ts_s is None:
                    last_orbbec_ts_s = now_ts_s
                    continue

                dt_s = max(0.0, min(now_ts_s - last_orbbec_ts_s, 0.2))
                last_orbbec_ts_s = now_ts_s
                gyro_rad_s = axis_mapper(
                    np.asarray(orbbec_sample.gyro_rad_s, dtype=np.float64)
                )
                if dt_s > 0.0:
                    orbbec_rotation = orbbec_rotation * R.from_rotvec(gyro_rad_s * dt_s)

                orbbec_delta = _rotation_to_rpy_delta(orbbec_rotation)
                imu760_delta = _imu760_rpy_delta(
                    current=imu760_current, initial=imu760_initial
                )
                samples.append(
                    CompareSample(
                        time_s=time.monotonic() - start_time,
                        orbbec_roll_delta_deg=orbbec_delta.roll,
                        orbbec_pitch_delta_deg=orbbec_delta.pitch,
                        orbbec_yaw_delta_deg=orbbec_delta.yaw,
                        imu760_roll_delta_deg=imu760_delta.roll,
                        imu760_pitch_delta_deg=imu760_delta.pitch,
                        imu760_yaw_delta_deg=imu760_delta.yaw,
                        roll_error_deg=_angle_delta_deg(
                            orbbec_delta.roll, imu760_delta.roll
                        ),
                        pitch_error_deg=_angle_delta_deg(
                            orbbec_delta.pitch, imu760_delta.pitch
                        ),
                        yaw_error_deg=_angle_delta_deg(
                            orbbec_delta.yaw, imu760_delta.yaw
                        ),
                        orbbec_gyro_x_rad_s=float(gyro_rad_s[0]),
                        orbbec_gyro_y_rad_s=float(gyro_rad_s[1]),
                        orbbec_gyro_z_rad_s=float(gyro_rad_s[2]),
                        imu760_roll_deg=imu760_current.roll,
                        imu760_pitch_deg=imu760_current.pitch,
                        imu760_yaw_deg=imu760_current.yaw,
                    )
                )

    finally:
        imu760.close()

    if not samples:
        raise RuntimeError("没有采集到有效对比样本，请检查 Orbbec IMU 和 IMU760 输出。")

    _write_samples_csv(samples=samples, output_csv=output_csv)
    _log_summary(samples)
    if plot:
        _plot_samples(samples)


# endregion


# region 设备与数据处理
def _configure_imu760_for_ahrs(imu760: IMU760, timeout_s: float) -> None:
    imu760.set_algorithm_mode(
        IMU760AlgorithmMode.AHRS, save_to_flash=False, timeout_s=timeout_s
    )
    imu760.set_output_content_mask(
        IMU760.OUTPUT_MASK_EULER, save_to_flash=False, timeout_s=timeout_s
    )
    imu760.clear_input_buffer()


def _read_latest_imu760_rpy(imu760: IMU760, timeout_s: float) -> RpyDeg | None:
    deadline = time.monotonic() + max(timeout_s, 0.01)
    latest: RpyDeg | None = None
    while time.monotonic() < deadline:
        try:
            payload = imu760.read_output_payload(timeout_s=max(0.005, timeout_s / 2.0))
        except TimeoutError:
            break
        for item in payload:
            if isinstance(item, IMU760EulerAngles):
                latest = RpyDeg(
                    roll=float(item.roll),
                    pitch=float(item.pitch),
                    yaw=float(item.yaw),
                )
    return latest


def _orbbec_sample_time_s(timestamp_us: int | None) -> float:
    if timestamp_us is None:
        return time.monotonic()
    return float(timestamp_us) * 1e-6


def _rotation_to_rpy_delta(rotation: R) -> RpyDeg:
    roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
    return RpyDeg(
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
    )


def _imu760_rpy_delta(current: RpyDeg, initial: RpyDeg) -> RpyDeg:
    return RpyDeg(
        roll=_angle_delta_deg(current.roll, initial.roll),
        pitch=_angle_delta_deg(current.pitch, initial.pitch),
        yaw=_angle_delta_deg(current.yaw, initial.yaw),
    )


def _angle_delta_deg(current: float, initial: float) -> float:
    return (float(current) - float(initial) + 180.0) % 360.0 - 180.0


def _parse_axis_map(axis_map: str):
    tokens = [token.strip().lower() for token in axis_map.split(",")]
    if len(tokens) != 3:
        raise ValueError("orbbec_gyro_axis_map 必须包含 3 个轴，例如 x,y,z 或 -x,y,z")

    axis_index = {"x": 0, "y": 1, "z": 2}
    parsed: list[tuple[int, float]] = []
    for token in tokens:
        sign = -1.0 if token.startswith("-") else 1.0
        name = token[1:] if token.startswith("-") else token
        if name not in axis_index:
            raise ValueError(f"不支持的轴映射：{token}")
        parsed.append((axis_index[name], sign))

    def mapper(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [sign * values[index] for index, sign in parsed], dtype=np.float64
        )

    return mapper


# endregion


# region 输出
def _write_samples_csv(samples: list[CompareSample], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(samples[0].__dataclass_fields__.keys())
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {key: getattr(sample, key) for key in sample.__dataclass_fields__}
            )
    logger.success("对比结果已保存：{}", output_csv)


def _log_summary(samples: list[CompareSample]) -> None:
    roll_rms = _rms([sample.roll_error_deg for sample in samples])
    pitch_rms = _rms([sample.pitch_error_deg for sample in samples])
    yaw_rms = _rms([sample.yaw_error_deg for sample in samples])
    logger.success(
        "RPY 误差 RMS：roll={:.3f} deg pitch={:.3f} deg yaw={:.3f} deg，样本数={} 个",
        roll_rms,
        pitch_rms,
        yaw_rms,
        len(samples),
    )


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _plot_samples(samples: list[CompareSample]) -> None:
    if plt is None:
        logger.warning("未安装 matplotlib，跳过绘图。")
        return
    times = [sample.time_s for sample in samples]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    series = [
        (
            "roll",
            "Roll delta (deg)",
            "orbbec_roll_delta_deg",
            "imu760_roll_delta_deg",
            "roll_error_deg",
        ),
        (
            "pitch",
            "Pitch delta (deg)",
            "orbbec_pitch_delta_deg",
            "imu760_pitch_delta_deg",
            "pitch_error_deg",
        ),
        (
            "yaw",
            "Yaw delta (deg)",
            "orbbec_yaw_delta_deg",
            "imu760_yaw_delta_deg",
            "yaw_error_deg",
        ),
    ]
    for ax, (_, ylabel, orbbec_key, imu760_key, error_key) in zip(
        axes, series, strict=True
    ):
        ax.plot(
            times,
            [getattr(sample, orbbec_key) for sample in samples],
            label="Orbbec gyro integration",
        )
        ax.plot(
            times,
            [getattr(sample, imu760_key) for sample in samples],
            label="IMU760 AHRS",
        )
        ax.plot(
            times,
            [getattr(sample, error_key) for sample in samples],
            linestyle="--",
            label="error",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()


# endregion


# region CLI
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 Orbbec 内置 IMU 与 IMU760 AHRS 的 RPY 变化"
    )
    parser.add_argument(
        "--imu760-port",
        type=str,
        default=None,
        help="IMU760 串口号；不填时自动探测",
    )
    parser.add_argument("--imu760-baudrate", type=int, default=DEFAULT_IMU760_BAUDRATE)
    parser.add_argument(
        "--imu760-timeout-s", type=float, default=DEFAULT_IMU760_TIMEOUT_S
    )
    parser.add_argument(
        "--imu760-port-probe-timeout-s",
        type=float,
        default=DEFAULT_IMU760_PORT_PROBE_TIMEOUT_S,
    )
    parser.add_argument(
        "--orbbec-timeout-ms", type=int, default=DEFAULT_ORBBEC_TIMEOUT_MS
    )
    parser.add_argument(
        "--orbbec-capture-fps", type=int, default=DEFAULT_ORBBEC_CAPTURE_FPS
    )
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=DEFAULT_PLOT
    )
    parser.add_argument(
        "--orbbec-gyro-axis-map", type=str, default=DEFAULT_ORBBEC_GYRO_AXIS_MAP
    )
    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = _parse_cli()
        main(
            imu760_port=args.imu760_port,
            imu760_baudrate=args.imu760_baudrate,
            imu760_timeout_s=args.imu760_timeout_s,
            imu760_port_probe_timeout_s=args.imu760_port_probe_timeout_s,
            orbbec_timeout_ms=args.orbbec_timeout_ms,
            orbbec_capture_fps=args.orbbec_capture_fps,
            duration_s=args.duration_s,
            output_csv=args.output_csv,
            plot=args.plot,
            orbbec_gyro_axis_map=args.orbbec_gyro_axis_map,
        )
    else:
        main()

# endregion
