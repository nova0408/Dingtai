from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imu.imu760 import (
    IMU760,
    IMU760Acceleration,
    IMU760AlgorithmMode,
    IMU760AngularVelocity,
    IMU760EulerAngles,
    IMU760MagneticNormalized,
    IMU760MagneticStrength,
    IMU760QuaternionData,
    decode_output_payload,
)
from src.imu.ttl import TTLSerialConfig, TTLSerialTransport
from src.utils.datas import Degree, Quaternion


class FakeTransport:
    def __init__(self, read_chunks: list[bytes] | None = None) -> None:
        self._is_open = True
        self._read_chunks = list(read_chunks or [])
        self.written = bytearray()

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def in_waiting(self) -> int:
        return sum(len(chunk) for chunk in self._read_chunks)

    @property
    def timeout(self) -> float | None:
        return 0.1

    @timeout.setter
    def timeout(self, value: float | None) -> None:
        _ = value

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def read(self, size: int = 1) -> bytes:
        if not self._read_chunks:
            return b""
        chunk = self._read_chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self._read_chunks.insert(0, chunk[size:])
        return chunk[:size]

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        self._read_chunks.clear()


def test_decode_output_payload_appendix_example() -> None:
    payload = bytes.fromhex(
        "10 0C 3B 21 FE FF 89 2C FE FF 59 9C 6A FF "
        "20 0C D1 A2 02 00 FA B1 05 00 38 D0 00 00 "
        "30 0C 20 FF F1 07 D0 A0 EA F4 C0 8E C6 EF "
        "31 0C B4 08 02 00 A2 29 FD FF B8 D8 FB FF "
        "40 0C CB E4 F4 FF B1 BE 09 00 30 32 B6 F6 "
        "41 10 D4 31 03 00 4D EC FF FF 86 E5 FF FF A9 14 F1 FF"
    )

    fields = decode_output_payload(payload)

    assert isinstance(fields[0], IMU760Acceleration)
    assert fields[0].ax == pytest.approx(-0.122565)
    assert fields[0].ay == pytest.approx(-0.119671)
    assert fields[0].az == pytest.approx(-9.790375)

    assert isinstance(fields[1], IMU760AngularVelocity)
    assert fields[1].wy == pytest.approx(0.373242)

    assert isinstance(fields[2], IMU760MagneticNormalized)
    assert fields[2].mz == pytest.approx(-272.2)

    assert isinstance(fields[3], IMU760MagneticStrength)
    assert fields[3].mx == pytest.approx(133.3)

    assert isinstance(fields[4], IMU760EulerAngles)
    assert isinstance(fields[4].pitch, Degree)
    assert float(fields[4].yaw) == pytest.approx(-155.831760, abs=3e-4)

    assert isinstance(fields[5], IMU760QuaternionData)
    quat = fields[5].as_quat()
    assert isinstance(quat, Quaternion)
    assert fields[5].q4 == pytest.approx(-0.977751, abs=1e-6)


def test_build_command_frame_matches_appendix_examples() -> None:
    assert IMU760.build_command_frame(0x03, 0x01, bytes([0x05])) == bytes.fromhex("59 53 03 09 00 05 11 2C")
    assert IMU760.build_command_frame(0x03, 0x02, bytes([0x05])) == bytes.fromhex("59 53 03 0A 00 05 12 2F")
    assert IMU760.build_command_frame(0x04, 0x02, bytes.fromhex("F8 00")) == bytes.fromhex("59 53 04 12 00 F8 00 0E 4C")


def test_parse_command_frame_matches_appendix_response() -> None:
    response = IMU760.parse_command_frame(bytes.fromhex("59 53 03 09 00 00 0C 27"))
    assert response.data_class == 0x03
    assert response.operator == 0x01
    assert response.data_length == 1
    assert response.data == bytes([0x00])


def test_debug_logs_tx_rx_when_enabled() -> None:
    fake = FakeTransport(read_chunks=[bytes.fromhex("59 53 03 09 00 00 0C 27")])
    imu = IMU760(transport=fake, debug_enabled=True)

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    try:
        imu.send_command(
            data_class=0x03,
            operator=0x01,
            data=bytes([0x05]),
            expect_response=True,
            timeout_s=0.2,
        )
    finally:
        logger.remove(sink_id)

    assert any("IMU760 TX:" in msg for msg in messages)
    assert any("IMU760 RX:" in msg for msg in messages)


@pytest.mark.hardware
def test_hardware_com5_460800_read_output_frame() -> None:
    """真机串口冒烟测试（默认 COM5 / 460800）。

    说明：
    - 默认跳过，避免在无硬件环境导致 CI 失败；
    - 设置环境变量 `IMU760_RUN_HARDWARE_TEST=1` 后执行。
    """

    if os.getenv("IMU760_RUN_HARDWARE_TEST", "0") != "1":
        pytest.skip("未启用真机测试，设置 IMU760_RUN_HARDWARE_TEST=1 后再运行。")

    _run_hardware_smoke(port="COM5", baudrate=460800, timeout_s=2.0, debug_enabled=True)


def _run_hardware_smoke(
    port: str = "COM5",
    baudrate: int = 460800,
    timeout_s: float = 2.0,
    debug_enabled: bool = True,
) -> None:
    """执行一次真机串口读取冒烟测试。"""

    config = TTLSerialConfig(port=port, baudrate=baudrate, timeout_s=0.2, write_timeout_s=0.2)
    transport = TTLSerialTransport(config=config)
    imu = IMU760(transport=transport, debug_enabled=debug_enabled)

    try:
        imu.open()
        imu.clear_input_buffer()
        frame = imu.read_output_frame(timeout_s=timeout_s)
        assert frame.payload_length > 0
        decoded = decode_output_payload(frame.payload)
        assert len(decoded) > 0
    finally:
        imu.close()


def _query_data(imu: IMU760, data_class: int, query_data: bytes = b"", timeout_s: float = 0.5) -> bytes | None:
    try:
        return imu.query(data_class=data_class, query_data=query_data, timeout_s=timeout_s).data
    except Exception as exc:
        logger.debug(f"[WARN] 查询失败 data_class=0x{data_class:02X}, query={query_data.hex(' ').upper()}: {exc}")
        return None


def _decode_product_info(data: bytes | None) -> str:
    if not data:
        return "N/A"
    payload = data[1:] if data and data[0] == 0x02 else data
    text = payload.decode("ascii", errors="replace").strip("\x00").strip()
    return text or payload.hex(" ").upper()


def _decode_single_code(data: bytes | None) -> int | None:
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    if len(data) >= 2:
        return data[-1]
    return None


def _decode_mask(data: bytes | None) -> int | None:
    if not data:
        return None
    if len(data) == 2:
        return int.from_bytes(data, byteorder="little", signed=False)
    if len(data) >= 3:
        return int.from_bytes(data[-2:], byteorder="little", signed=False)
    return None


def _iter_output_packets(payload: bytes) -> list[tuple[int, bytes]]:
    packets: list[tuple[int, bytes]] = []
    cursor = 0
    while cursor + 2 <= len(payload):
        data_id = payload[cursor]
        data_len = payload[cursor + 1]
        cursor += 2
        if cursor + data_len > len(payload):
            break
        data = payload[cursor : cursor + data_len]
        cursor += data_len
        packets.append((data_id, data))
    return packets


def _read_i32_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=True)


def _read_u32_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=False)


def _read_i16_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=True)


def _format_output_packet(data_id: int, data: bytes) -> str:
    if data_id == 0x01 and len(data) == 2:
        return f"IMU 温度：{_read_i16_le(data) * 0.01:.2f} °C"
    if data_id == 0x10 and len(data) == 12:
        ax, ay, az = (_read_i32_le(data[0:4]) * 1e-6, _read_i32_le(data[4:8]) * 1e-6, _read_i32_le(data[8:12]) * 1e-6)
        return f"加速度：ax={ax:.6f}, ay={ay:.6f}, az={az:.6f} m/s²"
    if data_id == 0x20 and len(data) == 12:
        wx, wy, wz = (_read_i32_le(data[0:4]) * 1e-6, _read_i32_le(data[4:8]) * 1e-6, _read_i32_le(data[8:12]) * 1e-6)
        return f"角速度：wx={wx:.6f}, wy={wy:.6f}, wz={wz:.6f} deg/s"
    if data_id == 0x30 and len(data) == 12:
        mx, my, mz = (_read_i32_le(data[0:4]) * 1e-6, _read_i32_le(data[4:8]) * 1e-6, _read_i32_le(data[8:12]) * 1e-6)
        return f"磁场归一化：mx={mx:.6f}, my={my:.6f}, mz={mz:.6f}"
    if data_id == 0x31 and len(data) == 12:
        mx, my, mz = (_read_i32_le(data[0:4]) * 1e-3, _read_i32_le(data[4:8]) * 1e-3, _read_i32_le(data[8:12]) * 1e-3)
        return f"磁场强度：mx={mx:.3f}, my={my:.3f}, mz={mz:.3f} mGauss"
    if data_id == 0x40 and len(data) == 12:
        pitch = _read_i32_le(data[0:4]) * 1e-6
        roll = _read_i32_le(data[4:8]) * 1e-6
        yaw = _read_i32_le(data[8:12]) * 1e-6
        return f"欧拉角：pitch={pitch:.6f}, roll={roll:.6f}, yaw={yaw:.6f} deg"
    if data_id == 0x41 and len(data) == 16:
        q1 = _read_i32_le(data[0:4]) * 1e-6
        q2 = _read_i32_le(data[4:8]) * 1e-6
        q3 = _read_i32_le(data[8:12]) * 1e-6
        q4 = _read_i32_le(data[12:16]) * 1e-6
        return f"四元数：q1={q1:.6f}, q2={q2:.6f}, q3={q3:.6f}, q4={q4:.6f}"
    if data_id == 0x51 and len(data) == 4:
        return f"采样时间戳：{_read_u32_le(data)} us"
    if data_id == 0x52 and len(data) == 4:
        return f"DataReady 时间戳：{_read_u32_le(data)} us"
    return f"未知字段 ID=0x{data_id:02X}, RAW={data.hex(' ').upper()}"


def _collect_and_log_all_info(
    port: str = "COM5",
    baudrate: int = 460800,
    timeout_s: float = 2.0,
    debug_enabled: bool = True,
) -> None:
    config = TTLSerialConfig(port=port, baudrate=baudrate, timeout_s=0.2, write_timeout_s=0.2)
    transport = TTLSerialTransport(config=config)
    imu = IMU760(transport=transport, debug_enabled=debug_enabled)

    baud_map = {v: k for k, v in IMU760.BAUDRATE_CODE_MAP.items()}
    rate_map = {v: k for k, v in IMU760.OUTPUT_RATE_CODE_MAP.items()}
    output_mask_bits: list[tuple[int, str]] = [
        (15, "时间戳输出"),
        (10, "IMU 温度输出"),
        (7, "加速度输出"),
        (6, "角速度输出"),
        (5, "磁场强度输出"),
        (4, "欧拉角输出"),
        (3, "四元数输出"),
    ]

    boot_mode_map = {0x00: "上电无等待启动", 0x01: "上电静置 5 秒启动"}
    algo_mode_map = {0x01: "AHRS", 0x02: "VRU(默认)", 0x03: "IMU"}
    platform_mode_map = {
        0x00: "工程机械挖机等应用",
        0x01: "工程机械塔吊应用",
        0x02: "工程机械碎石机应用",
        0x03: "车载 AGV 应用",
        0x04: "工程矿车应用",
        0x05: "船载应用",
        0x06: "风机应用",
        0x07: "机器人或农机应用",
        0x08: "小艇应用",
    }

    logger.debug("=" * 70)
    logger.debug(f"IMU760 全项读取 端口={port} 波特率={baudrate}")
    logger.debug("=" * 70)

    try:
        imu.open()
        imu.clear_input_buffer()

        debug_str = '"\n[1] 参数查询（交互协议）\n'
        product_info = _decode_product_info(
            _query_data(imu, data_class=0x00, query_data=bytes([0x02]), timeout_s=timeout_s)
        )
        boot_mode_code = _decode_single_code(_query_data(imu, data_class=0x01, timeout_s=timeout_s))
        baud_code = _decode_single_code(_query_data(imu, data_class=0x02, timeout_s=timeout_s))
        rate_code = _decode_single_code(_query_data(imu, data_class=0x03, timeout_s=timeout_s))
        output_mask = _decode_mask(_query_data(imu, data_class=0x04, timeout_s=timeout_s))
        algo_data = _query_data(imu, data_class=0x4D, query_data=bytes([0x02]), timeout_s=timeout_s)
        gyro_bias_data = _query_data(imu, data_class=0x08, query_data=bytes([0x00]), timeout_s=timeout_s)
        platform_data = _query_data(imu, data_class=0x07, query_data=bytes([0x00]), timeout_s=timeout_s)
        mag_mode_01 = _query_data(imu, data_class=0x06, query_data=bytes([0x01]), timeout_s=timeout_s)
        mag_mode_02 = _query_data(imu, data_class=0x06, query_data=bytes([0x02]), timeout_s=timeout_s)

        algo_code = _decode_single_code(algo_data)
        gyro_bias_code = _decode_single_code(gyro_bias_data)
        platform_code = _decode_single_code(platform_data)
        mag_mode_01_code = _decode_single_code(mag_mode_01)
        mag_mode_02_code = _decode_single_code(mag_mode_02)

        debug_str = f"\n- 产品信息：{product_info}"
        debug_str += f"\n- 上电模式：{boot_mode_map.get(boot_mode_code, hex(boot_mode_code))} (code={boot_mode_code})"
        debug_str += f"\n- 波特率：{baud_map.get(baud_code, '未知')} bps (code={baud_code})"
        debug_str += f"\n- 输出频率：{rate_map.get(rate_code, '未知')} Hz (code={rate_code})"
        debug_str += f"\n- 算法模式：{algo_mode_map.get(algo_code, '未知')} (code={algo_code})"
        debug_str += f"\n- 应用平台模式：{platform_mode_map.get(platform_code, '未知')} (code={platform_code})"
        debug_str += f"\n- 陀螺零偏静态补偿策略：{'有效' if gyro_bias_code == 0x01 else '无效'} (code={gyro_bias_code})"
        debug_str += f"\n- 磁场校准模式 (0x01): code={mag_mode_01_code}"
        debug_str += f"\n- 磁场校准模式 (0x02): code={mag_mode_02_code}"

        if output_mask is not None:
            enabled = [name for bit, name in output_mask_bits if output_mask & (1 << bit)]
            debug_str += f"\n- 输出内容掩码：0x{output_mask:04X}"
            debug_str += f"\n-  已启用项：{', '.join(enabled) if enabled else '无'}"
        else:
            debug_str += "\n- 输出内容掩码：N/A"

        debug_str += "\n[2] 实时输出帧读取（输出协议）"
        frame = imu.read_output_frame(timeout_s=timeout_s)
        debug_str += f"\n- 帧序号 TID: {frame.tid}"
        debug_str += f"\n- 负载长度：{frame.payload_length} bytes"
        debug_str += f"\n- 校验：CK1=0x{frame.checksum_1:02X}, CK2=0x{frame.checksum_2:02X}"

        packets = _iter_output_packets(frame.payload)
        if not packets:
            logger.debug("- 未解析到有效数据包")
        else:
            debug_str += "- 数据包明细："
            for idx, (data_id, data) in enumerate(packets, start=1):
                debug_str += f"\n  {idx:02d}. {_format_output_packet(data_id, data)}"

        logger.debug(debug_str)
    finally:
        imu.close()


def test_set_algorithm_mode_uses_enum() -> None:
    ack = IMU760.build_command_frame(data_class=0x4D, operator=0x01, data=bytes([0x00]))
    fake = FakeTransport(read_chunks=[ack])
    imu = IMU760(transport=fake)

    imu.set_algorithm_mode(IMU760AlgorithmMode.VRU, save_to_flash=False, timeout_s=0.2)

    assert bytes(fake.written) == IMU760.build_command_frame(
        data_class=0x4D,
        operator=0x01,
        data=bytes([0x02, int(IMU760AlgorithmMode.VRU)]),
    )


def main() -> int:
    """VSCode 直接运行入口：执行 COM5/460800 全项读取并友好打印。"""

    try:
        _collect_and_log_all_info(port="COM5", baudrate=460800, timeout_s=2.0, debug_enabled=True)
        logger.success("IMU760 全项读取完成。")
        return 0
    except Exception as exc:
        logger.warning(f"IMU760 全项读取失败：{exc}")
        return 1


if __name__ == "__main__":
    main()
