# -*- coding: utf-8 -*-
from __future__ import annotations

# region 导入
from dataclasses import dataclass
from enum import IntEnum
from time import monotonic

from loguru import logger

from src.utils.Datas import Degree, Quaternion

from .ttl import TTLSerialTransport

# endregion


# region 常量与枚举
FRAME_HEADER = b"\x59\x53"


class IMU760Operator(IntEnum):
    QUERY = 0x00
    WRITE_RAM = 0x01
    WRITE_FLASH = 0x02


class IMU760DataClass(IntEnum):
    PRODUCT_INFO = 0x00
    BOOT_MODE = 0x01
    BAUDRATE = 0x02
    OUTPUT_RATE = 0x03
    OUTPUT_CONTENT = 0x04
    CALIBRATION_PARAM = 0x05
    FUNCTION_MODE = 0x4D
    MAG_CALIBRATION_MODE = 0x06
    PLATFORM_MODE = 0x07
    GYRO_BIAS_STRATEGY = 0x08


class IMU760DataId(IntEnum):
    IMU_TEMPERATURE = 0x01
    ACC = 0x10
    GYRO = 0x20
    MAG_NORMALIZED = 0x30
    MAG_STRENGTH = 0x31
    EULER = 0x40
    QUATERNION = 0x41
    SAMPLE_TIMESTAMP = 0x51
    DATAREADY_TIMESTAMP = 0x52


class IMU760ModeSubtype(IntEnum):
    ALGO_MODE = 0x02
    GYRO_BIAS_INIT = 0x50
    MAG_ROTATION_CALIBRATION = 0x20
    MAG_MULTIPOINT_CALIBRATION = 0x30


class IMU760AlgorithmMode(IntEnum):
    """IMU760 算法模式。"""

    AHRS = 0x01  # 姿态航向参考模式
    VRU = 0x02  # 垂直参考模式（手册默认）
    IMU = 0x03  # 原始惯导模式


# endregion


# region 数据对象
IMU760Temperature = float
IMU760SampleTimestamp = int
IMU760DatareadyTimestamp = int


@dataclass(frozen=True, slots=True)
class IMU760Acceleration:
    """加速度，单位 m/s^2。"""

    ax: float
    ay: float
    az: float


@dataclass(frozen=True, slots=True)
class IMU760AngularVelocity:
    """角速度，单位 deg/s。"""

    wx: float
    wy: float
    wz: float


@dataclass(frozen=True, slots=True)
class IMU760MagneticNormalized:
    """归一化磁场，无量纲。"""

    mx: float
    my: float
    mz: float


@dataclass(frozen=True, slots=True)
class IMU760MagneticStrength:
    """磁场强度，单位 mGauss。"""

    mx: float
    my: float
    mz: float


@dataclass(frozen=True, slots=True)
class IMU760EulerAngles:
    """欧拉角（FRD），单位度。"""

    pitch: Degree
    roll: Degree
    yaw: Degree


@dataclass(frozen=True, slots=True)
class IMU760QuaternionData:
    """手册原始四元数定义（q1/q2/q3/q4）。"""

    q1: float
    q2: float
    q3: float
    q4: float

    def as_quat(self, index_map: tuple[int, int, int, int] = (0, 1, 2, 3)) -> Quaternion:
        """转换为项目 Quaternion 类型。

        Parameters
        ----------
        index_map : tuple[int, int, int, int], default=(0, 1, 2, 3)
            手册 `(q1,q2,q3,q4)` 到 `(w,x,y,z)` 的索引映射。
            例如当手册 `q2` 为标量 `w` 时，映射可设为 `(1,0,2,3)`。

        Returns
        -------
        Quaternion
            转换后的四元数对象，内部顺序为 `(w,x,y,z)`。
        """

        values = (self.q1, self.q2, self.q3, self.q4)
        w, x, y, z = (values[index_map[0]], values[index_map[1]], values[index_map[2]], values[index_map[3]])
        return Quaternion(w=w, x=x, y=y, z=z)


@dataclass(frozen=True, slots=True)
class IMU760UnknownField:
    """未知数据字段。"""

    data_id: int
    raw: bytes


IMU760DecodedData = (
    IMU760Temperature
    | IMU760Acceleration
    | IMU760AngularVelocity
    | IMU760MagneticNormalized
    | IMU760MagneticStrength
    | IMU760EulerAngles
    | IMU760QuaternionData
    | IMU760SampleTimestamp
    | IMU760DatareadyTimestamp
    | IMU760UnknownField
)


@dataclass(frozen=True, slots=True)
class IMU760OutputFrame:
    tid: int
    payload_length: int
    payload: bytes
    checksum_1: int
    checksum_2: int


@dataclass(frozen=True, slots=True)
class IMU760CommandFrame:
    data_class: int
    operator: int
    data_length: int
    data: bytes
    checksum_1: int
    checksum_2: int


# endregion


# region 工具函数
def imu760_checksum(data: bytes) -> tuple[int, int]:
    """计算 IMU760 双字节校验和。"""
    ck1 = 0
    ck2 = 0
    for value in data:
        ck1 = (ck1 + value) & 0xFF
        ck2 = (ck2 + ck1) & 0xFF
    return ck1, ck2


def _read_i16_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=True)


def _read_i32_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=True)


def _read_u16_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=False)


def _read_u32_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=False)


def decode_output_payload(message: bytes) -> tuple[IMU760DecodedData, ...]:
    """将输出 payload 解码为结构化对象序列。

    Parameters
    ----------
    message : bytes
        输出协议中 LEN 后的数据域。

    Returns
    -------
    tuple[IMU760DecodedData, ...]
        按报文中出现顺序返回的解码对象元组。
    """

    cursor = 0
    decoded: list[IMU760DecodedData] = []

    while cursor + 2 <= len(message):
        data_id = message[cursor]
        data_len = message[cursor + 1]
        cursor += 2
        if cursor + data_len > len(message):
            break

        data = message[cursor : cursor + data_len]
        cursor += data_len

        if data_id == IMU760DataId.IMU_TEMPERATURE and data_len == 2:
            decoded.append(_read_i16_le(data) * 0.01)
        elif data_id == IMU760DataId.ACC and data_len == 12:
            decoded.append(
                IMU760Acceleration(
                    ax=_read_i32_le(data[0:4]) * 1e-6,
                    ay=_read_i32_le(data[4:8]) * 1e-6,
                    az=_read_i32_le(data[8:12]) * 1e-6,
                )
            )
        elif data_id == IMU760DataId.GYRO and data_len == 12:
            decoded.append(
                IMU760AngularVelocity(
                    wx=_read_i32_le(data[0:4]) * 1e-6,
                    wy=_read_i32_le(data[4:8]) * 1e-6,
                    wz=_read_i32_le(data[8:12]) * 1e-6,
                )
            )
        elif data_id == IMU760DataId.MAG_NORMALIZED and data_len == 12:
            decoded.append(
                IMU760MagneticNormalized(
                    mx=_read_i32_le(data[0:4]) * 1e-6,
                    my=_read_i32_le(data[4:8]) * 1e-6,
                    mz=_read_i32_le(data[8:12]) * 1e-6,
                )
            )
        elif data_id == IMU760DataId.MAG_STRENGTH and data_len == 12:
            decoded.append(
                IMU760MagneticStrength(
                    mx=_read_i32_le(data[0:4]) * 1e-3,
                    my=_read_i32_le(data[4:8]) * 1e-3,
                    mz=_read_i32_le(data[8:12]) * 1e-3,
                )
            )
        elif data_id == IMU760DataId.EULER and data_len == 12:
            decoded.append(
                IMU760EulerAngles(
                    pitch=Degree(_read_i32_le(data[0:4]) * 1e-6),
                    roll=Degree(_read_i32_le(data[4:8]) * 1e-6),
                    yaw=Degree(_read_i32_le(data[8:12]) * 1e-6),
                )
            )
        elif data_id == IMU760DataId.QUATERNION and data_len == 16:
            decoded.append(
                IMU760QuaternionData(
                    q1=_read_i32_le(data[0:4]) * 1e-6,
                    q2=_read_i32_le(data[4:8]) * 1e-6,
                    q3=_read_i32_le(data[8:12]) * 1e-6,
                    q4=_read_i32_le(data[12:16]) * 1e-6,
                )
            )
        elif data_id == IMU760DataId.SAMPLE_TIMESTAMP and data_len == 4:
            decoded.append(_read_u32_le(data))
        elif data_id == IMU760DataId.DATAREADY_TIMESTAMP and data_len == 4:
            decoded.append(_read_u32_le(data))
        else:
            decoded.append(IMU760UnknownField(data_id=data_id, raw=bytes(data)))

    return tuple(decoded)


# endregion


# region 高层封装
class IMU760:
    """IMU760 高层通信封装。"""

    BAUDRATE_CODE_MAP: dict[int, int] = {
        9600: 0x01,
        38400: 0x02,
        115200: 0x03,
        460800: 0x04,
        921600: 0x05,
        19200: 0x06,
        57600: 0x07,
        76800: 0x08,
        230400: 0x09,
    }

    OUTPUT_RATE_CODE_MAP: dict[int, int] = {
        1: 0x01,
        2: 0x02,
        5: 0x03,
        10: 0x04,
        20: 0x05,
        25: 0x06,
        50: 0x07,
        100: 0x08,
        200: 0x09,
        400: 0x0A,
    }

    OUTPUT_MASK_TIMESTAMP = 1 << 15
    OUTPUT_MASK_IMU_TEMPERATURE = 1 << 10
    OUTPUT_MASK_ACC = 1 << 7
    OUTPUT_MASK_GYRO = 1 << 6
    OUTPUT_MASK_MAG_STRENGTH = 1 << 5
    OUTPUT_MASK_EULER = 1 << 4
    OUTPUT_MASK_QUATERNION = 1 << 3
    _VALID_DATA_CLASS_VALUES = {int(v) for v in IMU760DataClass}

    def __init__(self, transport: TTLSerialTransport, debug_enabled: bool = False) -> None:
        """创建 IMU760 通信对象。

        Parameters
        ----------
        transport : TTLSerialTransport
            底层 TTL 串口传输对象。
        debug_enabled : bool, default=False
            是否输出 TX/RX 调试日志（`loguru.debug`）。
        """
        self._transport = transport
        self._debug_enabled = debug_enabled
        self._rx_buffer = bytearray()

    @property
    def transport(self) -> TTLSerialTransport:
        """获取底层传输对象。

        Returns
        -------
        TTLSerialTransport
            当前绑定的串口传输实例。
        """
        return self._transport

    @property
    def debug_enabled(self) -> bool:
        """获取当前调试开关状态。

        Returns
        -------
        bool
            `True` 表示启用 TX/RX debug 输出。
        """
        return self._debug_enabled

    def set_debug_enabled(self, enabled: bool) -> None:
        """设置 TX/RX 调试日志开关。

        Parameters
        ----------
        enabled : bool
            `True` 启用日志，`False` 关闭日志。
        """
        self._debug_enabled = enabled

    def open(self) -> None:
        """打开串口。"""
        self._transport.open()

    def close(self) -> None:
        """关闭串口。"""
        self._transport.close()

    def clear_input_buffer(self) -> None:
        """清空串口输入缓冲与内部组帧缓冲。"""
        self._transport.clear_input_buffer()
        self._rx_buffer.clear()

    def read_output_frame(self, timeout_s: float = 0.2) -> IMU760OutputFrame:
        """读取一帧输出协议报文。

        Parameters
        ----------
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760OutputFrame
            已完成校验的输出帧对象。

        Raises
        ------
        TimeoutError
            在超时时间内未组到有效帧。
        """
        frame = self._read_frame(timeout_s=timeout_s, command_frame=False)
        tid = _read_u16_le(frame[2:4])
        payload_length = frame[4]
        payload = bytes(frame[5 : 5 + payload_length])
        return IMU760OutputFrame(
            tid=tid,
            payload_length=payload_length,
            payload=payload,
            checksum_1=frame[-2],
            checksum_2=frame[-1],
        )

    def read_output_payload(self, timeout_s: float = 0.2) -> tuple[IMU760DecodedData, ...]:
        """读取并解码输出 payload。

        Parameters
        ----------
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        tuple[IMU760DecodedData, ...]
            逐数据包的结构化对象序列。
        """
        frame = self.read_output_frame(timeout_s=timeout_s)
        return decode_output_payload(frame.payload)

    def query(self, data_class: int, query_data: bytes = b"", timeout_s: float = 0.2) -> IMU760CommandFrame:
        """发送查询指令并等待返回。

        Parameters
        ----------
        data_class : int
            数据类编码。
        query_data : bytes, default=b""
            查询数据域。
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame
            设备返回的命令帧。
        """
        response = self.send_command(
            data_class=data_class,
            operator=IMU760Operator.QUERY,
            data=query_data,
            expect_response=True,
            timeout_s=timeout_s,
        )
        if response is None:
            raise RuntimeError("查询失败")
        return response

    def write_ram(self, data_class: int, data: bytes, timeout_s: float = 0.2) -> IMU760CommandFrame | None:
        """写入 RAM 配置。

        Parameters
        ----------
        data_class : int
            数据类编码。
        data : bytes
            写入数据域。
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame | None
            返回帧（若该指令有返回），否则 `None`。
        """
        return self.send_command(
            data_class=data_class,
            operator=IMU760Operator.WRITE_RAM,
            data=data,
            expect_response=(data_class != IMU760DataClass.BAUDRATE),
            timeout_s=timeout_s,
        )

    def write_flash(self, data_class: int, data: bytes, timeout_s: float = 0.2) -> IMU760CommandFrame | None:
        """写入 Flash 配置。

        Parameters
        ----------
        data_class : int
            数据类编码。
        data : bytes
            写入数据域。
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame | None
            返回帧（若该指令有返回），否则 `None`。
        """
        return self.send_command(
            data_class=data_class,
            operator=IMU760Operator.WRITE_FLASH,
            data=data,
            expect_response=(data_class != IMU760DataClass.BAUDRATE),
            timeout_s=timeout_s,
        )

    def send_command(
        self,
        data_class: int,
        operator: int | IMU760Operator,
        data: bytes = b"",
        expect_response: bool = True,
        timeout_s: float = 0.2,
    ) -> IMU760CommandFrame | None:
        """发送命令帧。

        Parameters
        ----------
        data_class : int
            数据类编码。
        operator : int | IMU760Operator
            操作符编码。
        data : bytes, default=b""
            数据域内容。
        expect_response : bool, default=True
            是否等待设备返回。
        timeout_s : float, default=0.2
            等待返回时超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame | None
            若等待返回，返回解析后的命令帧；否则返回 `None`。
        """
        frame = self.build_command_frame(data_class=data_class, operator=int(operator), data=data)
        if self._debug_enabled:
            logger.debug("IMU760 TX: {}", frame.hex(" ").upper())
        self._transport.write(frame)
        if not expect_response:
            return None
        raw_response = self._read_frame(timeout_s=timeout_s, command_frame=True)
        return self.parse_command_frame(raw_response)

    def set_output_rate(
        self, hz: int, save_to_flash: bool = False, timeout_s: float = 0.2
    ) -> IMU760CommandFrame | None:
        """设置输出频率。

        Parameters
        ----------
        hz : int
            输出频率（1/2/5/10/20/25/50/100/200/400）。
        save_to_flash : bool, default=False
            `True` 表示掉电保存。
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame | None
            返回帧（若该指令有返回），否则 `None`。
        """
        if hz not in self.OUTPUT_RATE_CODE_MAP:
            raise ValueError(f"不支持的输出频率：{hz}")
        code = self.OUTPUT_RATE_CODE_MAP[hz]
        writer = self.write_flash if save_to_flash else self.write_ram
        return writer(IMU760DataClass.OUTPUT_RATE, bytes([code]), timeout_s=timeout_s)

    def query_output_rate(self, timeout_s: float = 0.2) -> int:
        """查询输出频率。

        Parameters
        ----------
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        int
            输出频率（Hz）。
        """
        response = self.query(IMU760DataClass.OUTPUT_RATE, timeout_s=timeout_s)
        if len(response.data) != 1:
            raise ValueError(f"输出频率查询返回长度异常：{len(response.data)}")
        rate_code = response.data[0]
        for hz, code in self.OUTPUT_RATE_CODE_MAP.items():
            if code == rate_code:
                return hz
        raise ValueError(f"未知输出频率编码：0x{rate_code:02X}")

    def set_baudrate(self, baudrate: int, save_to_flash: bool = False, timeout_s: float = 0.2) -> None:
        """设置串口波特率。

        Parameters
        ----------
        baudrate : int
            波特率。
        save_to_flash : bool, default=False
            `True` 表示掉电保存。
        timeout_s : float, default=0.2
            超时时间，单位秒。
        """
        if baudrate not in self.BAUDRATE_CODE_MAP:
            raise ValueError(f"不支持的波特率：{baudrate}")
        code = self.BAUDRATE_CODE_MAP[baudrate]
        writer = self.write_flash if save_to_flash else self.write_ram
        writer(IMU760DataClass.BAUDRATE, bytes([code]), timeout_s=timeout_s)

    def query_baudrate(self, timeout_s: float = 0.2) -> int:
        """查询串口波特率。

        Parameters
        ----------
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        int
            波特率。
        """
        response = self.query(IMU760DataClass.BAUDRATE, timeout_s=timeout_s)
        if len(response.data) != 1:
            raise ValueError(f"波特率查询返回长度异常：{len(response.data)}")
        baudrate_code = response.data[0]
        for baudrate, code in self.BAUDRATE_CODE_MAP.items():
            if code == baudrate_code:
                return baudrate
        raise ValueError(f"未知波特率编码：0x{baudrate_code:02X}")

    def set_output_content_mask(
        self,
        mask: int,
        save_to_flash: bool = False,
        timeout_s: float = 0.2,
    ) -> IMU760CommandFrame | None:
        """设置输出内容掩码。

        Parameters
        ----------
        mask : int
            16bit 输出掩码。
        save_to_flash : bool, default=False
            `True` 表示掉电保存。
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame | None
            返回帧（若该指令有返回），否则 `None`。
        """
        data = mask.to_bytes(2, byteorder="little", signed=False)
        writer = self.write_flash if save_to_flash else self.write_ram
        return writer(IMU760DataClass.OUTPUT_CONTENT, data, timeout_s=timeout_s)

    def query_output_content_mask(self, timeout_s: float = 0.2) -> int:
        """查询输出内容掩码。

        Parameters
        ----------
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        int
            16bit 输出掩码。
        """
        response = self.query(IMU760DataClass.OUTPUT_CONTENT, timeout_s=timeout_s)
        if len(response.data) != 2:
            raise ValueError(f"输出内容查询返回长度异常：{len(response.data)}")
        return _read_u16_le(response.data)

    def set_algorithm_mode(
        self,
        mode: IMU760AlgorithmMode,
        save_to_flash: bool = False,
        timeout_s: float = 0.2,
    ) -> IMU760CommandFrame | None:
        """设置算法模式。

        Parameters
        ----------
        mode : IMU760AlgorithmMode
            算法模式枚举值。
        save_to_flash : bool, default=False
            `True` 表示掉电保存。
        timeout_s : float, default=0.2
            超时时间，单位秒。

        Returns
        -------
        IMU760CommandFrame | None
            返回帧（若该指令有返回），否则 `None`。
        """
        data = bytes([IMU760ModeSubtype.ALGO_MODE, int(mode)])
        writer = self.write_flash if save_to_flash else self.write_ram
        return writer(IMU760DataClass.FUNCTION_MODE, data, timeout_s=timeout_s)

    @staticmethod
    def build_command_frame(data_class: int, operator: int, data: bytes = b"") -> bytes:
        """构建交互协议命令帧。

        Parameters
        ----------
        data_class : int
            数据类编码。
        operator : int
            3bit 操作符。
        data : bytes, default=b""
            数据域。

        Returns
        -------
        bytes
            完整命令帧。
        """
        if not (0 <= data_class <= 0xFF):
            raise ValueError(f"data_class 超出范围：{data_class}")
        if not (0 <= operator <= 0x07):
            raise ValueError(f"operator 超出 3bit 范围：{operator}")
        if len(data) > 0x1FFF:
            raise ValueError(f"数据长度超出 13bit 上限：{len(data)}")

        op_and_len = (len(data) << 3) | operator
        body = bytes([data_class]) + op_and_len.to_bytes(2, byteorder="little", signed=False) + data
        ck1, ck2 = imu760_checksum(body)
        return FRAME_HEADER + body + bytes([ck1, ck2])

    @staticmethod
    def parse_command_frame(frame: bytes) -> IMU760CommandFrame:
        """解析并校验命令帧。

        Parameters
        ----------
        frame : bytes
            原始命令帧字节。

        Returns
        -------
        IMU760CommandFrame
            解析结果。
        """
        if len(frame) < 7:
            raise ValueError(f"命令帧长度不足：{len(frame)}")
        if frame[:2] != FRAME_HEADER:
            raise ValueError("命令帧头错误")

        data_class = frame[2]
        op_and_len = _read_u16_le(frame[3:5])
        operator = op_and_len & 0x07
        data_length = (op_and_len >> 3) & 0x1FFF
        data = frame[5 : 5 + data_length]
        ck1 = frame[5 + data_length]
        ck2 = frame[6 + data_length]

        expected_ck1, expected_ck2 = imu760_checksum(frame[2 : 5 + data_length])
        if ck1 != expected_ck1 or ck2 != expected_ck2:
            raise ValueError(
                f"命令帧校验失败：收到 ({ck1:#04x},{ck2:#04x}) 期望 ({expected_ck1:#04x},{expected_ck2:#04x})"
            )

        return IMU760CommandFrame(
            data_class=data_class,
            operator=operator,
            data_length=data_length,
            data=bytes(data),
            checksum_1=ck1,
            checksum_2=ck2,
        )

    def _read_frame(self, timeout_s: float, command_frame: bool) -> bytes:
        """按协议从输入流组装有效帧。

        Parameters
        ----------
        timeout_s : float
            超时时间，单位秒。
        command_frame : bool
            `True` 表示按交互协议帧解析，`False` 表示按输出协议帧解析。

        Returns
        -------
        bytes
            通过校验的完整帧。
        """
        deadline = monotonic() + timeout_s
        while True:
            frame = self._try_extract_frame(command_frame=command_frame)
            if frame is not None:
                if self._debug_enabled:
                    logger.debug("IMU760 RX: {}", frame.hex(" ").upper())
                return frame
            if monotonic() >= deadline:
                raise TimeoutError("读取 IMU760 帧超时。")
            chunk = self._transport.read(size=256)
            if not chunk:
                continue
            self._rx_buffer.extend(chunk)

    def _try_extract_frame(self, command_frame: bool) -> bytes | None:
        """尝试从内部缓冲提取一帧。

        Parameters
        ----------
        command_frame : bool
            `True` 按交互协议解析，`False` 按输出协议解析。

        Returns
        -------
        bytes | None
            成功时返回完整帧，否则返回 `None`。
        """
        if len(self._rx_buffer) < 7:
            return None

        while True:
            header_pos = self._rx_buffer.find(FRAME_HEADER)
            if header_pos < 0:
                self._rx_buffer.clear()
                return None
            if header_pos > 0:
                del self._rx_buffer[:header_pos]
            if len(self._rx_buffer) < 7:
                return None

            if command_frame:
                # 交互查询时串口里可能混入持续输出帧：
                # 优先尝试按命令帧解析；若当前帧是合法输出帧则丢弃并继续搜寻响应帧。
                cmd_data_length = (_read_u16_le(self._rx_buffer[3:5]) >> 3) & 0x1FFF
                cmd_total_len = 2 + 3 + cmd_data_length + 2

                if len(self._rx_buffer) >= cmd_total_len:
                    cmd_frame = bytes(self._rx_buffer[:cmd_total_len])
                    cmd_ck1, cmd_ck2 = imu760_checksum(cmd_frame[2 : 5 + cmd_data_length])
                    cmd_data_class = cmd_frame[2]
                    cmd_operator = _read_u16_le(cmd_frame[3:5]) & 0x07
                    if (
                        cmd_frame[-2] == cmd_ck1
                        and cmd_frame[-1] == cmd_ck2
                        and cmd_data_class in self._VALID_DATA_CLASS_VALUES
                        and cmd_operator in (0x00, 0x01, 0x02)
                    ):
                        del self._rx_buffer[:cmd_total_len]
                        return cmd_frame

                out_data_length = self._rx_buffer[4]
                out_total_len = 2 + 3 + out_data_length + 2
                if len(self._rx_buffer) < out_total_len:
                    return None

                out_frame = bytes(self._rx_buffer[:out_total_len])
                out_ck1, out_ck2 = imu760_checksum(out_frame[2 : 5 + out_data_length])
                if out_frame[-2] == out_ck1 and out_frame[-1] == out_ck2:
                    del self._rx_buffer[:out_total_len]
                    if self._debug_enabled:
                        logger.debug("IMU760 RX(SKIP-OUTPUT): {}", out_frame.hex(" ").upper())
                    continue

                del self._rx_buffer[:2]
                continue

            data_length = self._rx_buffer[4]
            total_len = 2 + 3 + data_length + 2
            if len(self._rx_buffer) < total_len:
                return None

            frame = bytes(self._rx_buffer[:total_len])
            del self._rx_buffer[:total_len]

            ck1, ck2 = imu760_checksum(frame[2 : 5 + data_length])
            if frame[-2] != ck1 or frame[-1] != ck2:
                continue
            return frame


# endregion
