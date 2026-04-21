# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from serial import Serial


@dataclass
class TTLSerialConfig:
    """TTL 串口配置。"""

    port: str
    baudrate: int = 460_800
    timeout_s: float = 0.1
    write_timeout_s: float = 0.1
    bytesize: int = 8
    parity: str = "N"
    stopbits: float = 1


class TTLSerialTransport:
    """TTL 串口底层通信类。"""

    def __init__(self, config: TTLSerialConfig, serial_impl: Serial | None = None) -> None:
        self._config = config
        self._serial = serial_impl or self._build_serial_from_config(config)

    @staticmethod
    def _build_serial_from_config(config: TTLSerialConfig) -> Serial:
        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("未安装 pyserial，请先执行 `pip install pyserial`。") from exc

        return serial.Serial(
            port=config.port,
            baudrate=config.baudrate,
            timeout=config.timeout_s,
            write_timeout=config.write_timeout_s,
            bytesize=config.bytesize,
            parity=config.parity,
            stopbits=config.stopbits,
        )

    @property
    def config(self) -> TTLSerialConfig:
        return self._config

    @property
    def is_open(self) -> bool:
        return bool(self._serial.is_open)

    @property
    def bytes_available(self) -> int:
        return int(self._serial.in_waiting)

    def open(self) -> None:
        if not self.is_open:
            self._serial.open()

    def close(self) -> None:
        if self.is_open:
            self._serial.close()

    def clear_input_buffer(self) -> None:
        self._serial.reset_input_buffer()

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        return int(self._serial.write(data))

    def read(self, size: int = 1) -> bytes:
        if size <= 0:
            return b""
        return bytes(self._serial.read(size))

    def read_exact(self, size: int, timeout_s: float | None = None) -> bytes:
        if size < 0:
            raise ValueError(f"size 不能小于 0，收到 {size}")
        if size == 0:
            return b""

        buffer = bytearray()
        deadline = None if timeout_s is None else monotonic() + timeout_s

        while len(buffer) < size:
            if deadline is not None and monotonic() >= deadline:
                break
            chunk = self._serial.read(size - len(buffer))
            if not chunk:
                continue
            buffer.extend(chunk)

        return bytes(buffer)
