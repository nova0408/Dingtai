"""Frame protocol - matches C++ frame_protocol.hpp"""
import struct

HEADER_SIZE = 48
HEADER_FORMAT = '<4B4B3I4IQI'  # little-endian, 48 bytes
MAGIC = b'ZCAM'

DEPTH_FORMAT_NONE = 0
DEPTH_FORMAT_LZ4_UINT16 = 1
DEPTH_FORMAT_RAW_UINT16 = 2

CAMERA_NAMES = {4: 'HEAD', 5: 'CHEST', 6: 'LEFT', 7: 'RIGHT'}


def parse_header(data: bytes):
    """Parse FrameHeader from first 48 bytes. Returns dict."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Header too short: {len(data)}")
    if data[:4] != MAGIC:
        raise ValueError(f"Invalid magic: {data[:4]}")
    vals = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    return {
        'magic': data[:4],
        'version': vals[4],
        'camera_index': vals[5],
        'color_format': vals[6],
        'depth_format': vals[7],
        'color_width': vals[8],
        'color_height': vals[9],
        'color_data_size': vals[10],
        'depth_width': vals[11],
        'depth_height': vals[12],
        'depth_data_size': vals[13],
        'depth_original_size': vals[14],
        'timestamp_us': vals[15],
        'sequence': vals[16],
    }
