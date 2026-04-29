# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#  Modified for Chinese Table Output
# ******************************************************************************

from pathlib import Path
from typing import Any

from pyorbbecsdk import *

OUTPUT_FILE = Path("orbbec_device_profiles.md")


def md_escape(value: Any) -> str:
    """避免 Markdown 表格被特殊字符破坏。"""
    text = str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def get_attr_value(obj: Any, name: str, default: str = "-") -> Any:
    """兼容 pybind 对象字段访问。"""
    return getattr(obj, name, default)


def format_intrinsic(intrinsic: Any) -> str:
    """
    常见字段：
    fx, fy, cx, cy, width, height
    """
    if intrinsic is None:
        return "-"

    fx = get_attr_value(intrinsic, "fx")
    fy = get_attr_value(intrinsic, "fy")
    cx = get_attr_value(intrinsic, "cx")
    cy = get_attr_value(intrinsic, "cy")
    width = get_attr_value(intrinsic, "width")
    height = get_attr_value(intrinsic, "height")

    # 用 `` 包裹，防止逗号/等号破坏表格列对齐
    return f"`fx={fx} fy={fy} cx={cx} cy={cy} " f"w={width} h={height}`"


def format_distortion(distortion: Any) -> str:
    """
    常见字段：
    k1, k2, k3, k4, k5, k6, p1, p2
    """
    if distortion is None:
        return "-"

    names = ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"]
    parts = [f"{name}={get_attr_value(distortion, name)}" for name in names]
    # 用 `` 包裹，防止内容破坏表格
    return "`" + " ".join(parts) + "`"


def append_profile_table_header(lines: list[str]) -> None:
    lines.append("| 索引 | 格式 | 分辨率 | FPS | Profile 类型 | 内参 | 畸变参数 |")
    lines.append("| ---: | :--- | :--- | ---: | :--- | :--- | :--- |")


def main() -> None:
    lines: list[str] = []
    lines.append("# Orbbec 设备信息与 Stream Profile 列表")
    lines.append("")

    try:
        context = Context()
        device_list = context.query_devices()
        device_count = device_list.get_count()

        if device_count < 1:
            lines.append("> **[错误]** 未找到设备，请检查连接。")
            OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
            print(f"已生成：{OUTPUT_FILE.resolve()}")
            return

        lines.append(f"> 发现 **{device_count}** 个设备")
        lines.append("")

        for i in range(device_count):
            device = device_list[i]
            device_info = device.get_device_info()

            lines.append(f"## 设备 {i}")
            lines.append("")
            lines.append("| 项目 | 值 |")
            lines.append("| :--- | :--- |")
            lines.append(f"| 设备索引 | {i} |")
            lines.append(f"| 设备名称 | {md_escape(device_info.get_name())} |")
            lines.append(f"| 产品 PID | {md_escape(device_info.get_pid())} |")
            lines.append(f"| 序列号 | {md_escape(device_info.get_serial_number())} |")
            lines.append(f"| 连接类型 | {md_escape(device_info.get_connection_type())} |")
            lines.append("")

            sensor_list = device.get_sensor_list()

            for j in range(sensor_list.get_count()):
                sensor = sensor_list.get_sensor_by_index(j)
                sensor_type = sensor.get_type()

                lines.append(f"### 传感器 {j} — `{md_escape(sensor_type)}`")
                lines.append("")
                append_profile_table_header(lines)

                try:
                    profiles = sensor.get_stream_profile_list()

                    for k in range(profiles.get_count()):
                        profile = profiles.get_stream_profile_by_index(k)
                        profile_type = type(profile).__name__

                        if not hasattr(profile, "get_width"):
                            lines.append(f"| {k} | - | - | - | {md_escape(profile_type)} | - | - |")
                            continue

                        fmt = profile.get_format()
                        width = profile.get_width()
                        height = profile.get_height()
                        fps = profile.get_fps()
                        resolution = f"{width}×{height}"

                        intrinsic_text = "-"
                        distortion_text = "-"

                        try:
                            intrinsic = profile.get_intrinsic()
                            intrinsic_text = format_intrinsic(intrinsic)
                        except Exception as e:
                            intrinsic_text = f"读取失败：{md_escape(e)}"

                        try:
                            distortion = profile.get_distortion()
                            distortion_text = format_distortion(distortion)
                        except Exception as e:
                            distortion_text = f"读取失败：{md_escape(e)}"

                        lines.append(
                            f"| {k}"
                            f" | {md_escape(fmt)}"
                            f" | {resolution}"
                            f" | {fps}"
                            f" | {md_escape(profile_type)}"
                            f" | {intrinsic_text}"
                            f" | {distortion_text} |"
                        )

                except Exception as e:
                    lines.append(f"| - | - | - | - | 读取配置列表失败 | {md_escape(e)} | - |")

                lines.append("")

            lines.append("")

        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"已生成：{OUTPUT_FILE.resolve()}")

    except Exception as e:
        lines.append(f"> **[异常]** 运行时发生错误：{md_escape(e)}")
        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"运行失败，但已写入错误信息：{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
