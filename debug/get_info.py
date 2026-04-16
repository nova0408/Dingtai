# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#  Modified for Chinese Table Output
# ******************************************************************************

from pyorbbecsdk import *


def print_separator(char="-", length=110):
    print(char * length)


def main():
    try:
        context = Context()
        device_list = context.query_devices()
        device_count = device_list.get_count()

        if device_count < 1:
            print("\n[错误] 未找到设备，请检查连接。")
            return

        print(f"\n>>> 发现 {device_count} 个设备 <<<\n")

        for i in range(device_count):
            device = device_list[i]
            device_info = device.get_device_info()

            # 打印设备基本信息
            print_separator("=")
            print(f"设备索引：{i}")
            print(f"设备名称：{device_info.get_name()}")
            print(f"产品 PID: {device_info.get_pid()}")
            print(f"序列号  : {device_info.get_serial_number()}")
            print(f"连接类型：{device_info.get_connection_type()}")
            print_separator("=")

            # 获取传感器列表
            sensor_list = device.get_sensor_list()
            for j in range(sensor_list.get_count()):
                sensor = sensor_list.get_sensor_by_index(j)
                sensor_type = sensor.get_type()

                print(f"\n[传感器 #{j}] 类型：{sensor_type}")

                # 表格表头
                header = f"{'索引':<6} | {'格式':<15} | {'分辨率 (宽 x 高)':<15} | {'FPS':<6} | {'类型':<20}"
                print(header)
                print("-" * len(header))

                try:
                    profiles = sensor.get_stream_profile_list()
                    for k in range(profiles.get_count()):
                        profile = profiles.get_stream_profile_by_index(k)

                        # 检查是否为视频流配置
                        if hasattr(profile, "get_width"):
                            fmt = str(profile.get_format())
                            width = profile.get_width()
                            height = profile.get_height()
                            fps = profile.get_fps()
                            res = f"{width}x{height}"
                            p_type = type(profile).__name__

                            print(f"{k:<8} | {fmt:<15} | {res:<15} | {fps:<6} | {p_type:<20}")
                        else:
                            print(f"{k:<8} | 非视频流配置类型：{type(profile).__name__}")
                except Exception as e:
                    print(f"无法读取配置列表：{e}")

            print("\n")  # 设备间空行

    except Exception as e:
        print(f"\n[异常] 运行时发生错误：{e}")


if __name__ == "__main__":
    main()
