import cv2
import numpy as np
from pyorbbecsdk import *

# 1. Prerequisites (已安装 pyorbbecsdk)

# 2. Create pipeline
pipeline = Pipeline()

# 3. Start pipeline

pipeline.start()
prev_depth_norm = None
try:
    while True:
        # 4. Wait for frames
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        # 5. Get frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # --- 6. Render frames using OpenCV ---

        # 渲染彩色图
        if color_frame is not None:
            # 1. 获取原始数据
            raw_data = np.asanyarray(color_frame.get_data())

            # 2. 根据每像素 2 字节的特性，先 reshape 成 (height, width, 2)
            # 这里的 size 898880 正好符合 530 * 848 * 2
            color_yuv = raw_data.reshape((color_frame.get_height(), color_frame.get_width(), 2))

            # 3. 将 YUV (YUYV/YUY2) 转换为 BGR
            # 如果你的图像看起来颜色怪异，尝试改为 cv2.COLOR_YUV2BGR_I420 或其他 YUV 变体
            color_image = cv2.cvtColor(color_yuv, cv2.COLOR_YUV2BGR_YUYV)

            cv2.imshow("Color Stream", color_image)

        # 渲染深度图
        if depth_frame is not None:
            depth_data = np.asanyarray(depth_frame.get_data()).view(np.uint16)
            h, w = depth_frame.get_height(), depth_frame.get_width()
            depth_image = depth_data.reshape((h, w))

            # --- 高分辨率细节增强处理 ---

            # A. 针对 16 位深度图进行双边滤波 (边缘保留滤波)
            # d=5: 过滤直径；75, 75: 颜色空间和坐标空间的标准差
            # 相比 medianBlur，它能让物体边缘非常锋利
            depth_filtered = cv2.bilateralFilter(depth_image.astype(np.float32), 5, 75, 75)

            # B. 限制深度范围并归一化 (聚焦能提升细节感)
            min_dist, max_dist = 400, 2500  # 稍微缩短距离，细节会更丰富
            depth_clipped = np.clip(depth_filtered, min_dist, max_dist)
            depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # C. 图像锐化 (Unsharp Masking)
            # 通过拉高高频信号，让深度的“纹理”跳出来
            gaussian_blur = cv2.GaussianBlur(depth_norm, (7, 7), 2)
            depth_sharp = cv2.addWeighted(depth_norm, 1.5, gaussian_blur, -0.5, 0)

            # D. 伪彩色映射 (推荐 COLORMAP_TWILIGHT_SHIFTED，对比度极高且细节清晰)
            depth_view = cv2.applyColorMap(depth_sharp, cv2.COLORMAP_TWILIGHT_SHIFTED)

            # 处理无效区域
            depth_view[depth_image == 0] = 0

            # E. (可选) 窗口放大显示
            # 如果你的屏幕分辨率很高，可以把窗口插值放大一倍看看
            # depth_view = cv2.resize(depth_view, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

            cv2.imshow("High Resolution Detail Depth", depth_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # 7. Stop pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
