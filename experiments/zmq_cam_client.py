#!/usr/bin/env python3
"""
ZMQ 相机客户端 - 订阅4路相机2D和深度数据，统计FPS和延迟
"""
import sys
import time
import json
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zmq
import numpy as np
import cv2
import lz4.block

from frame_protocol import parse_header, HEADER_SIZE, DEPTH_FORMAT_LZ4_UINT16, CAMERA_NAMES

DEFAULT_HOST = "192.168.10.37"
# DEFAULT_HOST = "10.7.5.150"
PORTS = [5560, 5561, 5562, 5563]  # HEAD, CHEST, LEFT, RIGHT
CONTROL_PORT = 5570
STATS_WINDOW = 2.0  # seconds for FPS/latency window
PREVIEW_CAMERAS = [4, 5]  # HEAD, CHEST
PREVIEW_MAX_WIDTH = 640
PREVIEW_HTTP_PORT = 8765


class CameraStats:
    def __init__(self):
        self.color_fps = 0.0
        self.depth_fps = 0.0
        self.color_decode_ms = 0.0
        self.depth_decode_ms = 0.0
        self.color_latency_ms = 0.0   # 端到端延迟: 数据生成 -> 接收
        self.depth_latency_ms = 0.0
        self.color_recv_times = deque(maxlen=128)
        self.depth_recv_times = deque(maxlen=128)
        self.color_decode_times = deque(maxlen=64)
        self.depth_decode_times = deque(maxlen=64)
        self.color_latency_samples = deque(maxlen=64)
        self.depth_latency_samples = deque(maxlen=64)
        self.last_color_np = None
        self.last_depth_np = None
        # 带宽统计
        self.total_bytes = 0
        self.frame_count = 0
        self.last_color_size = 0
        self.last_depth_size = 0
        self.last_resolution = (0, 0)
        self.last_depth_resolution = (0, 0)


def get_intrinsics(host: str, cameras: list = None):
    """获取4路相机内参并打印"""
    if cameras is None:
        cameras = ['HEAD', 'CHEST', 'LEFT', 'RIGHT']
    ctx = zmq.Context()
    print("\n--- 获取相机内参 ---")
    for cam in cameras:
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 2000)
        req.connect(f'tcp://{host}:{CONTROL_PORT}')
        cmd = json.dumps({'cmd': 'get_intrinsics', 'camera': cam})
        try:
            req.send_string(cmd)
            resp = req.recv_string()
            j = json.loads(resp)
            if j.get('success') and 'data' in j:
                d = j['data']
                fx = d.get('fx', 0)
                fy = d.get('fy', 0)
                cx = d.get('cx', 0)
                cy = d.get('cy', 0)
                dist = d.get('dist', [])
                print(f"\n  {cam}:")
                print(f"    fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                print(f"    dist(k1-k6,p1,p2)={dist}")
            else:
                print(f"  {cam}: 获取失败 - {resp}")
        except (zmq.Again, zmq.ZMQError, json.JSONDecodeError) as e:
            print(f"  {cam}: 请求失败 ({e})")
        finally:
            req.close()
    ctx.destroy()
    print()


def enable_depth(host: str, cameras: list = None):
    """通过控制通道开启深度流"""
    if cameras is None:
        cameras = ['HEAD', 'CHEST', 'LEFT', 'RIGHT']
    ctx = zmq.Context()
    for cam in cameras:
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 2000)
        req.connect(f'tcp://{host}:{CONTROL_PORT}')
        cmd = json.dumps({'cmd': 'set_depth_enabled', 'camera': cam, 'params': {'enable': True}})
        try:
            req.send_string(cmd)
            resp = req.recv_string()
            print(f"  {cam} depth_enable: {resp}")
        except (zmq.Again, zmq.ZMQError) as e:
            print(f"  {cam} depth_enable: timeout or error ({e})")
        req.close()
    ctx.destroy()


def _resize_to_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def _annotate_frame(frame: np.ndarray, title: str, fps: float) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _depth_to_vis(depth_np: np.ndarray) -> np.ndarray:
    valid = depth_np.astype(np.float32)
    mask = valid > 0
    if not np.any(mask):
        return np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
    lo, hi = np.percentile(valid[mask], (5, 95))
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((valid - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    norm[~mask] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def _stack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    if not tiles:
        placeholder = np.zeros((360, PREVIEW_MAX_WIDTH, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for frames...", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        return placeholder
    max_h = max(t.shape[0] for t in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < max_h:
            pad = np.zeros((max_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    return cv2.hconcat(padded)


def build_preview_frame(stats: dict, show_depth: bool = False) -> np.ndarray:
    tiles = []
    for cam_idx in PREVIEW_CAMERAS:
        st = stats[cam_idx]
        name = CAMERA_NAMES.get(cam_idx, '?')
        if st.last_color_np is not None:
            color = _resize_to_width(
                _annotate_frame(st.last_color_np, f"{name} RGB", st.color_fps),
                PREVIEW_MAX_WIDTH,
            )
            tiles.append(color)
        if show_depth and st.last_depth_np is not None:
            depth = _resize_to_width(
                _annotate_frame(_depth_to_vis(st.last_depth_np), f"{name} Depth", st.depth_fps),
                PREVIEW_MAX_WIDTH,
            )
            tiles.append(depth)
    return _stack_tiles(tiles)


class PreviewSession:
    """OpenCV 窗口预览；若无 GUI 则自动回退到浏览器 MJPEG。"""

    @staticmethod
    def opencv_gui_available() -> bool:
        try:
            cv2.namedWindow('__opencv_gui_test__', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('__opencv_gui_test__')
            return True
        except cv2.error:
            return False

    def __init__(self, show_depth: bool, http_port: int = PREVIEW_HTTP_PORT):
        self.show_depth = show_depth
        self.http_port = http_port
        self.mode = None
        self._jpeg = None
        self._lock = threading.Lock()
        self._running = True
        self._server = None

    def start(self):
        if self.opencv_gui_available():
            self.mode = 'opencv'
            cv2.namedWindow('ZMQ Camera Preview (q/ESC quit)', cv2.WINDOW_NORMAL)
            print('OpenCV 窗口预览 (q / ESC 退出)\n')
        else:
            self.mode = 'web'
            self._start_web_server()
            url = f'http://127.0.0.1:{self.http_port}/'
            print('当前 OpenCV 无 GUI 支持 (多为 opencv-python-headless)。')
            print(f'已改用浏览器预览: {url}')
            print('在浏览器中查看实时画面，终端 Ctrl+C 退出\n')
            webbrowser.open(url)

    def _start_web_server(self):
        session = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    html = (
                        '<!DOCTYPE html><html><head><meta charset="utf-8">'
                        '<title>ZMQ Camera Preview</title>'
                        '<style>body{margin:0;background:#111;color:#ccc;text-align:center;'
                        'font-family:sans-serif}img{max-width:98vw;max-height:90vh}</style>'
                        '</head><body><h3>HEAD / CHEST 实时预览</h3>'
                        '<img src="/stream"><p>终端 Ctrl+C 停止</p></body></html>'
                    )
                    self.wfile.write(html.encode('utf-8'))
                    return
                if self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.end_headers()
                    while session._running:
                        with session._lock:
                            jpeg = session._jpeg
                        if jpeg:
                            try:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                                self.wfile.write(jpeg)
                                self.wfile.write(b'\r\n')
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                return
                        time.sleep(0.04)
                    return
                self.send_response(404)
                self.end_headers()

        self._server = ThreadingHTTPServer(('127.0.0.1', self.http_port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def update(self, stats: dict) -> bool:
        frame = build_preview_frame(stats, show_depth=self.show_depth)
        ok, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return False
        if self.mode == 'web':
            with self._lock:
                self._jpeg = jpeg.tobytes()
            return False
        cv2.imshow('ZMQ Camera Preview (q/ESC quit)', frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord('q'), 27)

    def close(self):
        self._running = False
        if self._server:
            self._server.shutdown()


def run_client(
    host: str = DEFAULT_HOST,
    duration: float = 60.0,
    enable_depth_stream: bool = True,
    show_preview: bool = False,
    preview_port: int = PREVIEW_HTTP_PORT,
):
    ctx = zmq.Context()
    sockets = []
    for i, port in enumerate(PORTS):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 1)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b'')
        addr = f'tcp://{host}:{port}'
        s.connect(addr)
        sockets.append(s)
        print(f"[{CAMERA_NAMES.get(4+i, '?')}] connect {addr}")

    poller = zmq.Poller()
    for s in sockets:
        poller.register(s, zmq.POLLIN)

    # 1. 先获取4路相机内参
    get_intrinsics(host)

    if enable_depth_stream:
        print("开启深度流...")
        enable_depth(host)
        time.sleep(1)

    stats = {4 + i: CameraStats() for i in range(4)}
    start_time = time.perf_counter()
    poll_timeout = 100  # ms

    preview = None
    if show_preview:
        print('\n--- 预览模式: 接收 HEAD / CHEST 实时画面 ---\n')
        preview = PreviewSession(show_depth=enable_depth_stream, http_port=preview_port)
        preview.start()
    else:
        print(f'\n--- 开始接收 {duration}s，统计 FPS / 端到端延迟 / 解码耗时 ---\n')

    user_quit = False
    while (duration <= 0 or (time.perf_counter() - start_time) < duration) and not user_quit:
        for s in poller.poll(poll_timeout):
            sock = s[0]
            idx = sockets.index(sock)
            cam_idx = 4 + idx

            try:
                msg = sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue

            recv_time = time.perf_counter()
            if len(msg) < HEADER_SIZE:
                continue

            try:
                hdr = parse_header(msg)
            except ValueError as e:
                print(f"[{CAMERA_NAMES.get(cam_idx, '?')}] parse error: {e}")
                continue

            st = stats[cam_idx]
            st.total_bytes += len(msg)
            st.frame_count += 1
            st.last_color_size = hdr['color_data_size']
            st.last_depth_size = hdr['depth_data_size']
            st.last_resolution = (hdr['color_width'], hdr['color_height'])
            st.last_depth_resolution = (hdr['depth_width'], hdr['depth_height'])
            # 端到端延迟: recv_time(us) - timestamp_us (需同一单调时钟，本机时近似)
            recv_time_us = recv_time * 1e6
            ts_us = hdr['timestamp_us']
            latency_ms = (recv_time_us - ts_us) / 1000.0
            if latency_ms < 0 or latency_ms > 5000:  # 异常值过滤
                latency_ms = None

            # Color (JPEG)
            color_size = hdr['color_data_size']
            color_offset = HEADER_SIZE
            if color_size > 0 and len(msg) >= color_offset + color_size:
                t0 = time.perf_counter()
                jpeg_buf = np.frombuffer(msg, dtype=np.uint8, offset=color_offset, count=color_size)
                color_np = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
                if color_np is not None:
                    st.last_color_np = color_np
                    st.color_recv_times.append(recv_time)
                    st.color_decode_times.append((time.perf_counter() - t0) * 1000)
                    if latency_ms is not None:
                        st.color_latency_samples.append(latency_ms)

            # Depth (LZ4)
            depth_size = hdr['depth_data_size']
            depth_orig = hdr['depth_original_size']
            depth_offset = HEADER_SIZE + color_size
            if hdr['depth_format'] == DEPTH_FORMAT_LZ4_UINT16 and depth_size > 0 and depth_orig > 0:
                if len(msg) >= depth_offset + depth_size:
                    t0 = time.perf_counter()
                    compressed = bytes(msg[depth_offset:depth_offset + depth_size])
                    decompressed = lz4.block.decompress(compressed, uncompressed_size=depth_orig)
                    depth_np = np.frombuffer(decompressed, dtype=np.uint16)
                    depth_np = depth_np.reshape((hdr['depth_height'], hdr['depth_width']))
                    st.last_depth_np = depth_np
                    st.depth_recv_times.append(recv_time)
                    st.depth_decode_times.append((time.perf_counter() - t0) * 1000)
                    if latency_ms is not None:
                        st.depth_latency_samples.append(latency_ms)

        # Update FPS stats (use sliding window)
        now = time.perf_counter()
        for cam_idx, st in stats.items():
            # Prune old color recv times
            while st.color_recv_times and st.color_recv_times[0] < now - STATS_WINDOW:
                st.color_recv_times.popleft()
            if len(st.color_recv_times) >= 2:
                st.color_fps = len(st.color_recv_times) / STATS_WINDOW
            # Prune old depth recv times
            while st.depth_recv_times and st.depth_recv_times[0] < now - STATS_WINDOW:
                st.depth_recv_times.popleft()
            if len(st.depth_recv_times) >= 2:
                st.depth_fps = len(st.depth_recv_times) / STATS_WINDOW
            # Decode time (ms)
            if st.color_decode_times:
                st.color_decode_ms = np.mean(list(st.color_decode_times)[-20:])
            if st.depth_decode_times:
                st.depth_decode_ms = np.mean(list(st.depth_decode_times)[-20:])
            # 平均端到端延迟 (ms)
            if st.color_latency_samples:
                st.color_latency_ms = np.mean(list(st.color_latency_samples)[-20:])
            if st.depth_latency_samples:
                st.depth_latency_ms = np.mean(list(st.depth_latency_samples)[-20:])

        if preview:
            user_quit = preview.update(stats)

    if preview:
        preview.close()

    elapsed = time.perf_counter() - start_time

    # Print report
    print("\n" + "=" * 80)
    print("统计结果 (FPS / 端到端延迟 / 解码耗时 / 帧间隔)")
    print("=" * 80)
    for cam_idx in [4, 5, 6, 7]:
        st = stats[cam_idx]
        name = CAMERA_NAMES.get(cam_idx, '?')
        c_interval = 1000.0 / st.color_fps if st.color_fps > 0 else 0
        d_interval = 1000.0 / st.depth_fps if st.depth_fps > 0 else 0
        print(f"\n{cam_idx}. {name}:")
        print(f"  2D:   FPS={st.color_fps:.1f}  |  端到端延迟={st.color_latency_ms:.1f} ms  |  解码={st.color_decode_ms:.2f} ms  |  帧间隔={c_interval:.0f} ms")
        print(f"  深度: FPS={st.depth_fps:.1f}  |  端到端延迟={st.depth_latency_ms:.1f} ms  |  解码={st.depth_decode_ms:.2f} ms  |  帧间隔={d_interval:.0f} ms")

    print("\n" + "=" * 80)
    print(f"带宽统计 (实测 {elapsed:.1f}s，按 ZMQ 消息总字节)")
    print("=" * 80)
    active_cams = []
    per_cam_at_15 = []
    for cam_idx in [4, 5, 6, 7]:
        st = stats[cam_idx]
        name = CAMERA_NAMES.get(cam_idx, '?')
        if st.frame_count == 0:
            print(f"\n{cam_idx}. {name}: 无数据")
            continue
        active_cams.append(cam_idx)
        avg_frame = st.total_bytes / st.frame_count
        actual_mbps = st.total_bytes * 8 / elapsed / 1e6
        actual_mbs = st.total_bytes / elapsed / 1e6
        at_15_mbps = avg_frame * 15 * 8 / 1e6
        per_cam_at_15.append(at_15_mbps)
        cw, ch = st.last_resolution
        dw, dh = st.last_depth_resolution
        print(f"\n{cam_idx}. {name}:")
        print(f"  分辨率: 2D={cw}x{ch}  深度={dw}x{dh}")
        print(f"  帧数={st.frame_count}  平均帧={avg_frame/1024:.1f} KB  (JPEG={st.last_color_size/1024:.1f} KB + 深度={st.last_depth_size/1024:.1f} KB + header={HEADER_SIZE} B)")
        print(f"  实测带宽: {actual_mbs:.2f} MB/s  ({actual_mbps:.1f} Mbps)")
        print(f"  外推 @15 FPS: {at_15_mbps:.1f} Mbps/路")

    if per_cam_at_15:
        avg_per_cam = np.mean(per_cam_at_15)
        total_3 = avg_per_cam * 3
        total_4 = avg_per_cam * 4
        print(f"\n--- 外推汇总 (活跃路平均 {avg_per_cam:.1f} Mbps/路 @15 FPS) ---")
        print(f"  3 路相机: {total_3:.1f} Mbps  ({total_3/8:.2f} MB/s)")
        print(f"  4 路相机: {total_4:.1f} Mbps  ({total_4/8:.2f} MB/s)")
        if len(per_cam_at_15) > 1:
            per_cam_vals = [stats[i].total_bytes / stats[i].frame_count * 15 * 8 / 1e6 for i in active_cams]
            min_mbps = min(per_cam_vals)
            max_mbps = max(per_cam_vals)
            print(f"  各路范围 @15 FPS: {min_mbps:.1f} ~ {max_mbps:.1f} Mbps/路")
            print(f"  3 路 (按最小~最大): {min_mbps*3:.1f} ~ {max_mbps*3:.1f} Mbps")
            print(f"  4 路 (按最小~最大): {min_mbps*4:.1f} ~ {max_mbps*4:.1f} Mbps")
        total_actual = sum(stats[i].total_bytes for i in active_cams)
        print(f"  实测合计 ({len(active_cams)} 路): {total_actual/elapsed/1e6:.2f} MB/s  ({total_actual*8/elapsed/1e6:.1f} Mbps)")
    print("\n" + "=" * 80)

    ctx.destroy()


if __name__ == '__main__':
    preview = '--preview' in sys.argv
    no_depth = '--no-depth' in sys.argv
    preview_port = PREVIEW_HTTP_PORT
    positional = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--preview-port':
            if i + 1 < len(argv):
                preview_port = int(argv[i + 1])
                i += 2
                continue
        elif arg.startswith('--'):
            i += 1
            continue
        positional.append(arg)
        i += 1
    host = positional[0] if len(positional) > 0 else DEFAULT_HOST
    if preview and len(positional) < 2:
        duration = 0.0
    else:
        duration = float(positional[1]) if len(positional) > 1 else 10.0
    run_client(
        host=host,
        duration=duration,
        enable_depth_stream=not no_depth,
        show_preview=preview,
        preview_port=preview_port,
    )
