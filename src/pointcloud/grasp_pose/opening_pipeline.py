from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

from .types import OpeningDetection, TrayMaskResult


class OpeningDetectionPipeline:
    """开口检测与开口相关掩码流程。"""

    def __init__(self) -> None:
        self._near_grow_max_pixels = 26000
        self._near_grow_local_diff = 14
        self._near_grow_global_diff = 30

    def estimate_from_tray(
        self, rgb_bgr: np.ndarray, tray_mask: np.ndarray, tray_detect_ok: bool, hp_gray: np.ndarray, hp_edge: np.ndarray
    ) -> tuple[TrayMaskResult, OpeningDetection]:
        opening = self.detect_opening(rgb_bgr, tray_mask, hp_gray)
        near_plane_mask, no_hole_mask = self.compute_mask_pipeline(tray_mask, tray_detect_ok, opening, hp_gray, hp_edge)
        top_quad = self.fit_rotated_quad(no_hole_mask)
        return (
            TrayMaskResult(
                tray_mask=np.asarray(tray_mask, dtype=np.uint8),
                tray_detect_ok=bool(tray_detect_ok),
                near_plane_mask=near_plane_mask,
                no_hole_mask=no_hole_mask,
                top_quad_uv=top_quad,
            ),
            opening,
        )

    def detect_opening(self, rgb_bgr: np.ndarray, tray_mask: np.ndarray, hp_gray: np.ndarray) -> OpeningDetection:
        return self._detect_rect_opening_auto(rgb_bgr, tray_mask, hp_gray)

    def compute_mask_pipeline(
        self,
        tray_mask: np.ndarray,
        tray_detect_ok: bool,
        opening: OpeningDetection,
        hp_gray: np.ndarray,
        hp_edge: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        near_plane_mask = self._build_near_dark_plane_mask(tray_mask, opening, hp_gray, hp_edge) if tray_detect_ok else None
        no_hole_mask = self._build_no_hole_top_plane_mask(tray_mask, opening, near_plane_mask, hp_gray, hp_edge)
        return self._enforce_disjoint_region_masks(near_plane_mask, no_hole_mask)

    def filter_opening_local_points(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        opening: OpeningDetection,
        img_w: int,
        img_h: int,
        uv: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        local_roi = self._build_opening_local_roi_mask((img_h, img_w), opening)
        uv_ok = valid & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
        mask = np.zeros_like(valid)
        idx = np.where(uv_ok)[0]
        if idx.size > 0:
            u = uv[idx, 0]
            v = uv[idx, 1]
            mask[idx] = local_roi[v, u] > 0
        if np.count_nonzero(mask) > 30:
            inten = np.max((rgb[mask] * 255.0).astype(np.float32), axis=1)
            thr = float(np.clip(np.percentile(inten, 78), 70, 185))
            sel = np.where(mask)[0]
            keep = inten <= thr
            mask2 = np.zeros_like(mask)
            mask2[sel[keep]] = True
            mask = mask2
        if np.count_nonzero(mask) >= 80:
            zz = xyz[mask, 2]
            z_med = float(np.median(zz))
            z_abs = np.abs(xyz[:, 2] - z_med)
            mask &= z_abs <= 22.0
        return xyz[mask]

    @staticmethod
    def estimate_top_plane_normal(
        xyz: np.ndarray, mask: np.ndarray | None, uv: np.ndarray, valid: np.ndarray
    ) -> np.ndarray | None:
        if mask is None:
            return None
        m = np.asarray(mask) > 0
        idx = np.where(valid)[0]
        if idx.size < 120:
            return None
        u = uv[idx, 0]
        v = uv[idx, 1]
        sel = idx[m[v, u]]
        if sel.size < 120:
            return None
        pts = np.asarray(xyz[sel], dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 100:
            return None
        if pts.shape[0] >= 180:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
            try:
                _, inliers = pcd.segment_plane(distance_threshold=1.8, ransac_n=3, num_iterations=800)
                if len(inliers) >= 100:
                    pts = np.asarray(pts[np.asarray(inliers, dtype=np.int32)], dtype=np.float64)
            except RuntimeError:
                pass
        c = np.mean(pts, axis=0)
        q = pts - c.reshape(1, 3)
        cov = (q.T @ q) / max(1, q.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        n = np.asarray(vecs[:, int(np.argmin(vals))], dtype=np.float64)
        n = n / max(1e-12, float(np.linalg.norm(n)))
        if np.dot(n, np.array([0.0, 0.0, -1.0], dtype=np.float64)) < 0.0:
            n = -n
        return n

    @staticmethod
    def fit_rotated_quad(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        m = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if np.count_nonzero(m) < 40:
            return None
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return None
        return cv2.boxPoints(cv2.minAreaRect(np.vstack(cnts))).astype(np.float64)

    @staticmethod
    def _enforce_disjoint_region_masks(
        near_plane_mask: np.ndarray | None, top_plane_mask: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if near_plane_mask is None or top_plane_mask is None:
            return near_plane_mask, top_plane_mask
        near = (near_plane_mask > 0).astype(np.uint8) * 255
        top = (top_plane_mask > 0).astype(np.uint8) * 255
        if near.shape != top.shape:
            return near_plane_mask, top_plane_mask
        near_guard = cv2.dilate(near, np.ones((3, 3), dtype=np.uint8), iterations=1)
        top = cv2.bitwise_and(top, cv2.bitwise_not(near_guard))
        top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        if np.count_nonzero(top) < 20:
            top = np.zeros_like(top)
        return near, top

    def _detect_rect_opening_auto(self, rgb_bgr: np.ndarray, tray_mask: np.ndarray, hp_gray: np.ndarray) -> OpeningDetection:
        h, w = rgb_bgr.shape[:2]
        if np.count_nonzero(tray_mask) < 0.01 * h * w:
            raise RuntimeError("托盘掩码无效，无法执行严格开口检测")
        tx, ty, tw, th = self._mask_bbox_xywh(tray_mask)
        x1 = int(max(0, tx + 0.08 * tw))
        x2 = int(min(w - 1, tx + 0.92 * tw))
        y1 = int(max(0, ty + 0.72 * th))
        y2 = int(min(h - 1, ty + 0.97 * th))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("前立面 ROI 无效")
        roi = rgb_bgr[y1:y2, x1:x2]
        roi_tray = tray_mask[y1:y2, x1:x2]
        if roi.size == 0 or np.count_nonzero(roi_tray) < 50:
            raise RuntimeError("前立面托盘像素不足")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        hp = hp_gray[y1:y2, x1:x2]
        thresholds = sorted(set(int(np.clip(t, 20, 180)) for t in np.percentile(gray_blur, [4, 6, 8, 12, 16, 20, 25, 30])))
        candidates: list[tuple[float, np.ndarray]] = []
        for thr in thresholds:
            mask = np.zeros_like(gray_blur, dtype=np.uint8)
            mask[(gray_blur <= thr) & (roi_tray > 0)] = 255
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, gray_blur.shape[1] // 12), 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < 40.0:
                    continue
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), _ = rect
                long_side = max(float(rw), float(rh))
                short_side = max(1.0, min(float(rw), float(rh)))
                aspect = long_side / short_side
                if not (2.8 <= aspect <= 20.0):
                    continue
                wr = long_side / max(1.0, float(tw))
                hr = short_side / max(1.0, float(th))
                if not (0.10 <= wr <= 0.62 and 0.010 <= hr <= 0.095):
                    continue
                y_pref = (y1 + cy - ty) / max(1.0, float(th))
                if y_pref < 0.72:
                    continue
                box = cv2.boxPoints(rect)
                patch_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.fillConvexPoly(patch_mask, np.round(box).astype(np.int32), 255)
                in_roi = (patch_mask > 0) & (roi_tray > 0)
                if np.count_nonzero(in_roi) < 20:
                    continue
                patch_raw = gray[in_roi]
                patch_hp = hp[in_roi]
                dark_ratio_raw = float(np.mean(patch_raw <= thr))
                dark_ratio_hp = float(np.mean(patch_hp <= np.percentile(hp[roi_tray > 0], 16)))
                ring = self._patch_ring(gray_blur, int(cx - rw / 2), int(cy - rh / 2), int(max(1, rw)), int(max(1, rh)))
                slot_mean = float(np.mean(patch_raw))
                ring_mean = float(np.mean(ring)) if ring.size > 0 else slot_mean
                contrast_score = float(np.clip((ring_mean - slot_mean) / 45.0, 0.0, 1.5))
                x_center_pref = 1.0 - min(abs((x1 + cx) - (tx + 0.5 * tw)) / max(1.0, 0.5 * tw), 1.0)
                score = (
                    2.4 * dark_ratio_raw
                    + 1.5 * dark_ratio_hp
                    + 1.5 * min(aspect / 8.0, 2.0)
                    + 1.1 * contrast_score
                    + 1.0 * x_center_pref
                    + 0.7 * y_pref
                )
                box[:, 0] += x1
                box[:, 1] += y1
                candidates.append((float(score), box.astype(np.float64)))
        if len(candidates) == 0:
            raise RuntimeError("未检测到开口")
        candidates.sort(key=lambda x: x[0], reverse=True)
        quad = candidates[0][1]
        bx, by, bw, bh = cv2.boundingRect(np.round(quad).astype(np.int32))
        return OpeningDetection(center_uv=np.mean(quad, axis=0).astype(np.float64), bbox_xywh=(int(bx), int(by), int(bw), int(bh)), quad_uv=quad, score=float(candidates[0][0]))

    def _build_near_dark_plane_mask(
        self, tray_mask: np.ndarray, opening: OpeningDetection, hp_gray: np.ndarray, hp_edge: np.ndarray
    ) -> np.ndarray | None:
        h, w = tray_mask.shape[:2]
        ring = self._build_opening_surround_ring_mask((h, w), opening)
        work = cv2.bitwise_and(ring, tray_mask)
        if np.count_nonzero(work) == 0:
            return None
        seeds = self._collect_opening_boundary_seeds(opening, work, hp_gray)
        if len(seeds) == 0:
            return None
        edge_block = cv2.dilate(hp_edge, np.ones((3, 3), dtype=np.uint8), iterations=1)
        grow = self._region_grow_from_seeds(
            gray=hp_gray,
            allowed_mask=work,
            edge_block=edge_block,
            seeds=seeds,
            local_diff=self._near_grow_local_diff,
            global_diff=self._near_grow_global_diff,
            max_pixels=self._near_grow_max_pixels,
        )
        if np.count_nonzero(grow) < 40:
            return None
        grow = cv2.morphologyEx(grow, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        grow = cv2.morphologyEx(grow, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
        grow = self._select_components_in_ring(grow, ring)
        if np.count_nonzero(grow) < 30:
            return None
        return grow

    def _build_no_hole_top_plane_mask(
        self,
        tray_mask: np.ndarray,
        opening: OpeningDetection,
        near_plane_mask: np.ndarray | None,
        hp_gray: np.ndarray,
        hp_edge: np.ndarray,
    ) -> np.ndarray | None:
        h, w = tray_mask.shape[:2]
        roi_poly = self._build_no_hole_roi_poly((h, w), opening, tray_mask)
        if roi_poly is None:
            return None
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(poly_mask, np.round(roi_poly).astype(np.int32), 255)
        base_mask = cv2.bitwise_and(poly_mask, tray_mask)
        if np.count_nonzero(base_mask) == 0:
            return None
        target = np.mean(roi_poly, axis=0)
        if near_plane_mask is not None and np.count_nonzero(near_plane_mask) > 0:
            near_ring = cv2.dilate(near_plane_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
            near_ring = cv2.bitwise_and(near_ring, cv2.bitwise_not(near_plane_mask))
            seed_band = cv2.bitwise_and(near_ring, base_mask)
            ys, xs = np.where(seed_band > 0)
            if xs.size > 0:
                step = max(1, xs.size // 140)
                seeds: list[tuple[int, int, int]] = []
                for i in range(0, xs.size, step):
                    sx = int(xs[i]); sy = int(ys[i]); seeds.append((sx, sy, int(hp_gray[sy, sx])))
                edge_block = cv2.dilate(hp_edge, np.ones((3, 3), dtype=np.uint8), iterations=1)
                grown = self._region_grow_from_seeds(
                    gray=hp_gray,
                    allowed_mask=base_mask,
                    edge_block=edge_block,
                    seeds=seeds,
                    local_diff=max(10, self._near_grow_local_diff - 2),
                    global_diff=max(18, self._near_grow_global_diff - 6),
                    max_pixels=max(12000, self._near_grow_max_pixels),
                )
                grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
                grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
                grown = self._select_component_near_target(grown, target)
                if np.count_nonzero(grown) >= 180:
                    return grown
        edge_soft = cv2.GaussianBlur(hp_edge.astype(np.float32), (5, 5), 0)
        low_edge = np.zeros_like(base_mask)
        thr = float(np.percentile(edge_soft[base_mask > 0], 55)) if np.count_nonzero(base_mask) > 0 else 0.0
        low_edge[(edge_soft <= thr) & (base_mask > 0)] = 255
        fallback = cv2.morphologyEx(low_edge, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8), iterations=2)
        fallback = self._select_component_near_target(fallback, target)
        if np.count_nonzero(fallback) < 180:
            return None
        return fallback

    @staticmethod
    def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            h, w = mask.shape[:2]
            return 0, 0, w, h
        x1 = int(np.min(xs)); x2 = int(np.max(xs)); y1 = int(np.min(ys)); y2 = int(np.max(ys))
        return x1, y1, x2 - x1 + 1, y2 - y1 + 1

    @staticmethod
    def _select_component_near_target(mask: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        num, cc, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask
        best_id = 0
        best_score = -1e18
        tgt = np.asarray(target_xy, dtype=np.float64)
        for idx in range(1, num):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            c = np.asarray(cent[idx], dtype=np.float64)
            dist = float(np.linalg.norm(c - tgt))
            score = area - 1.3 * dist
            if score > best_score:
                best_score = score
                best_id = idx
        out = np.zeros_like(mask)
        if best_id > 0:
            out[cc == best_id] = 255
        return out

    @staticmethod
    def _select_components_in_ring(mask: np.ndarray, ring: np.ndarray) -> np.ndarray:
        num, cc, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        if num <= 1:
            return mask
        ring_px = ring > 0
        out = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num):
            comp = cc == i; area = float(stats[i, cv2.CC_STAT_AREA])
            if area < 12.0:
                continue
            overlap = float(np.count_nonzero(comp & ring_px))
            if overlap <= 0.0:
                continue
            out[comp] = 255
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
        return out

    @staticmethod
    def _build_opening_local_roi_mask(image_shape: tuple[int, int], opening: OpeningDetection) -> np.ndarray:
        h, w = image_shape
        quad = np.asarray(opening.quad_uv, dtype=np.float64)
        c = np.mean(quad, axis=0)
        v_long, v_short = OpeningDetectionPipeline._opening_axes_from_quad(quad)
        long_len, short_len = OpeningDetectionPipeline._opening_long_short_lengths(quad)
        rect_w = max(16.0, long_len * 2.35); rect_h = max(12.0, short_len * 3.10)
        poly = OpeningDetectionPipeline._rot_rect_to_poly(c, v_long, v_short, rect_w, rect_h)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1); poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8); cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
        return mask

    @staticmethod
    def _build_opening_surround_ring_mask(image_shape: tuple[int, int], opening: OpeningDetection) -> np.ndarray:
        h, w = image_shape
        quad = np.asarray(opening.quad_uv, dtype=np.float64)
        c = np.mean(quad, axis=0)
        v_long, v_short = OpeningDetectionPipeline._opening_axes_from_quad(quad)
        long_len, short_len = OpeningDetectionPipeline._opening_long_short_lengths(quad)
        outer = OpeningDetectionPipeline._rot_rect_to_poly(c, v_long, v_short, max(16.0, long_len * 3.20), max(14.0, short_len * 4.20))
        inner = OpeningDetectionPipeline._rot_rect_to_poly(c, v_long, v_short, max(8.0, long_len * 1.08), max(6.0, short_len * 1.35))
        outer[:, 0] = np.clip(outer[:, 0], 0, w - 1); outer[:, 1] = np.clip(outer[:, 1], 0, h - 1)
        inner[:, 0] = np.clip(inner[:, 0], 0, w - 1); inner[:, 1] = np.clip(inner[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(outer).astype(np.int32), 255); cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 0)
        return mask

    @staticmethod
    def _collect_opening_boundary_seeds(
        opening: OpeningDetection, allowed_mask: np.ndarray, gray: np.ndarray
    ) -> list[tuple[int, int, int]]:
        h, w = allowed_mask.shape[:2]
        quad = np.round(opening.quad_uv).astype(np.int32)
        edge_mask = np.zeros((h, w), dtype=np.uint8); cv2.polylines(edge_mask, [quad], True, 255, 2, cv2.LINE_AA)
        edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        seed_mask = cv2.bitwise_and(edge_mask, allowed_mask)
        ys, xs = np.where(seed_mask > 0)
        if xs.size == 0:
            return []
        step = max(1, xs.size // 120)
        return [(int(xs[i]), int(ys[i]), int(gray[int(ys[i]), int(xs[i])])) for i in range(0, xs.size, step)]

    @staticmethod
    def _region_grow_from_seeds(
        gray: np.ndarray,
        allowed_mask: np.ndarray,
        edge_block: np.ndarray,
        seeds: list[tuple[int, int, int]],
        local_diff: int,
        global_diff: int,
        max_pixels: int,
    ) -> np.ndarray:
        h, w = gray.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8); visited = np.zeros((h, w), dtype=np.uint8)
        q: list[tuple[int, int, int, int]] = []
        for sx, sy, sgv in seeds:
            if sx < 0 or sx >= w or sy < 0 or sy >= h or allowed_mask[sy, sx] == 0:
                continue
            q.append((sx, sy, sgv, sgv)); visited[sy, sx] = 1
        head = 0; pix_count = 0
        while head < len(q):
            x, y, seed_ref, parent_gray = q[head]; head += 1
            gv = int(gray[y, x])
            if abs(gv - seed_ref) > global_diff or abs(gv - parent_gray) > local_diff:
                continue
            if edge_block[y, x] > 0 and abs(gv - parent_gray) > max(4, local_diff // 2):
                continue
            if out[y, x] == 0:
                out[y, x] = 255; pix_count += 1
                if pix_count >= max_pixels:
                    break
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or nx >= w or ny < 0 or ny >= h or visited[ny, nx] != 0:
                    continue
                visited[ny, nx] = 1
                if allowed_mask[ny, nx] == 0:
                    continue
                q.append((nx, ny, seed_ref, gv))
        return out

    @staticmethod
    def _build_no_hole_roi_poly(image_shape: tuple[int, int], opening: OpeningDetection, tray_mask: np.ndarray) -> np.ndarray | None:
        h, w = image_shape
        quad = np.asarray(opening.quad_uv, dtype=np.float64); c = np.mean(quad, axis=0)
        v_long, v_short = OpeningDetectionPipeline._opening_axes_from_quad(quad)
        ys, xs = np.where(tray_mask > 0)
        if xs.size == 0:
            return None
        tray_c = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
        if float(np.dot(v_short, tray_c - c)) < 0.0:
            v_short = -v_short
        long_len, short_len = OpeningDetectionPipeline._opening_long_short_lengths(quad)
        rect_c = c + v_short * (2.05 * short_len)
        poly = OpeningDetectionPipeline._rot_rect_to_poly(rect_c, v_long, v_short, max(18.0, long_len * 1.45), max(14.0, short_len * 2.90))
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1); poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        return poly

    @staticmethod
    def _opening_axes_from_quad(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        best_i = 0; best_len = -1.0
        for i in range(4):
            j = (i + 1) % 4; ll = float(np.linalg.norm(quad[j] - quad[i]))
            if ll > best_len:
                best_len = ll; best_i = i
        v_long = OpeningDetectionPipeline._normalize(quad[(best_i + 1) % 4] - quad[best_i])
        v_short = OpeningDetectionPipeline._normalize(np.array([-v_long[1], v_long[0]], dtype=np.float64))
        return v_long, v_short

    @staticmethod
    def _opening_long_short_lengths(quad: np.ndarray) -> tuple[float, float]:
        lens = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
        return (max(lens), max(2.0, min(lens))) if len(lens) > 0 else (20.0, 6.0)

    @staticmethod
    def _rot_rect_to_poly(center: np.ndarray, v_long: np.ndarray, v_short: np.ndarray, w: float, h: float) -> np.ndarray:
        hw = 0.5 * float(w); hh = 0.5 * float(h); c = np.asarray(center, dtype=np.float64)
        return np.asarray([c - v_long * hw - v_short * hh, c + v_long * hw - v_short * hh, c + v_long * hw + v_short * hh, c - v_long * hw + v_short * hh], dtype=np.float64)

    @staticmethod
    def _patch_ring(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        ih, iw = gray.shape[:2]
        pad_x = max(3, int(round(0.35 * w))); pad_y = max(2, int(round(0.90 * h)))
        x1 = max(0, x - pad_x); y1 = max(0, y - pad_y); x2 = min(iw, x + w + pad_x); y2 = min(ih, y + h + pad_y)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0,), dtype=np.uint8)
        outer = gray[y1:y2, x1:x2]; inner = np.zeros_like(outer, dtype=np.uint8)
        ix1 = x - x1; iy1 = y - y1; ix2 = min(ix1 + w, outer.shape[1]); iy2 = min(iy1 + h, outer.shape[0])
        if ix2 > ix1 and iy2 > iy1:
            inner[iy1:iy2, ix1:ix2] = 255
        return outer[inner == 0]

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < eps:
            raise RuntimeError(f"向量归一化失败，norm={n}")
        return np.asarray(v, dtype=np.float64) / n
