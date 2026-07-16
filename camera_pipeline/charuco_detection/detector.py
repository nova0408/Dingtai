from __future__ import annotations

# pyright: reportMissingImports=false

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..protocol import RgbdFrameProtocol
from .types import (
    CharucoDebugArtifacts,
    CharucoDetectionConfig,
    CharucoDetectionResult,
)


# region 内部数据
@dataclass(frozen=True, slots=True)
class _CornerObservation:
    """单条图像预处理分支产生的角点观测。

    该内部结构只在单次 `detect` 调用期间存在，不跨线程共享，也不持有原图。
    本类不继承业务基类，仅用于约束 OpenCV 返回数组的形状和生命周期。
    """

    marker_corners_px: tuple[np.ndarray, ...]
    "marker 四角点，每项形状 `(4, 1, 2)`，dtype `float32`。"
    marker_ids: np.ndarray
    "marker ID，形状 `(M, 1)`，dtype `int32`。"
    charuco_corners_px: np.ndarray
    "ChArUco 角点，形状 `(N, 1, 2)`，dtype `float32`。"
    charuco_ids: np.ndarray
    "ChArUco ID，形状 `(N, 1)`，dtype `int32`。"


# endregion


# region 检测器
class CharucoDetector:
    """对单帧稳定相机图像执行 ChArUco 检测与位姿求解。

    职责边界：本类只做灰度预处理、角点检测、三分支融合、PnP 求解和可选 overlay。
    它不获取相机帧、不执行稳定性等待、不重试、不传输或保存结果。

    设计思想：先使用原始灰度图执行最低成本检测；位姿失败时复用原始分支结果，
    再增加 CLAHE 和 unsharp 两个分支，并按 ID 对重复角点取像素坐标中位数。

    生命周期：实例持有调用方提供的原生 `cv2.aruco.CharucoBoard` 和 OpenCV
    检测器，可对连续帧复用；不持有帧缓冲和硬件句柄。OpenCV 检测器不保证并发安全，
    同一实例应由单一编排线程调用。本类不继承业务基类。
    """

    def __init__(
        self,
        board: cv2.aruco.CharucoBoard,
        config: CharucoDetectionConfig | None = None,
    ) -> None:
        self._board = board
        self._config = CharucoDetectionConfig() if config is None else config
        self._validate_config()
        self._detector = cv2.aruco.ArucoDetector(
            board.getDictionary(), cv2.aruco.DetectorParameters()
        )

    def detect(
        self,
        frame: RgbdFrameProtocol,
        *,
        enable_debug: bool = False,
    ) -> CharucoDetectionResult:
        """检测单帧 ChArUco 标定板并计算板到相机位姿。

        Parameters
        ----------
        frame:
            相机帧协议。彩色图形状 `(H, W, 3)`、dtype `uint8`、BGR 顺序；
            内参单位 pixel，`distortion` 按 OpenCV 畸变系数顺序排列。
        enable_debug:
            是否复制输入彩色图并绘制 marker、ChArUco 角点和 pose 坐标轴。

        Returns
        -------
        CharucoDetectionResult
            单帧检测结果。原始灰度图无法得到位姿时，结果自动融合 none、CLAHE、
            unsharp 三条分支的唯一角点后再次求解。

        Raises
        ------
        ValueError
            图像、相机内参或畸变参数不合法。

        Notes
        -----
        `t_cam_board_mm` 满足 `p_cam = T_cam_board @ p_board`。由于板尺寸约定为 mm，
        PnP 平移分量同样为 mm。
        """

        image_bgr = np.asarray(frame.color_bgr)
        camera_matrix = self._camera_matrix(frame)
        dist_coeffs = np.asarray(frame.distortion, dtype=np.float64).reshape(-1, 1)
        self._validate_inputs(image_bgr, camera_matrix, dist_coeffs)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        raw = self._detect_corners(gray, camera_matrix, dist_coeffs)
        raw_pose = self._solve_pose(raw, camera_matrix, dist_coeffs)
        if raw_pose is not None:
            return self._build_result(
                image_bgr, raw, raw_pose, camera_matrix, dist_coeffs, enable_debug
            )

        observations = (
            raw,
            self._detect_corners(self._apply_clahe(gray), camera_matrix, dist_coeffs),
            self._detect_corners(self._apply_unsharp(gray), camera_matrix, dist_coeffs),
        )
        merged = self._merge_observations(observations)
        merged_pose = self._solve_pose(merged, camera_matrix, dist_coeffs)
        return self._build_result(
            image_bgr,
            merged,
            merged_pose,
            camera_matrix,
            dist_coeffs,
            enable_debug,
        )

    # region 检测与融合
    def _detect_corners(
        self,
        gray: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> _CornerObservation:
        """从一幅灰度图检测 marker 并插值 ChArUco 角点。

        Parameters
        ----------
        gray:
            灰度图，形状 `(H, W)`，dtype `uint8`。
        camera_matrix:
            相机内参矩阵，形状 `(3, 3)`，dtype `float64`。
        dist_coeffs:
            OpenCV 畸变系数，形状 `(D, 1)`，dtype `float64`。

        Returns
        -------
        _CornerObservation
            当前预处理分支的规范化角点观测；未检测到时数组为空。
        """

        marker_corners_raw, marker_ids_raw, _ = self._detector.detectMarkers(gray)
        marker_corners = self._normalize_marker_corners(marker_corners_raw)
        marker_ids = self._normalize_ids(marker_ids_raw)
        if not marker_corners or marker_ids.size == 0:
            return self._empty_observation()

        _, charuco_corners_raw, charuco_ids_raw = cv2.aruco.interpolateCornersCharuco(
            list(marker_corners),
            marker_ids,
            gray,
            self._board,
            camera_matrix,
            dist_coeffs,
        )
        charuco_corners = self._normalize_charuco_corners(charuco_corners_raw)
        charuco_ids = self._normalize_ids(charuco_ids_raw)
        return _CornerObservation(
            marker_corners_px=marker_corners,
            marker_ids=marker_ids,
            charuco_corners_px=charuco_corners,
            charuco_ids=charuco_ids,
        )

    def _merge_observations(
        self, observations: Sequence[_CornerObservation]
    ) -> _CornerObservation:
        """按 ID 融合三条预处理分支的角点。

        Parameters
        ----------
        observations:
            none、CLAHE、unsharp 分支观测序列，长度通常为 3。

        Returns
        -------
        _CornerObservation
            每个唯一 ID 只保留一个坐标；重复坐标逐维取中位数，降低单分支偏差。

        Notes
        -----
        marker 数组按 `(4, 1, 2)` 融合，ChArUco 数组按 `(1, 2)` 融合。
        """

        marker_samples: dict[int, list[np.ndarray]] = {}
        charuco_samples: dict[int, list[np.ndarray]] = {}
        for observation in observations:
            for index, marker_id in enumerate(observation.marker_ids.reshape(-1)):
                marker_samples.setdefault(int(marker_id), []).append(
                    observation.marker_corners_px[index]
                )
            for index, charuco_id in enumerate(observation.charuco_ids.reshape(-1)):
                charuco_samples.setdefault(int(charuco_id), []).append(
                    observation.charuco_corners_px[index]
                )

        marker_ids_sorted = sorted(marker_samples)
        charuco_ids_sorted = sorted(charuco_samples)
        # stack: (B, 4, 1, 2) -> median: (4, 1, 2)，B 为检测到该 ID 的分支数。
        marker_corners = tuple(
            np.median(np.stack(marker_samples[item]), axis=0).astype(np.float32)
            for item in marker_ids_sorted
        )
        # stack: (B, 1, 2) -> median: (1, 2)，保证每个物理角点只参与一次 PnP。
        charuco_corners = (
            np.stack(
                [
                    np.median(np.stack(charuco_samples[item]), axis=0)
                    for item in charuco_ids_sorted
                ]
            ).astype(np.float32)
            if charuco_ids_sorted
            else np.empty((0, 1, 2), dtype=np.float32)
        )
        return _CornerObservation(
            marker_corners_px=marker_corners,
            marker_ids=np.asarray(marker_ids_sorted, dtype=np.int32).reshape(-1, 1),
            charuco_corners_px=charuco_corners,
            charuco_ids=np.asarray(charuco_ids_sorted, dtype=np.int32).reshape(-1, 1),
        )

    # endregion

    # region 位姿与输出
    def _solve_pose(
        self,
        observation: _CornerObservation,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | None:
        """从唯一 ChArUco 角点求解板到相机位姿。

        Parameters
        ----------
        observation:
            已规范化且按 ID 去重的角点观测。
        camera_matrix:
            相机内参矩阵，形状 `(3, 3)`，dtype `float64`。
        dist_coeffs:
            OpenCV 畸变系数，形状 `(D, 1)`，dtype `float64`。

        Returns
        -------
        tuple[np.ndarray, float, np.ndarray, np.ndarray] | None
            成功时返回 `(T_cam_board, error_px, rvec, tvec)`；角点不足或 PnP 失败时为空。
        """

        if observation.charuco_ids.shape[0] < self._config.min_charuco_corners:
            return None
        board_corners = np.asarray(
            self._board.getChessboardCorners(), dtype=np.float64
        )
        ids = observation.charuco_ids.reshape(-1)
        # obj_points: (N, 3) mm；img_points: (N, 2) pixel，二者按唯一 ID 一一对应。
        obj_points = board_corners[ids]
        img_points = observation.charuco_corners_px.reshape(-1, 2).astype(np.float64)
        success, rvec_raw, tvec_raw = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        rvec = np.asarray(rvec_raw, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec_raw, dtype=np.float64).reshape(3)
        projected, _ = cv2.projectPoints(
            obj_points,
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            camera_matrix,
            dist_coeffs,
        )
        # projected/img_points: (N, 2)，逐点欧氏距离均值作为像素重投影误差。
        error_px = float(
            np.mean(np.linalg.norm(projected.reshape(-1, 2) - img_points, axis=1))
        )
        t_cam_board = np.eye(4, dtype=np.float64)
        t_cam_board[:3, :3], _ = cv2.Rodrigues(rvec)
        t_cam_board[:3, 3] = tvec
        return t_cam_board, error_px, rvec, tvec

    def _build_result(
        self,
        image_bgr: np.ndarray,
        observation: _CornerObservation,
        pose: tuple[np.ndarray, float, np.ndarray, np.ndarray] | None,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        enable_debug: bool,
    ) -> CharucoDetectionResult:
        """构造核心结果，并仅在启用时生成 overlay。

        Parameters
        ----------
        image_bgr:
            原始彩色图，形状 `(H, W, 3)`，dtype `uint8`。
        observation:
            最终用于位姿求解的融合角点。
        pose:
            位姿、误差、旋转向量和平移向量；求解失败时为空。
        camera_matrix, dist_coeffs:
            OpenCV 内参矩阵与畸变系数，用于绘制 pose 坐标轴。
        enable_debug:
            是否创建调试图像和角点数组副本。

        Returns
        -------
        CharucoDetectionResult
            不含 IO 副作用的检测结果。
        """

        debug_artifacts: tuple[CharucoDebugArtifacts, ...] = ()
        if enable_debug:
            overlay = image_bgr.copy()
            if observation.marker_ids.size > 0:
                cv2.aruco.drawDetectedMarkers(
                    overlay,
                    list(observation.marker_corners_px),
                    observation.marker_ids,
                )
            if observation.charuco_ids.size > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    overlay,
                    observation.charuco_corners_px,
                    observation.charuco_ids,
                )
            if pose is not None:
                _, _, rvec, tvec = pose
                square_length_mm = float(self._board.getSquareLength())
                cv2.drawFrameAxes(
                    overlay,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    square_length_mm * self._config.axis_length_scale,
                    3,
                )
            debug_artifacts = (
                CharucoDebugArtifacts(
                    overlay_bgr=overlay,
                    marker_corners_px=tuple(
                        item.copy() for item in observation.marker_corners_px
                    ),
                    marker_ids=observation.marker_ids.copy(),
                    charuco_corners_px=observation.charuco_corners_px.copy(),
                    charuco_ids=observation.charuco_ids.copy(),
                ),
            )

        if pose is None:
            return CharucoDetectionResult(
                status="missing",
                t_cam_board_mm=np.empty((0, 0), dtype=np.float64),
                error_px=float("inf"),
                marker_num=int(observation.marker_ids.shape[0]),
                charuco_num=int(observation.charuco_ids.shape[0]),
                debug_artifacts=debug_artifacts,
            )
        t_cam_board, error_px, _, _ = pose
        return CharucoDetectionResult(
            status="detected",
            t_cam_board_mm=t_cam_board,
            error_px=error_px,
            marker_num=int(observation.marker_ids.shape[0]),
            charuco_num=int(observation.charuco_ids.shape[0]),
            debug_artifacts=debug_artifacts,
        )

    # endregion

    # region 预处理与校验
    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """对灰度图应用 CLAHE 局部对比度增强。

        Parameters
        ----------
        gray:
            输入灰度图，形状 `(H, W)`，dtype `uint8`。

        Returns
        -------
        np.ndarray
            增强灰度图，形状和 dtype 与输入一致。
        """

        clahe = cv2.createCLAHE(
            clipLimit=self._config.clahe_clip_limit,
            tileGridSize=self._config.clahe_grid_size,
        )
        return clahe.apply(gray)

    def _apply_unsharp(self, gray: np.ndarray) -> np.ndarray:
        """对灰度图应用轻量反锐化增强。

        Parameters
        ----------
        gray:
            输入灰度图，形状 `(H, W)`，dtype `uint8`。

        Returns
        -------
        np.ndarray
            增强灰度图，形状 `(H, W)`，dtype `uint8`。
        """

        blurred = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=self._config.unsharp_sigma
        )
        return cv2.addWeighted(
            gray,
            1.0 + self._config.unsharp_amount,
            blurred,
            -self._config.unsharp_amount,
            0.0,
        )

    @staticmethod
    def _camera_matrix(frame: RgbdFrameProtocol) -> np.ndarray:
        """从帧协议构造 OpenCV 相机内参矩阵。

        Parameters
        ----------
        frame:
            含 `fx/fy/cx/cy` 的只读相机帧协议，单位 pixel。

        Returns
        -------
        np.ndarray
            相机内参矩阵，形状 `(3, 3)`，dtype `float64`。
        """

        return np.asarray(
            [[frame.fx, 0.0, frame.cx], [0.0, frame.fy, frame.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _normalize_marker_corners(
        corners: Sequence[np.ndarray] | np.ndarray | None,
    ) -> tuple[np.ndarray, ...]:
        """规范化 OpenCV marker 角点为固定形状元组。

        Parameters
        ----------
        corners:
            OpenCV marker 角点序列，每项可转换为 8 个浮点坐标。

        Returns
        -------
        tuple[np.ndarray, ...]
            每项形状 `(4, 1, 2)`、dtype `float32`；无角点时为空元组。
        """

        if corners is None:
            return ()
        return tuple(
            np.asarray(item, dtype=np.float32).reshape(4, 1, 2) for item in corners
        )

    @staticmethod
    def _normalize_charuco_corners(corners: np.ndarray | None) -> np.ndarray:
        """规范化 ChArUco 角点数组。

        Parameters
        ----------
        corners:
            OpenCV ChArUco 角点数组或空值。

        Returns
        -------
        np.ndarray
            形状 `(N, 1, 2)`、dtype `float32`；无角点时 `N=0`。
        """

        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

    @staticmethod
    def _normalize_ids(ids: np.ndarray | None) -> np.ndarray:
        """规范化 OpenCV ID 数组。

        Parameters
        ----------
        ids:
            OpenCV marker 或 ChArUco ID 数组。

        Returns
        -------
        np.ndarray
            形状 `(N, 1)`、dtype `int32`；无 ID 时 `N=0`。
        """

        if ids is None:
            return np.empty((0, 1), dtype=np.int32)
        return np.asarray(ids, dtype=np.int32).reshape(-1, 1)

    @staticmethod
    def _empty_observation() -> _CornerObservation:
        """构造不含 marker 和 ChArUco 角点的内部观测。"""

        return _CornerObservation(
            marker_corners_px=(),
            marker_ids=np.empty((0, 1), dtype=np.int32),
            charuco_corners_px=np.empty((0, 1, 2), dtype=np.float32),
            charuco_ids=np.empty((0, 1), dtype=np.int32),
        )

    def _validate_config(self) -> None:
        """校验检测配置中的数量、尺寸和增强参数范围。"""

        if self._config.min_charuco_corners < 4:
            raise ValueError("min_charuco_corners must be at least 4")
        if self._config.clahe_clip_limit <= 0.0:
            raise ValueError("clahe_clip_limit must be greater than zero")
        if min(self._config.clahe_grid_size) <= 0:
            raise ValueError("clahe_grid_size values must be greater than zero")
        if self._config.unsharp_sigma <= 0.0:
            raise ValueError("unsharp_sigma must be greater than zero")
        if self._config.unsharp_amount <= 0.0:
            raise ValueError("unsharp_amount must be greater than zero")
        if self._config.axis_length_scale <= 0.0:
            raise ValueError("axis_length_scale must be greater than zero")

    @staticmethod
    def _validate_inputs(
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """校验 ChArUco 检测所需图像与相机标定数组。"""

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("color_bgr must have shape (H, W, 3)")
        if image_bgr.dtype != np.uint8:
            raise ValueError("color_bgr must use uint8 dtype")
        if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
            raise ValueError("camera intrinsics must be finite")
        if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be greater than zero")
        if dist_coeffs.size == 0 or not np.all(np.isfinite(dist_coeffs)):
            raise ValueError("camera distortion must contain finite coefficients")

    # endregion


# endregion
