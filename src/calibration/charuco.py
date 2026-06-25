from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# region 数据结构
CHARUCO_200_12_9 = cv2.aruco.CharucoBoard(
    (12, 9),
    15.0,
    11.25,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
)


@dataclass(frozen=True, slots=True)
class CharucoPoseResult:
    """单帧 ChArUco 检测与位姿结果。

    该结果只表达计算产物，不持有检测器和图像对象。
    调用方可以直接从 `transform` 获取最终位姿，也可以使用 `rvec`、`tvec`
    做更细粒度的调试显示。

    生命周期：
    - 纯数据结构
    - 适合在预览页、记录页和机器人标定页之间传递
    - 不负责再次计算
    """

    marker_count: int
    "检测到的 marker 数量。"
    charuco_count: int
    "插值得到的 ChArUco 角点数量。"
    board_visible: bool
    "当前帧是否获得足够角点用于位姿估计。"
    reprojection_error_px: float | None
    "重投影误差，单位像素。未求得位姿时为 `None`。"
    marker_corners_px: list[np.ndarray]
    "检测到的 marker 四角点列表，每项形状为 `(4, 1, 2)`，dtype 为 `float32`。"
    marker_ids: np.ndarray | None
    "marker 编号数组，形状为 `(N, 1)`，dtype 为 `int32`。"
    charuco_corners_px: np.ndarray | None
    "插值得到的 ChArUco 角点坐标，形状为 `(N, 2)`，dtype 为 `float64`。"
    charuco_ids: np.ndarray | None
    "插值得到的 ChArUco 角点编号，形状为 `(N, 1)`，dtype 为 `int32`。"
    rvec: np.ndarray | None
    "旋转向量，形状为 `(3,)`，单位为弧度。"
    tvec: np.ndarray | None
    "平移向量，形状为 `(3,)`，单位与板尺寸一致，当前为 mm。"
    transform_se3: np.ndarray | None
    "最终位姿，4x4 SE(3) 齐次矩阵，dtype 为 `float64`。"


# endregion


# region 核心求解器
class CharucoPoseEstimator:
    """ChArUco 板检测与位姿求解器。

    这个类只负责单帧图像到位姿结果的转换，不负责采集线程、UI 显示或机器人控制。
    通过把板参数、字典和检测流程集中到一个对象中，后续接机器人时只要提供图像和
    相机标定参数即可复用同一套算法，不需要依赖奥比中光 SDK。

    设计思想：
    - 板参数使用显式配置对象，避免散落硬编码
    - 检测流程严格按照 OpenCV 官方示例，先 marker 后 ChArUco 插值，再做位姿求解
    - 不做版本兼容分支，不使用 `hasattr` 规避缺失 API

    生命周期：
    - 可长期复用
    - 持有 OpenCV 检测器与板配置
    - 不持有图像缓冲和硬件句柄
    """

    def __init__(self, board: cv2.aruco.CharucoBoard) -> None:
        self._board = board
        self._detector = cv2.aruco.ArucoDetector(board.getDictionary(), cv2.aruco.DetectorParameters())

    @property
    def board(self) -> cv2.aruco.CharucoBoard:
        """返回当前使用的原生 ChArUco 板对象。"""
        return self._board

    @property
    def legacy_pattern(self) -> bool:
        """返回 legacy 图案开关。"""
        return bool(self._board.getLegacyPattern())

    def build_board(self) -> cv2.aruco.CharucoBoard:
        """构造当前参数对应的 ChArUco 板对象。"""
        return self._board

    def generate_board_image(self, out_size_px: tuple[int, int]) -> np.ndarray:
        """生成用于人工对照的理论板图。

        Parameters
        ----------
        out_size_px:
            输出图像尺寸，格式为 `(width, height)`，单位像素。

        Returns
        -------
        np.ndarray
            生成的板图，形状为 `(H, W)` 或 `(H, W, 3)`，取决于 OpenCV 输出。
        """
        board = self.build_board()
        return board.generateImage(out_size_px)

    def estimate_pose(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        min_charuco_corners: int = 6,
    ) -> CharucoPoseResult:
        """从单帧图像计算 ChArUco 位姿。

        Parameters
        ----------
        image_bgr:
            输入彩色图像，形状为 `(H, W, 3)`，BGR 顺序，dtype 通常为 `uint8`。
        camera_matrix:
            相机内参矩阵，形状为 `(3, 3)`，dtype 为 `float64`。
        dist_coeffs:
            相机畸变系数，形状为 `(N, 1)` 或 `(1, N)`，dtype 为 `float64`。
        min_charuco_corners:
            允许进入位姿求解的最小 ChArUco 角点数，单位为点。默认值为 6。

        Returns
        -------
        CharucoPoseResult
            识别、插值和位姿求解结果。若角点数量不足，则 `transform_se3` 为 `None`。

        Raises
        ------
        ValueError
            当输入图像或内参尺寸不合法时抛出。

        Notes
        -----
        该方法严格沿用 OpenCV 官方 ChArUco 流程。
        1. 先检测 marker
        2. 再插值 ChArUco 角点
        3. 角点数量足够时调用 `solvePnP`
        4. 最终将 `rvec`、`tvec` 转为 `Transform`

        当前实现不依赖任何 SDK，适合后续接入机器人控制器或其它相机来源。
        """
        self._validate_inputs(image_bgr, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        board = self.build_board()

        marker_corners_raw, marker_ids_raw, _ = self._detector.detectMarkers(gray)
        marker_corners = self._normalize_marker_corners(marker_corners_raw)
        marker_ids = np.asarray(marker_ids_raw, dtype=np.int32).reshape(-1, 1)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))

        if not marker_corners or marker_ids is None:
            return CharucoPoseResult(
                marker_count=marker_count,
                charuco_count=0,
                board_visible=False,
                reprojection_error_px=None,
                marker_corners_px=marker_corners,
                marker_ids=marker_ids,
                charuco_corners_px=None,
                charuco_ids=None,
                rvec=None,
                tvec=None,
                transform_se3=None,
            )

        charuco_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
            camera_matrix,
            dist_coeffs,
        )

        if charuco_ids is None or charuco_corners is None or charuco_count < min_charuco_corners:
            return CharucoPoseResult(
                marker_count=marker_count,
                charuco_count=0 if charuco_count is None else int(charuco_count),
                board_visible=False,
                reprojection_error_px=None,
                marker_corners_px=marker_corners,
                marker_ids=marker_ids,
                charuco_corners_px=None,
                charuco_ids=None,
                rvec=None,
                tvec=None,
                transform_se3=None,
            )

        obj_points, img_points = self._charuco_object_image_points(charuco_corners, charuco_ids, board)
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return CharucoPoseResult(
                marker_count=marker_count,
                charuco_count=int(charuco_count),
                board_visible=True,
                reprojection_error_px=None,
                marker_corners_px=marker_corners,
                marker_ids=marker_ids,
                charuco_corners_px=np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
                charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1),
                rvec=None,
                tvec=None,
                transform_se3=None,
            )

        rvec_vec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec_vec = np.asarray(tvec, dtype=np.float64).reshape(3)
        projected, _ = cv2.projectPoints(
            obj_points,
            rvec_vec.reshape(3, 1),
            tvec_vec.reshape(3, 1),
            camera_matrix,
            dist_coeffs,
        )
        projected_2d = projected.reshape(-1, 2)
        reprojection_error = float(np.mean(np.linalg.norm(projected_2d - img_points, axis=1)))
        transform_se3 = np.eye(4, dtype=np.float64)
        transform_se3[:3, :3], _ = cv2.Rodrigues(rvec_vec)
        transform_se3[:3, 3] = tvec_vec
        return CharucoPoseResult(
            marker_count=marker_count,
            charuco_count=int(charuco_count),
            board_visible=True,
            reprojection_error_px=reprojection_error,
            marker_corners_px=marker_corners,
            marker_ids=marker_ids,
            charuco_corners_px=np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
            charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1),
            rvec=rvec_vec,
            tvec=tvec_vec,
            transform_se3=transform_se3,
        )

    def _charuco_object_image_points(
        self,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
        board: cv2.aruco.CharucoBoard,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 ChArUco 角点编号映射为 3D-2D 对应点。

        这里的对象点来自棋盘平面坐标，单位与板尺寸一致，当前为 mm。
        图像点保持像素坐标不变，供 `solvePnP` 使用。
        """
        board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
        ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        img_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        obj_points = board_corners[ids_flat]
        return obj_points, img_points

    def _validate_inputs(self, image_bgr: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """校验单帧推理所需输入。"""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"期望 BGR 图像形状为 (H, W, 3)，实际为 {image_bgr.shape}")
        if camera_matrix.shape != (3, 3):
            raise ValueError(f"期望相机内参形状为 (3, 3)，实际为 {camera_matrix.shape}")
        if dist_coeffs.ndim not in (1, 2):
            raise ValueError(f"期望畸变参数为 1D 或 2D 数组，实际为 {dist_coeffs.shape}")

    @staticmethod
    def _normalize_marker_corners(marker_corners: Sequence[np.ndarray] | np.ndarray | None) -> list[np.ndarray]:
        """将 marker 角点整理成 OpenCV 绑定要求的输入形状。"""
        if marker_corners is None:
            return []
        normalized: list[np.ndarray] = []
        for item in marker_corners:
            normalized.append(np.asarray(item, dtype=np.float32).reshape(4, 1, 2))
        return normalized


# endregion
