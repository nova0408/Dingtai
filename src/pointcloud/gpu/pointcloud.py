import time

import numpy as np
import torch
from loguru import logger


# region GPUPointCloud
class GPUPointCloud:
    """GPU 常驻点云容器。

    Notes
    -----
    - 点云主数据以 ``torch.Tensor`` 形式常驻 GPU。
    - 空间索引仅基于 XYZ（三列）构建。
    - Faiss 索引数据来源于 CPU 侧连续 ``float32`` 缓存。
    - 支持缓存复用：空间索引缓存与体素化缓存。
    - 对超大点云的半径离群去除支持自动预体素化。
    """

    # region 初始化与基础属性
    def __init__(self, points_np: np.ndarray | torch.Tensor, device: torch.device | None = None):
        """构建 GPU 点云对象。

        Parameters
        ----------
        points_np : np.ndarray | torch.Tensor
            输入点云，形状为 ``(N, D)``，且 ``D >= 3``。
        device : torch.device | None, optional
            目标设备。为 ``None`` 时优先使用 CUDA，否则使用 CPU。

        Raises
        ------
        ValueError
            输入类型不支持，或输入维度不满足 ``(N, D>=3)``。
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if isinstance(points_np, np.ndarray):
            if points_np.ndim != 2 or points_np.shape[1] < 3:
                raise ValueError(f"points_np 形状必须为 (N, D) 且 D>=3，实际为 {points_np.shape}")
            points_np = np.ascontiguousarray(points_np.astype(np.float32, copy=False))
            self.tensor = torch.from_numpy(points_np).to(self.device, non_blocking=False)
            # 仅在需要 CPU 侧索引/导出时再构建，避免初始化阶段重复拷贝大数组。
            self._xyz_cpu_cache: np.ndarray | None = None
        elif isinstance(points_np, torch.Tensor):
            if points_np.ndim != 2 or points_np.shape[1] < 3:
                raise ValueError(f"points_np 形状必须为 (N, D) 且 D>=3，实际为 {tuple(points_np.shape)}")
            self.tensor = points_np.to(device=self.device, dtype=torch.float32).contiguous()
            self._xyz_cpu_cache: np.ndarray | None = None
        else:
            raise ValueError("Input must be numpy array or torch tensor")

        self.N, self.D = self.tensor.shape
        self.centroid: torch.Tensor | None = None
        self._spatial_index_cache: dict[tuple[str, int, int, int], "GPUSpatialIndex"] = {}
        self._voxel_cache: dict[tuple[float, str, bool], tuple["GPUPointCloud", torch.Tensor]] = {}

    @property
    def xyz(self) -> torch.Tensor:
        """返回 XYZ 视图。

        Returns
        -------
        torch.Tensor
            形状为 ``(N, 3)`` 的张量视图。
        """
        return self.tensor[:, :3]

    # endregion

    # region 基础操作与缓存
    def center_data(self) -> torch.Tensor:
        """将点云按 XYZ 质心平移到原点附近。

        Returns
        -------
        torch.Tensor
            平移前质心，形状为 ``(3,)``。
        """
        self.centroid = self.xyz.mean(dim=0)
        self.tensor[:, :3] -= self.centroid
        if self._xyz_cpu_cache is not None:
            centroid_np = self.centroid.detach().cpu().numpy().astype(np.float32, copy=False)
            self._xyz_cpu_cache = np.ascontiguousarray(self._xyz_cpu_cache - centroid_np)
        self._clear_all_caches()
        return self.centroid

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """按索引收集点。

        Parameters
        ----------
        indices : torch.Tensor
            可用于高级索引的张量。

        Returns
        -------
        torch.Tensor
            被索引后的点集。
        """
        return self.tensor[indices]

    def to_numpy(self) -> np.ndarray:
        """导出完整点云为 NumPy。

        Returns
        -------
        np.ndarray
            CPU 侧数组。
        """
        return self.tensor.detach().cpu().numpy()

    def xyz_cpu_numpy(self) -> np.ndarray:
        """获取 CPU 侧连续 XYZ 缓存。

        Returns
        -------
        np.ndarray
            形状为 ``(N, 3)`` 的 ``float32`` 连续数组。
        """
        if self._xyz_cpu_cache is None:
            self._xyz_cpu_cache = np.ascontiguousarray(self.xyz.detach().cpu().numpy().astype(np.float32, copy=False))
        return self._xyz_cpu_cache

    def icp_point_to_point(
        self,
        target: "GPUPointCloud | torch.Tensor | np.ndarray",
        max_iterations: int = 30,
        tolerance: float = 1e-5,
        max_correspondence_distance: float | None = None,
        init_transform: torch.Tensor | None = None,
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
    ) -> "ICPPointToPointResult":
        """执行点到点 ICP（GPU 张量优先）。"""
        from .icp import ICPPointToPointResult, icp_point_to_point

        if isinstance(target, GPUPointCloud):
            target_pcd = target
        else:
            target_pcd = GPUPointCloud(target, device=self.device)

        target_index = target_pcd.get_spatial_index(
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
        )
        return icp_point_to_point(
            source_xyz=self.xyz,
            target_xyz=target_pcd.xyz,
            target_index=target_index,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            init_transform=init_transform,
        )

    def icp_adaptive_curvature(
        self,
        target: "GPUPointCloud | torch.Tensor | np.ndarray",
        max_iterations: int = 40,
        tolerance: float = 1e-5,
        max_correspondence_distance: float | None = None,
        init_transform: torch.Tensor | None = None,
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
        normal_k: int = 20,
        curvature_k: int = 20,
        curvature_power: float = 1.0,
        curvature_min_weight: float = 0.20,
        trim_ratio: float = 0.10,
        huber_delta: float = 0.80,
        lm_lambda: float = 1e-4,
        step_scale: float = 0.35,
    ) -> "ICPAdaptiveCurvatureResult":
        """执行曲率自适应加权 point-to-plane ICP。"""
        from .icp import ICPAdaptiveCurvatureResult, icp_adaptive_curvature

        if isinstance(target, GPUPointCloud):
            target_pcd = target
        else:
            target_pcd = GPUPointCloud(target, device=self.device)

        target_index = target_pcd.get_spatial_index(
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
        )
        return icp_adaptive_curvature(
            source_xyz=self.xyz,
            target_xyz=target_pcd.xyz,
            target_index=target_index,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            init_transform=init_transform,
            normal_k=normal_k,
            curvature_k=curvature_k,
            curvature_power=curvature_power,
            curvature_min_weight=curvature_min_weight,
            trim_ratio=trim_ratio,
            huber_delta=huber_delta,
            lm_lambda=lm_lambda,
            step_scale=step_scale,
        )

    def clear_cache(self) -> None:
        """清理所有派生缓存。"""
        self._xyz_cpu_cache = None
        self._clear_all_caches()

    def release_index_cache(self) -> None:
        """主动释放空间索引缓存及其底层资源。"""
        self._clear_spatial_index_cache()

    def _clear_all_caches(self) -> None:
        """清理内部缓存（索引缓存 + 体素缓存）。"""
        self._clear_spatial_index_cache()
        self._voxel_cache.clear()

    def _clear_spatial_index_cache(self) -> None:
        """清理并释放空间索引缓存。"""
        for index in self._spatial_index_cache.values():
            try:
                index.release()
            except Exception:
                logger.debug("释放空间索引资源失败，将继续清理缓存。")
        self._spatial_index_cache.clear()

    # endregion

    # region 索引与体素缓存工厂
    def get_spatial_index(
        self,
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
    ) -> "GPUSpatialIndex":
        """获取或构建空间索引。

        Parameters
        ----------
        nprobe : int, optional
            IVF 搜索时探测的倒排列表数量。
        temp_memory_mb : int, optional
            Faiss GPU 临时内存池大小（MB）。
        train_size : int, optional
            IVF 训练样本数上限。

        Returns
        -------
        GPUSpatialIndex
            缓存命中或新建的空间索引对象。
        """
        cache_key = (
            "IVF",
            int(max(1, nprobe)),
            int(max(1, temp_memory_mb)),
            int(max(1, train_size)),
        )
        cached = self._spatial_index_cache.get(cache_key)
        if cached is not None:
            return cached

        from .index import GPUSpatialIndex

        index = GPUSpatialIndex(
            self,
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
        )
        self._spatial_index_cache[cache_key] = index
        return index

    def get_or_create_voxelized(
        self,
        voxel_size: float,
        keep: str = "centroid",
        xyz_only: bool = False,
    ) -> tuple["GPUPointCloud", torch.Tensor]:
        """获取或构建体素化结果及逆映射。

        Parameters
        ----------
        voxel_size : float
            体素尺寸。
        keep : str, optional
            体素代表点策略，支持 ``"first"`` / ``"centroid"``。
        xyz_only : bool, optional
            为 ``True`` 时仅对 XYZ 列进行输出。

        Returns
        -------
        tuple[GPUPointCloud, torch.Tensor]
            下采样点云与逆映射（原始点 -> 体素点索引）。
        """
        key = (float(voxel_size), keep, bool(xyz_only))
        cached = self._voxel_cache.get(key)
        if cached is not None:
            return cached

        down_pcd, inverse = self.voxel_down_sample_with_inverse(
            voxel_size=voxel_size,
            keep=keep,
            xyz_only=xyz_only,
        )
        self._voxel_cache[key] = (down_pcd, inverse)
        return down_pcd, inverse

    # endregion

    # region 体素降采样
    def voxel_down_sample_with_inverse(
        self,
        voxel_size: float,
        keep: str = "centroid",
        xyz_only: bool = False,
    ) -> tuple["GPUPointCloud", torch.Tensor]:
        """执行体素下采样并返回逆映射。

        Parameters
        ----------
        voxel_size : float
            体素尺寸，必须大于 0。
        keep : str, optional
            体素代表点策略：``"first"`` 或 ``"centroid"``。
        xyz_only : bool, optional
            是否仅输出 XYZ 三列。

        Returns
        -------
        tuple[GPUPointCloud, torch.Tensor]
            下采样点云与逆映射（原始点 -> 体素点索引）。

        Raises
        ------
        ValueError
            参数不合法时抛出。
        """
        if voxel_size <= 0:
            raise ValueError(f"voxel_size 必须大于 0，实际为 {voxel_size}")
        if keep not in {"first", "centroid"}:
            raise ValueError(f"keep 仅支持 'first' 或 'centroid'，实际为 {keep!r}")

        src = self.xyz
        work_tensor = self.xyz if xyz_only else self.tensor

        min_bound = torch.min(src, dim=0).values
        voxel_coords = torch.floor((src - min_bound) / voxel_size).to(torch.int64)

        unique_voxels, inverse = torch.unique(
            voxel_coords,
            dim=0,
            sorted=True,
            return_inverse=True,
        )

        if keep == "first":
            out_tensor = self._voxel_first_reduce(
                points=work_tensor,
                inverse=inverse,
                num_voxels=unique_voxels.shape[0],
            )
        else:
            out_tensor = self._voxel_centroid_reduce(
                points=work_tensor,
                inverse=inverse,
                num_voxels=unique_voxels.shape[0],
            )

        return GPUPointCloud(out_tensor, device=self.device), inverse

    def voxel_down_sample(
        self,
        voxel_size: float,
        keep: str = "centroid",
        xyz_only: bool = False,
    ) -> "GPUPointCloud":
        """执行体素下采样。

        Parameters
        ----------
        voxel_size : float
            体素尺寸。
        keep : str, optional
            体素代表点策略。
        xyz_only : bool, optional
            是否仅输出 XYZ。

        Returns
        -------
        GPUPointCloud
            下采样后的点云。
        """
        out_pcd, _ = self.voxel_down_sample_with_inverse(
            voxel_size=voxel_size,
            keep=keep,
            xyz_only=xyz_only,
        )
        return out_pcd

    def _voxel_first_reduce(
        self,
        points: torch.Tensor,
        inverse: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        """体素归约：保留每个体素首次出现的点。"""
        n_points = points.shape[0]
        point_indices = torch.arange(n_points, device=points.device, dtype=torch.long)
        first_indices = torch.full(
            (num_voxels,),
            fill_value=n_points,
            device=points.device,
            dtype=torch.long,
        )
        first_indices.scatter_reduce_(
            0,
            inverse,
            point_indices,
            reduce="amin",
            include_self=True,
        )
        return points[first_indices]

    def _voxel_centroid_reduce(
        self,
        points: torch.Tensor,
        inverse: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        """体素归约：保留每个体素质心。"""
        out = torch.zeros(
            (num_voxels, points.shape[1]),
            dtype=points.dtype,
            device=points.device,
        )
        counts = torch.zeros(
            (num_voxels,),
            dtype=points.dtype,
            device=points.device,
        )

        out.index_add_(0, inverse, points)
        counts.index_add_(
            0,
            inverse,
            torch.ones((points.shape[0],), dtype=points.dtype, device=points.device),
        )

        out /= counts.clamp_min(1).unsqueeze(1)
        return out

    # endregion

    # region 过滤与半径离群
    def select_by_mask(self, mask: torch.Tensor | np.ndarray) -> "GPUPointCloud":
        """根据布尔掩码筛选点云。

        Parameters
        ----------
        mask : torch.Tensor | np.ndarray
            形状为 ``(N,)`` 的布尔掩码。

        Returns
        -------
        GPUPointCloud
            筛选后的点云。

        Raises
        ------
        ValueError
            掩码形状不匹配。
        """
        if isinstance(mask, np.ndarray):
            if mask.ndim != 1 or mask.shape[0] != self.N:
                raise ValueError(f"mask 形状必须为 ({self.N},)，实际为 {mask.shape}")
            mask_t = torch.from_numpy(mask.astype(np.bool_, copy=False)).to(self.device)
        else:
            mask_t = mask
            if mask_t.ndim != 1 or mask_t.shape[0] != self.N:
                raise ValueError(f"mask 形状必须为 ({self.N},)，实际为 {tuple(mask_t.shape)}")
            if mask_t.dtype != torch.bool:
                mask_t = mask_t.to(dtype=torch.bool)
            if mask_t.device != self.device:
                mask_t = mask_t.to(self.device)

        return GPUPointCloud(self.tensor[mask_t], device=self.device)

    def radius_outlier_removal(
        self,
        radius: float,
        min_neighbors: int,
        max_nn: int = 64,
        include_self: bool = False,
        batch_size: int | None = None,
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
        progress_log_interval: int = 0,
    ) -> "GPUPointCloud":
        """执行半径离群去除。

        Parameters
        ----------
        radius : float
            邻域半径。
        min_neighbors : int
            最小邻居数阈值。
        max_nn : int, optional
            每点检索的近邻上限。
        include_self : bool, optional
            统计邻居数时是否包含点自身。
        batch_size : int | None, optional
            查询批大小。``None`` 或 ``<=0`` 时自动估算。
        nprobe : int, optional
            IVF nprobe.
        temp_memory_mb : int, optional
            Faiss 临时内存池（MB）。
        train_size : int, optional
            索引训练样本数上限。
        progress_log_interval : int, optional
            进度日志间隔（按 batch 计）。0 表示关闭。

        Returns
        -------
        GPUPointCloud
            过滤后的点云。
        """
        result, _ = self.radius_outlier_removal_with_mask(
            radius=radius,
            min_neighbors=min_neighbors,
            max_nn=max_nn,
            include_self=include_self,
            batch_size=batch_size,
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
            progress_log_interval=progress_log_interval,
        )
        return result

    def radius_outlier_removal_with_mask(
        self,
        radius: float,
        min_neighbors: int,
        max_nn: int = 64,
        include_self: bool = False,
        batch_size: int | None = None,
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
        progress_log_interval: int = 0,
    ) -> tuple["GPUPointCloud", torch.Tensor]:
        """执行半径离群去除并返回保留掩码。

        Parameters
        ----------
        radius : float
            邻域半径。
        min_neighbors : int
            最小邻居数阈值。
        max_nn : int, optional
            每点近邻检索上限。
        include_self : bool, optional
            是否将查询点自身计入邻居数。
        batch_size : int | None, optional
            查询批大小。自动模式由内部策略决定。
        nprobe : int, optional
            IVF nprobe.
        temp_memory_mb : int, optional
            Faiss 临时内存池（MB）。
        train_size : int, optional
            IVF 训练样本数上限。
        progress_log_interval : int, optional
            进度日志间隔（按 batch 计）。

        Returns
        -------
        tuple[GPUPointCloud, torch.Tensor]
            过滤后点云与保留掩码（设备与当前点云一致）。

        Raises
        ------
        ValueError
            参数不合法时抛出。

        Notes
        -----
        该实现为近似快速路径：
        - 使用 IVF 近似索引。
        - 仅按 XYZ 建索引。
        - 超大点云可自动预体素化后再映射回原点集。
        """
        if radius <= 0:
            raise ValueError(f"radius 必须大于 0，实际为 {radius}")
        if min_neighbors < 1:
            raise ValueError(f"min_neighbors 必须大于等于 1，实际为 {min_neighbors}")
        if max_nn < 1:
            raise ValueError(f"max_nn 必须大于等于 1，实际为 {max_nn}")
        if max_nn < min_neighbors:
            raise ValueError(f"max_nn 不应小于 min_neighbors，实际 max_nn={max_nn}, min_neighbors={min_neighbors}")

        work_pcd, inverse = self._prepare_radius_search_source(radius=radius)

        if batch_size is None or batch_size <= 0:
            batch_size = work_pcd._suggest_query_batch_size(max_nn=max_nn)

        keep_mask = work_pcd._compute_radius_keep_mask(
            radius=radius,
            min_neighbors=min_neighbors,
            max_nn=max_nn,
            include_self=include_self,
            batch_size=batch_size,
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
            progress_log_interval=progress_log_interval,
        )

        if inverse is None:
            return self.select_by_mask(keep_mask), keep_mask

        keep_mask = keep_mask[inverse]
        return self.select_by_mask(keep_mask), keep_mask

    def radius_outlier_removal_with_stats(
        self,
        radius: float,
        min_neighbors: int,
        max_nn: int = 64,
        include_self: bool = False,
        batch_size: int | None = None,
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
        progress_log_interval: int = 0,
    ) -> tuple["GPUPointCloud", torch.Tensor, dict[str, float | int | bool]]:
        """执行半径离群去除并返回统计信息。

        Returns
        -------
        tuple[GPUPointCloud, torch.Tensor, dict[str, float | int | bool]]
            过滤结果、保留掩码以及统计字典。
        """
        t0 = time.perf_counter()
        work_pcd, inverse = self._prepare_radius_search_source(radius=radius)
        auto_batch_size = batch_size is None or batch_size <= 0
        final_batch_size = work_pcd._suggest_query_batch_size(max_nn=max_nn) if auto_batch_size else int(batch_size)  # type: ignore

        result, keep_mask = self.radius_outlier_removal_with_mask(
            radius=radius,
            min_neighbors=min_neighbors,
            max_nn=max_nn,
            include_self=include_self,
            batch_size=final_batch_size,
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
            progress_log_interval=progress_log_interval,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        kept_points = int(keep_mask.sum().item())
        input_points = int(self.N)
        source_points = int(work_pcd.N)
        stats: dict[str, float | int | bool] = {
            "input_points": input_points,
            "source_points": source_points,
            "kept_points": kept_points,
            "removed_points": int(max(0, input_points - kept_points)),
            "keep_ratio": float(kept_points / max(1, input_points)),
            "source_ratio": float(source_points / max(1, input_points)),
            "used_prevoxel": bool(inverse is not None),
            "batch_size": int(final_batch_size),
            "batch_size_auto": bool(auto_batch_size),
            "elapsed_ms": float(round(elapsed_ms, 3)),
        }
        return result, keep_mask, stats

    def _prepare_radius_search_source(
        self,
        radius: float,
    ) -> tuple["GPUPointCloud", torch.Tensor | None]:
        """准备半径搜索所用点集。

        Parameters
        ----------
        radius : float
            半径参数，用于决定是否自动预体素化。

        Returns
        -------
        tuple[GPUPointCloud, torch.Tensor | None]
            搜索点云；若启用预体素，则附带逆映射。
        """
        pre_voxel_size = self._auto_prevoxel_size(radius=radius)
        if pre_voxel_size is None:
            return self, None

        down_pcd, inverse = self.get_or_create_voxelized(
            voxel_size=pre_voxel_size,
            keep="centroid",
            xyz_only=False,
        )
        reduction_ratio = float(down_pcd.N) / float(max(1, self.N))
        logger.debug(
            "[半径离群点剔除] 自动预体素化启用 | 体素尺寸 {:.4f} 毫米 | 原始点数 {} 个 | 体素点数 {} 个 | 压缩比例 {:.4f}",
            pre_voxel_size,
            self.N,
            down_pcd.N,
            reduction_ratio,
        )
        return down_pcd, inverse

    def _auto_prevoxel_size(self, radius: float) -> float | None:
        """为超大点云自动估算预体素尺寸。"""
        if self.N < 2_000_000:
            return None

        if radius <= 0:
            return None

        if self.N >= 80_000_000:
            factor = 0.35
            min_voxel = 0.25
        elif self.N >= 20_000_000:
            factor = 0.30
            min_voxel = 0.18
        elif self.N >= 5_000_000:
            factor = 0.25
            min_voxel = 0.12
        else:
            factor = 0.20
            min_voxel = 0.08

        voxel_size = max(min_voxel, float(radius) * factor)
        return float(round(voxel_size, 6))

    def _compute_radius_keep_mask(
        self,
        radius: float,
        min_neighbors: int,
        max_nn: int,
        include_self: bool,
        batch_size: int,
        nprobe: int,
        temp_memory_mb: int,
        train_size: int,
        progress_log_interval: int = 0,
    ) -> torch.Tensor:
        """计算 CPU 侧保留掩码。"""
        gpu_index = self.get_spatial_index(
            nprobe=nprobe,
            temp_memory_mb=temp_memory_mb,
            train_size=train_size,
        )

        query_xyz = self.xyz
        radius2 = float(radius) * float(radius)
        keep_mask = torch.empty((self.N,), dtype=torch.bool, device=self.device)
        total = query_xyz.shape[0]
        total_batches = max((total + batch_size - 1) // batch_size, 1)
        log_every = int(max(0, progress_log_interval))

        for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
            end = min(start + batch_size, total)
            dists2_t, indices_t = gpu_index.search(
                query_xyz[start:end],
                k=max_nn,
                return_torch=True,
            )
            within_radius = dists2_t <= radius2
            if not include_self and within_radius.shape[1] > 0 and self is gpu_index.pcd:
                query_indices = torch.arange(start, end, dtype=indices_t.dtype, device=self.device).unsqueeze(1)
                self_hits = indices_t == query_indices
                within_radius[self_hits] = False
            neighbor_count = within_radius.sum(axis=1)
            keep_mask[start:end] = neighbor_count >= min_neighbors
            if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
                logger.debug(
                    "[半径离群点剔除] keep_mask 进度 批次号 {} / {} 批次",
                    batch_idx,
                    total_batches,
                )

        return keep_mask

    def _suggest_query_batch_size(self, max_nn: int) -> int:
        """按邻域规模估算查询批大小。"""
        if self.N <= 1_000_000:
            return self.N
        if max_nn <= 16:
            return 300_000
        if max_nn <= 32:
            return 220_000
        if max_nn <= 64:
            return 120_000
        if max_nn <= 128:
            return 60_000
        return 40_000

    # endregion


# endregion
