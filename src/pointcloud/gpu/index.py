from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from typing import TYPE_CHECKING

try:
    import faiss
except Exception:  # pragma: no cover - faiss 可能未安装
    faiss = None  # type: ignore[assignment]

try:
    import faiss.contrib.torch_utils  # type: ignore[attr-defined]  # noqa: F401

    _FAISS_TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于 faiss 构建
    _FAISS_TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from .pointcloud import GPUPointCloud


class GPUSpatialIndex:
    """GPU 空间索引。

    优先使用 Faiss GPU + torch 张量接口；若不可用，则退化为 torch brute-force，
    仍保持检索链路在 GPU 张量内执行，避免 query 的 CPU/NumPy 往返。
    """

    def __init__(
        self,
        gpu_pcd: "GPUPointCloud",
        nprobe: int = 16,
        temp_memory_mb: int = 256,
        train_size: int = 300_000,
    ) -> None:
        self.pcd = gpu_pcd
        self.D = 3
        self.nprobe = int(max(1, nprobe))
        self._temp_memory_bytes = int(max(1, temp_memory_mb)) * 1024 * 1024
        self._data_xyz = self.pcd.xyz[:, :3].contiguous()

        self.res = None
        self.gpu_index = None
        self.nlist = 0
        self.backend = "torch_bruteforce"

        can_use_faiss = (
            faiss is not None
            and _FAISS_TORCH_AVAILABLE
            and self.pcd.device.type == "cuda"
            and torch.cuda.is_available()
        )
        if not can_use_faiss:
            return

        try:
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(self._temp_memory_bytes)

            actual_train_size = min(int(train_size), int(self.pcd.N))
            nlist = self._choose_nlist(n_points=self.pcd.N, actual_train_size=actual_train_size)
            cpu_index = faiss.index_factory(self.D, f"IVF{nlist},Flat")
            gpu_device_id = self._resolve_faiss_gpu_device_id(self.pcd.device)
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, gpu_device_id, cpu_index)

            train_tensor = self._build_train_set(self._data_xyz, train_size=actual_train_size)
            self.gpu_index.train(train_tensor)
            self.gpu_index.add(self._data_xyz)
            self.gpu_index.nprobe = int(max(1, min(self.nprobe, nlist)))
            self.nlist = int(nlist)
            self.backend = "faiss_gpu_torch"
        except Exception as exc:
            logger.debug("Faiss GPU(torch) 初始化失败，退化为 torch brute-force。原因={}", repr(exc))
            self.gpu_index = None
            self.res = None
            self.nlist = 0
            self.backend = "torch_bruteforce"

    def release(self) -> None:
        self.gpu_index = None
        self.res = None

    def search(
        self,
        query_points: np.ndarray | torch.Tensor,
        k: int = 1,
        return_torch: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        if k < 1:
            raise ValueError(f"k 必须大于等于 1，实际为 {k}")

        query_t = self._as_query_tensor(query_points)
        if query_t.shape[0] == 0:
            dists_empty = torch.empty((0, k), device=self.pcd.device, dtype=torch.float32)
            indices_empty = torch.zeros((0, k), device=self.pcd.device, dtype=torch.long)
            if return_torch:
                return dists_empty, indices_empty
            return dists_empty.detach().cpu().numpy(), indices_empty.detach().cpu().numpy()

        if self.backend == "faiss_gpu_torch" and self.gpu_index is not None:
            dists2, indices = self.gpu_index.search(query_t, k)
            if not isinstance(dists2, torch.Tensor):
                dists2 = torch.from_numpy(np.asarray(dists2)).to(self.pcd.device)
            if not isinstance(indices, torch.Tensor):
                indices = torch.from_numpy(np.asarray(indices)).to(self.pcd.device)
            dists2 = dists2.to(dtype=torch.float32, device=self.pcd.device)
            indices = indices.to(dtype=torch.long, device=self.pcd.device)
        else:
            dists2, indices = self._search_torch_bruteforce(query_t, k)

        if torch.any(indices < 0):
            indices = indices.clamp_min(0)

        if return_torch:
            return dists2, indices
        return dists2.detach().cpu().numpy(), indices.detach().cpu().numpy()

    def _search_torch_bruteforce(self, query_t: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_t = self._data_xyz
        m = int(query_t.shape[0])
        n = int(data_t.shape[0])
        k_eff = int(min(max(1, k), max(1, n)))

        # 估算单批次矩阵大小：B * N * 4 字节（float32）
        max_batch = max(1, self._temp_memory_bytes // max(1, n * 4))
        batch = int(min(m, max_batch))
        batch = max(1, batch)

        all_d2: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        for start in range(0, m, batch):
            end = min(start + batch, m)
            q = query_t[start:end]
            # cdist 为欧式距离，平方后与原先 faiss L2 行为一致。
            d = torch.cdist(q, data_t, p=2.0)
            d2 = d * d
            vals, idx = torch.topk(d2, k=k_eff, dim=1, largest=False, sorted=True)
            if k_eff < k:
                pad_n = k - k_eff
                vals = torch.nn.functional.pad(vals, (0, pad_n), value=float("inf"))
                idx = torch.nn.functional.pad(idx, (0, pad_n), value=0)
            all_d2.append(vals)
            all_idx.append(idx)

        return torch.cat(all_d2, dim=0), torch.cat(all_idx, dim=0)

    def _as_query_tensor(self, query_points: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(query_points, torch.Tensor):
            query_t = query_points[..., :3].to(device=self.pcd.device, dtype=torch.float32)
        else:
            arr = np.ascontiguousarray(query_points[..., :3], dtype=np.float32)
            query_t = torch.from_numpy(arr).to(self.pcd.device)
        if query_t.ndim != 2 or query_t.shape[1] < 3:
            raise ValueError(f"query_points 形状必须是 (M, D>=3)，实际为 {tuple(query_t.shape)}")
        return query_t[:, :3].contiguous()

    def _choose_nlist(self, n_points: int, actual_train_size: int) -> int:
        if n_points < 50_000:
            auto_nlist = 64
        elif n_points < 100_000:
            auto_nlist = 128
        elif n_points < 1_000_000:
            auto_nlist = 1024
        elif n_points < 10_000_000:
            auto_nlist = 4096
        elif n_points < 50_000_000:
            auto_nlist = 8192
        else:
            auto_nlist = 16384

        # 经验下限：Faiss 建议训练样本量大约 >= 39 * nlist，避免小样本训练告警。
        max_nlist_by_train = max(1, actual_train_size // 39)
        capped_nlist = self._round_down_power_of_two(max_nlist_by_train)
        return max(32, min(auto_nlist, capped_nlist))

    @staticmethod
    def _round_down_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value.bit_length() - 1)

    def _build_train_set(self, data_xyz: torch.Tensor, train_size: int) -> torch.Tensor:
        if train_size <= 0:
            raise ValueError(f"train_size 必须大于 0，实际为 {train_size}")

        total = int(data_xyz.shape[0])
        actual = min(int(train_size), total)
        if actual == total:
            return data_xyz

        step = max(total // actual, 1)
        return data_xyz[::step][:actual].contiguous()

    @staticmethod
    def _resolve_faiss_gpu_device_id(device: torch.device) -> int:
        if device.type != "cuda":
            raise ValueError(f"GPUSpatialIndex 仅支持 CUDA 设备，实际为 {device}.")
        if device.index is not None:
            return int(device.index)
        return int(torch.cuda.current_device())
