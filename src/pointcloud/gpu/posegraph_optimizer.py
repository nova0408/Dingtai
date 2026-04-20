import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pypose as pp
import torch
from loguru import logger
from pypose.optim import LevenbergMarquardt


@dataclass(frozen=True)
class PoseGraphEdgeConfig:
    src: int
    dst: int
    edge_type: str
    weight: float


# ==========================================
# 1. 底层 PyPose 优化模型 (通用，无须修改)
# ==========================================
class PoseGraphModule(torch.nn.Module):
    def __init__(self, num_poses, edges_src, edges_dst, edge_measurements, edge_weights):
        super().__init__()
        self.num_poses = num_poses
        self.register_buffer("edges_src", edges_src)
        self.register_buffer("edges_dst", edges_dst)
        self.register_buffer("edge_meas", edge_measurements)
        self.register_buffer("edge_weights", edge_weights)
        self.register_buffer("pose_0", pp.identity_SE3(1))

        # 优化变量：T1-Tn (假设 0 固定)
        self.poses_rest = pp.Parameter(pp.identity_SE3(num_poses - 1))

    def set_initial_guess(self, initial_poses_guess):
        with torch.no_grad():
            self.poses_rest.copy_(initial_poses_guess[1:])

    def forward(self, input=None):
        all_poses = torch.cat([self.pose_0, self.poses_rest])
        src_poses = all_poses[self.edges_src]
        dst_poses = all_poses[self.edges_dst]

        # 预测与误差计算
        pred_rel_pose = src_poses.Inv() @ dst_poses
        diff_pose = self.edge_meas.Inv() @ pred_rel_pose
        error = diff_pose.Log()

        weighted_error = error * self.edge_weights
        return weighted_error.view(-1)


# ==========================================
# 2. 通用高层封装 (重构核心)
# ==========================================
class PoseGraphOptimizer:
    def __init__(self, device: str | None = None, debug: bool = False):
        requested_device = device or str(os.getenv("BG_POSEGRAPH_DEVICE", "cuda"))
        self.device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
        self.debug = bool(debug)
        self._debug("Optimizer initialized on: {}", self.device)

    def _debug(self, message: str, *args: object) -> None:
        if self.debug:
            logger.debug(message, *args)

    def _np_to_se3(self, matrix: np.ndarray) -> pp.LieTensor:
        tensor = torch.from_numpy(matrix).float().to(self.device)
        return pp.mat2SE3(tensor)

    def _se3_to_np(self, pose: pp.LieTensor) -> np.ndarray:
        return pose.matrix().detach().cpu().numpy()

    def _compute_icp(
        self,
        src_cloud: o3d.geometry.PointCloud,
        dst_cloud: o3d.geometry.PointCloud,
        init_global_src: np.ndarray,
        init_global_dst: np.ndarray,
        threshold: float = 5.0,
    ) -> pp.LieTensor:
        """计算两点云间的相对变换"""
        # 计算初始相对猜测 T_src_inv * T_dst
        t_src_inv = np.linalg.inv(init_global_src)
        init_rel = t_src_inv @ init_global_dst

        # 运行 ICP (PointToPlane 精度更高，需有点云法线)
        # 如果没有法线，请改用 TransformationEstimationPointToPoint
        reg = o3d.pipelines.registration.registration_icp(
            src_cloud,
            dst_cloud,
            threshold,
            init_rel,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        return self._np_to_se3(reg.transformation)

    def optimize(
        self,
        chains: list[list[int]],
        pcd_dict: dict[int, o3d.geometry.PointCloud],
        init_poses: dict[int, np.ndarray],
        val_pcd_dict: dict[int, o3d.geometry.PointCloud] | None = None,
        base_weight: float = 1.0,
        loop_weight: float = 100.0,
        icp_threshold: float = 5.0,
    ) -> dict[int, np.ndarray]:
        """
        通用位姿图优化入口

        Args:
            chains: 拓扑链条列表。例如 [[0,1,2...6], [0,7,8...13]]
                    系统会自动连接列表中的相邻元素构建主干边，
                    并自动连接各条链的末端构建闭环边。
            pcd_dict: 用于常规配准的点云字典
            init_poses: 初始位姿字典
            val_pcd_dict: (可选) 用于闭环边的高精度验证点云
            base_weight: 主干边的默认权重
            loop_weight: 闭环边的默认权重
            icp_threshold: ICP 最大对应距离阈值

        Returns:
            优化后的位姿字典
        """

        # --- 1. 动态分析拓扑结构 ---
        edges_config: list[PoseGraphEdgeConfig] = []
        all_nodes = set()

        # A. 构建主干边 (Chain Edges)
        chain_ends = []
        for chain in chains:
            if not chain:
                continue
            chain_ends.append(chain[-1])  # 记录末端用于闭环
            for i in range(len(chain) - 1):
                u, v = chain[i], chain[i + 1]
                all_nodes.add(u)
                all_nodes.add(v)
                edges_config.append(PoseGraphEdgeConfig(src=u, dst=v, edge_type="chain", weight=base_weight))

        # B. 构建闭环边 (Loop Edges) - 连接所有链的末端
        # 策略：如果有 2 条链，连接 end1-end2；如果有 3 条，连接 end1-end2, end2-end3...
        if len(chain_ends) > 1:
            for i in range(len(chain_ends) - 1):
                u, v = chain_ends[i], chain_ends[i + 1]
                edges_config.append(PoseGraphEdgeConfig(src=u, dst=v, edge_type="loop", weight=loop_weight))
                self._debug("闭环边：{} <--> {}", u, v)

        num_poses = max(all_nodes) + 1
        self._debug("图拓扑：{} nodes, {} edges.", len(all_nodes), len(edges_config))

        # --- 2. 计算观测值 ---
        self._debug("开始计算观测值")
        measurements_list: list[pp.LieTensor] = []
        weights_list: list[torch.Tensor] = []
        src_indices: list[int] = []
        dst_indices: list[int] = []

        for edge in edges_config:
            u, v = edge.src, edge.dst
            edge_type = edge.edge_type

            # 关键逻辑修改：
            # 如果是主干边，且我们信任输入位姿 -> 直接数学计算，不跑 ICP
            if edge_type == "chain":
                # 从全局位姿推导相对变换观测值
                # Meas = T_u_inv * T_v
                # 这样初始状态下，主干边的 Residual 刚好为 0
                pose_u = self._np_to_se3(init_poses[u])
                pose_v = self._np_to_se3(init_poses[v])
                meas = pose_u.Inv() @ pose_v

            else:
                # 如果是闭环边 (Loop)，或者我们必须跑 ICP
                # 因为闭环边的初始位姿隐含了巨大的累积误差，不能作为观测值

                # 准备点云
                c_u, c_v = pcd_dict[u], pcd_dict[v]
                if edge_type == "loop" and val_pcd_dict and u in val_pcd_dict and v in val_pcd_dict:
                    self._debug("  -> Loop Closure ICP ({}-{}) using Validation Cloud", u, v)
                    c_u, c_v = val_pcd_dict[u], val_pcd_dict[v]
                else:
                    self._debug("  -> Performing ICP for {} edge ({}-{})", edge_type, u, v)

                meas = self._compute_icp(c_u, c_v, init_poses[u], init_poses[v], threshold=icp_threshold)

            measurements_list.append(meas)
            weights_list.append(torch.full((6,), edge.weight, device=self.device))
            src_indices.append(u)
            dst_indices.append(v)

        # --- 3. 组装 Tensor ---
        edge_meas_tensor = torch.stack(measurements_list).to(self.device)
        edge_weights_tensor = torch.stack(weights_list)
        edges_src = torch.tensor(src_indices, device=self.device, dtype=torch.long)
        edges_dst = torch.tensor(dst_indices, device=self.device, dtype=torch.long)

        # 组装初始值 (按索引顺序填充)
        # 注意：需要处理可能的空洞索引，这里假设索引是连续的 0..N
        # 如果索引不连续，需要建立 map 映射，这里简化处理假设连续
        init_pose_list = []
        for i in range(num_poses):
            if i in init_poses:
                init_pose_list.append(self._np_to_se3(init_poses[i]))
            else:
                # 如果某个中间 ID 缺失，给个单位阵占位 (防止 crash，但在图里是孤立点)
                init_pose_list.append(pp.identity_SE3(1).to(self.device))
        init_pose_tensor = torch.stack(init_pose_list)

        # --- 4. PyPose 优化 ---
        self._debug("开始位姿图优化（非线性最小二乘 Levenberg-Marquardt）")
        model = PoseGraphModule(num_poses, edges_src, edges_dst, edge_meas_tensor, edge_weights_tensor).to(self.device)
        model.set_initial_guess(init_pose_tensor)

        lm = LevenbergMarquardt(model)

        for i in range(20):
            loss = lm.step(input=None)
            if i % 5 == 0:
                self._debug("  Iter {:02d} | Loss: {:.4f}", i, float(loss))
            if loss < 1e-6:
                break

        # --- 5. 导出结果 ---
        final_poses_se3 = torch.cat([model.pose_0, model.poses_rest])
        result = {}
        for i in range(num_poses):
            if i in all_nodes:  # 只返回参与图优化的节点
                result[i] = self._se3_to_np(final_poses_se3[i])

        return result
