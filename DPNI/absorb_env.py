import gymnasium as gym
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from gymnasium import spaces
from sklearn import metrics
from utils import absorb_low_degree_nodes, add_adaptive_laplace_noise, cal_kl, cal_rel, cal_MAE, cal_overlap, cal_diam
import community as community_louvain


class AbsorbThresholdEnv(gym.Env):
    def __init__(self, mat0, mat0_graph, epsilon=1.0, eval_weights=None):
        super(AbsorbThresholdEnv, self).__init__()
        self.mat0 = mat0
        self.mat0_graph = mat0_graph
        self.epsilon = epsilon
        self.n = mat0.shape[0]
        self.g = ig.Graph.Adjacency((mat0 > 0).tolist())
        self.degrees = np.array(self.g.degree())
        self.eval_weights = eval_weights or {'nmi': 1.0, 'kl': -1.0, 'mae': -1.0}

        # 初始社区划分
        self.mat0_par = community_louvain.best_partition(mat0_graph)
        self.labels_true = list(self.mat0_par.values())

        # 原始图结构参考值
        self.mat0_deg_dist = np.bincount(np.sum(mat0, axis=0).astype(int))
        self.mat0_evc = nx.eigenvector_centrality(mat0_graph, max_iter=10000)
        self.mat0_evc_sorted = sorted(self.mat0_evc.items(), key=lambda x: x[1], reverse=True)
        self.mat0_evc_val = np.array([v for _, v in self.mat0_evc_sorted])
        self.mat0_evc_nodes = [i for i, _ in self.mat0_evc_sorted]
        self.evc_kn = max(1, int(0.01 * self.n))

        # 动作空间：min_degree ∈ [1, 5]
        self.action_space = spaces.Discrete(5)  # 动作 [0, 1, 2, 3, 4] 映射到 min_degree=[1,2,3,4,5]
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # dummy state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0])
        return self.state, {}

    def step(self, action):
        min_deg = int(action) + 1

        g_leiden = ig.Graph.Adjacency((self.mat0 > 0).tolist())
        partition = leidenalg.find_partition(g_leiden, leidenalg.ModularityVertexPartition)
        communities = partition.membership

        mat_absorbed = absorb_low_degree_nodes(self.mat0, communities, min_deg)
        mat_noisy = add_adaptive_laplace_noise(mat_absorbed, self.degrees, self.epsilon)
        mat_noisy = (mat_noisy + mat_noisy.T) / 2
        mat_bin = (mat_noisy > 0.5).astype(int)

        # 重构图
        mat_graph = nx.from_numpy_array(mat_bin, create_using=nx.Graph)
        mat_par = community_louvain.best_partition(mat_graph)
        labels_pred = list(mat_par.values())

        deg_dist = np.bincount(np.sum(mat_bin, axis=0).astype(int))
        evc = nx.eigenvector_centrality(mat_graph, max_iter=10000)
        evc_sorted = sorted(evc.items(), key=lambda x: x[1], reverse=True)
        evc_val = np.array([v for _, v in evc_sorted])
        evc_nodes = [i for i, _ in evc_sorted]

        # 结构指标评估
        nmi = metrics.normalized_mutual_info_score(self.labels_true, labels_pred)
        evc_overlap = cal_overlap(self.mat0_evc_nodes, evc_nodes, self.evc_kn)
        evc_mae = cal_MAE(self.mat0_evc_val, evc_val, self.evc_kn)
        deg_kl = cal_kl(self.mat0_deg_dist, deg_dist)
        diam_rel = cal_rel(cal_diam(self.mat0), cal_diam(mat_bin))
        cc_rel = cal_rel(nx.transitivity(self.mat0_graph), nx.transitivity(mat_graph))
        mod_rel = cal_rel(
            community_louvain.modularity(self.mat0_par, self.mat0_graph),
            community_louvain.modularity(mat_par, mat_graph)
        )

        # 奖励函数：你可以调节权重
        reward = (
                + 1.0 * nmi
                + 1.0 * evc_overlap
                - 1.0 * evc_mae
                - 1.0 * deg_kl
                - 1.0 * diam_rel
                - 1.0 * cc_rel
                - 1.0 * mod_rel
        )

        terminated = True  # 或根据逻辑设置
        truncated = False  # 可根据训练时长限制设置

        info = {
            'min_degree': min_deg,
            'nmi': nmi,
            'evc_overlap': evc_overlap,
            'evc_MAE': evc_mae,
            'deg_kl': deg_kl,
            'diam_rel': diam_rel,
            'cc_rel': cc_rel,
            'mod_rel': mod_rel,
            'reward': reward
        }

        return self.state, reward, terminated, truncated, info

