import numpy as np
from numpy.random import laplace
import pandas as pd
import networkx as nx
import leidenalg
import igraph as ig
import random
import time
import itertools
from heapq import *
from heapq import nlargest


# 读取数据并生成邻接矩阵
def get_mat(data_path):
    # data_path = './data/' + dataset_name + '.txt'
    data = np.loadtxt(data_path)

    # 初始化统计
    dat = (np.append(data[:, 0], data[:, 1])).astype(int)
    dat_c = np.bincount(dat)

    d = {}
    node = 0
    mid = []
    for i in range(len(dat_c)):
        if dat_c[i] > 0:
            d[i] = node
            mid.append(i)
            node = node + 1
    mid = np.array(mid, dtype=np.int32)

    # 初始化统计
    Edge_num = data.shape[0]
    c = len(d)

    # 生成邻接矩阵
    mat0 = np.zeros([c, c], dtype=np.uint8)
    for i in range(Edge_num):
        mat0[d[int(data[i, 0])], d[int(data[i, 1])]] = 1

    # 将有向图转换为无向图
    mat0 = mat0 + np.transpose(mat0)
    mat0 = np.triu(mat0, 1)
    mat0 = mat0 + np.transpose(mat0)
    mat0[mat0 > 0] = 1
    return mat0, mid


def community_init(mat0, mat0_graph, epsilon, nr, t=1.0):
    # 生成 Laplace 噪声，并添加到邻接矩阵


    # 将噪声添加到原始邻接矩阵
    noisy_mat0 = mat0

    # 将噪声后的邻接矩阵转换为图
    g = ig.Graph.Adjacency((noisy_mat0 > 0).tolist())  # 保证只有正权重边

    # 使用Leiden算法进行社区划分
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

    # 获取每个节点的社区标签
    communities = partition.membership

    # 这里进行聚合度数小于2的点
    aggregated_graph = aggregate_low_degree_nodes(noisy_mat0, communities)
    laplace_noise = np.random.laplace(loc=0.0, scale=1 / epsilon, size=aggregated_graph.shape)
    aggregated_graph = aggregated_graph + laplace_noise

    return aggregated_graph


# 聚合度数小于2的节点
def aggregate_low_degree_nodes(mat0, communities=None, return_mapping=False):
    A = np.asarray(mat0)
    n = A.shape[0]

    # 度数
    deg = A.sum(axis=1)
    low_nodes = np.where(deg < 2)[0]

    # 如果没有低度节点，直接返回
    if len(low_nodes) == 0:
        if return_mapping:
            mapping = np.arange(n)
            groups = {'supernode': [], 'kept': list(range(n))}
            return A.copy(), mapping, groups
        return A.copy()

    keep_nodes = np.setdiff1d(np.arange(n), low_nodes)

    # 超级节点与保留节点的边：取 low_nodes 中是否有人与该保留点相连
    super_edges = (A[low_nodes][:, keep_nodes].sum(axis=0) > 0).astype(int)

    # 新矩阵大小
    m = len(keep_nodes) + 1
    new_mat = np.zeros((m, m), dtype=int)

    # 保留节点之间的边
    new_mat[:-1, :-1] = A[np.ix_(keep_nodes, keep_nodes)]

    # 超级节点连边
    new_mat[-1, :-1] = super_edges
    new_mat[:-1, -1] = super_edges

    # 映射表：旧节点 -> 新索引
    mapping = -np.ones(n, dtype=int)
    mapping[keep_nodes] = np.arange(len(keep_nodes))
    mapping[low_nodes] = m - 1  # 聚合到超级节点

    groups = {'supernode': low_nodes.tolist(), 'kept': keep_nodes.tolist()}

    if return_mapping:
        return new_mat, mapping, groups
    return new_mat


# 获取上三角矩阵的元素
def get_uptri_arr(mat_init, ind=0):
    a = len(mat_init)
    res = []
    for i in range(a):
        dat = mat_init[i][i + ind:]
        res.extend(dat)
    arr = np.array(res)
    return arr


# 生成上三角矩阵
def get_upmat(arr, k, ind=0):
    mat = np.zeros([k, k], dtype=np.int32)
    left = 0
    for i in range(k):
        delta = k - i - ind
        mat[i, i + ind:] = arr[left:left + delta]
        left = left + delta

    return mat


# 后处理
def FO_pp(data_noise, type='norm_sub'):
    if type == 'norm_sub':
        data = norm_sub_deal(data_noise)

    if type == 'norm_mul':
        data = norm_mul_deal(data_noise)

    return data


# 对数据进行归一化处理（减法）
def norm_sub_deal(data):
    data = np.array(data, dtype=np.int32)
    data_min = np.min(data)
    data_sum = np.sum(data)
    delta_m = 0 - data_min

    if delta_m > 0:
        dm = 100000000
        data_seq = np.zeros([len(data)], dtype=np.int32)
        for i in range(0, delta_m):
            data_t = data - i
            data_t[data_t < 0] = 0
            data_t_s = np.sum(data_t)
            dt = np.abs(data_t_s - data_sum)
            if dt < dm:
                dm = dt
                data_seq = data_t
                if dt == 0:
                    break
    else:
        data_seq = data
    return data_seq


# 基于度数序列生成图（内部边）
def generate_intra_edge(dd1, div=1):
    dd1 = np.array(dd1, dtype=np.int32)
    dd1[dd1 < 0] = 0
    dd1_len = len(dd1)
    dd1_p = dd1.reshape(dd1_len, 1) * dd1.reshape(1, dd1_len)
    s1 = np.sum(dd1)

    dd1_res = np.zeros([dd1_len, dd1_len], dtype=np.int8)
    if s1 > 0:
        batch_num = int(dd1_len / div)
        begin_id = 0
        for i in range(div):
            if i == div - 1:
                batch_n = dd1_len - begin_id
                dd1_r = np.random.randint(0, high=s1, size=(batch_n, dd1_len))
                res = dd1_p[begin_id:, :] - dd1_r
                res[res > 0] = 1
                res[res < 1] = 0
                dd1_res[begin_id:, :] = res
            else:
                dd1_r = np.random.randint(0, high=s1, size=(batch_num, dd1_len))
                res = dd1_p[begin_id:begin_id + batch_num, :] - dd1_r
                res[res > 0] = 1
                res[res < 1] = 0
                dd1_res[begin_id:begin_id + batch_num, :] = res
                begin_id = begin_id + batch_num

    # 确保生成的邻接矩阵是对称的
    dd1_out = np.triu(dd1_res, 1)
    dd1_out = dd1_out + np.transpose(dd1_out)
    return dd1_out


# 计算图的直径
def cal_diam(mat):
    mat_graph = nx.from_numpy_array(mat, create_using=nx.Graph)
    max_diam = 0
    for com in nx.connected_components(mat_graph):
        com_list = list(com)
        mat_sub = mat[np.ix_(com_list, com_list)]
        sub_g = nx.from_numpy_array(mat_sub, create_using=nx.Graph)
        diam = nx.diameter(sub_g)
        if diam > max_diam:
            max_diam = diam
    return max_diam


# 计算两个集合的重叠率
def cal_overlap(la, lb, k):
    la = la[:k]
    lb = lb[:k]
    la_s = set(la)
    lb_s = set(lb)
    num = len(la_s & lb_s)
    rate = num / k
    return rate


# 计算KL散度
def cal_kl(A, B):
    p = A / sum(A)
    q = B / sum(B)
    if A.shape[0] > B.shape[0]:
        q = np.pad(q, (0, p.shape[0] - q.shape[0]), 'constant', constant_values=(0, 0))
    elif A.shape[0] < B.shape[0]:
        p = np.pad(p, (0, q.shape[0] - p.shape[0]), 'constant', constant_values=(0, 0))
    kl = p * np.log((p + np.finfo(np.float64).eps) / (q + np.finfo(np.float64).eps))
    kl = np.sum(kl)
    return kl


# 计算相对误差
def cal_rel(A, B):
    eps = 0.000000000000001
    A = np.float64(A)
    B = np.float64(B)
    res = abs((A - B) / (A + eps))
    return res


# 计算均方误差
def cal_MSE(A, B):
    res = np.mean((A - B) ** 2)
    return res


# 计算平均绝对误差
def cal_MAE(A, B, k=None):
    if k == None:
        res = np.mean(abs(A - B))
    else:
        a = np.array(A[:k])
        b = np.array(B[:k])
        res = np.mean(abs(a - b))
    return res


# 写入边数据到文件
def write_edge_txt(mat0, mid, file_name):
    a0 = np.where(mat0 == 1)[0]
    a1 = np.where(mat0 == 1)[1]
    with open(file_name, 'w+') as f:
        for i in range(len(a0)):
            f.write('%d\t%d\n' % (mid[a0[i]], mid[a1[i]]))


# Degree Discount IC模型
def degreeDiscountIC(G, k, p=0.01):
    S = []
    dd = PriorityQueue()  # degree discount
    t = dict()  # 邻居数量
    d = dict()  # 每个节点的度数

    # 初始化degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # 每条边增加度数1
        dd.add_task(u, -d[u])  # 添加每个节点的度数
        t[u] = 0

    # 贪心地添加节点到S中
    for i in range(k):
        u, priority = dd.pop_item()  # 获取最大度数的节点
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']  # 增加邻居节点的选中数量
                priority = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p  # 计算度数折扣
                dd.add_task(v, -priority)
    return S


# 运行IC模型
def runIC(G, S, p=0.01):
    from copy import deepcopy
    from random import random
    T = deepcopy(S)  # 复制已选节点

    i = 0
    while i < len(T):
        for v in G[T[i]]:  # 对已选节点的邻居进行传播
            if v not in T:  # 如果邻居还未选中
                w = G[T[i]][v]['weight']  # 计算边的权重
                if random() <= 1 - (1 - p) ** w:  # 通过边传播影响
                    T.append(v)
        i += 1
    return T


# 找到种子节点
def find_seed(graph_path, seed_size=20):
    # 读取图
    G = nx.Graph()
    with open(graph_path) as f:
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)

    S = degreeDiscountIC(G, seed_size)
    return S


# 计算传播的影响力
def cal_spread(graph_path, S_all, p=0.01, seed_size=20, iterations=100):
    # 读取图
    G = nx.Graph()
    with open(graph_path) as f:
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)

    # 初始化种子集
    if seed_size <= len(S_all):
        S = S_all[:seed_size]
    else:
        print('seed_size is too large.')
        S = S_all

    avg = 0
    for i in range(iterations):
        T = runIC(G, S, p)
        avg += float(len(T)) / iterations

    avg_final = int(round(avg))
    return avg_final
