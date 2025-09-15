import community
import networkx as nx
import time
import numpy as np
import community as community_louvain
from numpy.random import laplace
from sklearn import metrics
import pandas as pd
from utils import *  # 需要包含你的工具函数，如：cal_kl, cal_diam等
import os
import leidenalg as la
import igraph as ig



# 定义主函数
def main_func(dataset_name='CA-HepPh', eps=[0.5, 1, 1.5, 2, 2.5, 3, 3.5], e1_r=1 / 2, e2_r=0, N=20, t=1.0,
              exp_num=10, save_csv=True):
    t_begin = time.time()  # 记录开始时间

    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)  # 构建数据文件路径并读取数据

    # 结果数据框的列名
    cols = ['eps', 'exper', 'nmi', 'evc_overlap', 'evc_MAE', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel']
    all_data = pd.DataFrame(None, columns=cols)  # 创建一个空的DataFrame存储结果

    # 原始图
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print(f'数据集: {dataset_name}')
    print(f'节点数量: {mat0_node}')
    print(f'边数量: {mat0_edge}')

    mat0_par = community_louvain.best_partition(mat0_graph)  # 使用Louvain算法进行初始社区发现

    mat0_degree = np.sum(mat0, 0)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree))  # 度分布

    mat0_evc = nx.eigenvector_centrality(mat0_graph, max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(), key=lambda x: x[1], reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    evc_kn = np.int64(0.01 * mat0_node)  # 选择前1%节点作为特征

    mat0_diam = cal_diam(mat0)  # 计算图的直径
    mat0_cc = nx.transitivity(mat0_graph)  # 全局聚类系数
    mat0_mod = community_louvain.modularity(mat0_par, mat0_graph)  # 原图的模块度

    # 存储每个epsilon值的结果
    all_deg_kl = []
    all_mod_rel = []
    all_nmi_arr = []
    all_evc_overlap = []
    all_evc_MAE = []
    all_cc_rel = []
    all_diam_rel = []

    for ei, epsilon in enumerate(eps):
        ti = time.time()

        e1 = e1_r * epsilon
        e2 = e2_r * epsilon
        e3_r = 1 - e1_r - e2_r
        e3 = e3_r * epsilon

        ed = e3
        ev = e3
        ev_lambda = 1 / ed
        dd_lam = 2 / ev

        # 每个实验的结果存储数组
        nmi_arr = np.zeros([exp_num])
        deg_kl_arr = np.zeros([exp_num])
        mod_rel_arr = np.zeros([exp_num])
        cc_rel_arr = np.zeros([exp_num])
        diam_rel_arr = np.zeros([exp_num])
        evc_overlap_arr = np.zeros([exp_num])
        evc_MAE_arr = np.zeros([exp_num])

        for exper in range(exp_num):
            print(f'-----------epsilon={epsilon:.1f}, exper={exper + 1}/{exp_num}-------------')

            t1 = time.time()

            # 使用Leiden算法进行社区初始化
            mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)


            # 将mat1_pvarr1转换为Leiden算法格式进行社区发现
            mat1_graph = ig.Graph.Adjacency((mat1_pvarr1 > 0).tolist())
            partition = la.find_partition(mat1_graph, la.ModularityVertexPartition)

            # 提取社区结构
            part1 = partition.membership  # 每个节点的社区标签
            mat1_pvarr = np.array(part1)
            mat1_pvs = [np.where(mat1_pvarr == i)[0].tolist() for i in range(max(mat1_pvarr) + 1)]
            comm_n = len(mat1_pvs)

            # 构建边向量矩阵
            ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)
            for i in range(comm_n):
                pi = mat1_pvs[i]
                ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
                for j in range(i + 1, comm_n):
                    pj = mat1_pvs[j]
                    ev_mat[i, j] = np.sum(mat0[np.ix_(pi, pj)])
                    ev_mat[j, i] = ev_mat[i, j]
            # 对边矩阵添加噪声并重构
            ga = get_uptri_arr(ev_mat, ind=1)
            ga_noise = ga
            ga_noise_pp = FO_pp(ga_noise)
            ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

            # 使用Laplace噪声调整度序列
            dd_s = []
            for i in range(comm_n):
                dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
                dd1 = np.sum(dd1, 1)
                dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
                dd1 = FO_pp(dd1)
                dd1[dd1 < 0] = 0
                dd1[dd1 >= len(dd1)] = len(dd1) - 1
                dd_s.append(list(dd1))

            # 使用新社区结构重构图
            mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
            for i in range(comm_n):
                dd_ind = mat1_pvs[i]
                dd1 = dd_s[i]
                mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)
                for j in range(i + 1, comm_n):
                    ev1 = ev_mat[i, j]
                    pj = mat1_pvs[j]
                    if ev1 > 0:
                        c1 = np.random.choice(pi, ev1)
                        c2 = np.random.choice(pj, ev1)
                        for ind in range(ev1):
                            mat2[c1[ind], c2[ind]] = 1
                            mat2[c2[ind], c1[ind]] = 1

            mat2 = mat2 + np.transpose(mat2)
            mat2 = np.triu(mat2, 1)
            mat2 = mat2 + np.transpose(mat2)
            mat2[mat2 > 0] = 1

            mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)

            # 评估新图
            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()
            mat2_par = community_louvain.best_partition(mat2_graph)
            mat2_mod = community_louvain.modularity(mat2_par, mat2_graph)
            mat2_cc = nx.transitivity(mat2_graph)

            mat2_degree = np.sum(mat2, 0)
            mat2_deg_dist = np.bincount(np.int64(mat2_degree))
            mat2_evc = nx.eigenvector_centrality(mat2_graph, max_iter=10000)
            mat2_evc_a = dict(sorted(mat2_evc.items(), key=lambda x: x[1], reverse=True))
            mat2_evc_ak = list(mat2_evc_a.keys())
            mat2_evc_val = np.array(list(mat2_evc_a.values()))
            mat2_diam = cal_diam(mat2)

            # 计算评估指标
            cc_rel = cal_rel(mat0_cc, mat2_cc)
            deg_kl = cal_kl(mat0_deg_dist, mat2_deg_dist)
            mod_rel = cal_rel(mat0_mod, mat2_mod)
            labels_true = list(mat0_par.values())
            labels_pred = list(mat2_par.values())
            nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
            evc_overlap = cal_overlap(mat0_evc_ak, mat2_evc_ak, np.int64(0.01 * mat0_node))
            evc_MAE = cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)
            diam_rel = cal_rel(mat0_diam, mat2_diam)

            # 存储每次实验的结果
            nmi_arr[exper] = nmi
            cc_rel_arr[exper] = cc_rel
            deg_kl_arr[exper] = deg_kl
            mod_rel_arr[exper] = mod_rel
            evc_overlap_arr[exper] = evc_overlap
            evc_MAE_arr[exper] = evc_MAE
            diam_rel_arr[exper] = diam_rel

            print(f'节点数={mat2_node}, 边数={mat2_edge}, nmi={nmi:.4f}, cc_rel={cc_rel:.4f}, deg_kl={deg_kl:.4f}, '
                  f'mod_rel={mod_rel:.4f}, evc_overlap={evc_overlap:.4f}, evc_MAE={evc_MAE:.4f}, diam_rel={diam_rel:.4f}')

            # 将实验数据加入到数据框中
            data_col = [epsilon, exper, nmi, evc_overlap, evc_MAE, deg_kl, diam_rel, cc_rel, mod_rel]
            data1 = pd.DataFrame(np.array(data_col).reshape(1, -1), columns=cols)
            all_data = pd.concat([all_data, data1], ignore_index=True)

        # 计算所有实验的平均结果
        all_nmi_arr.append(np.mean(nmi_arr))
        all_cc_rel.append(np.mean(cc_rel_arr))
        all_deg_kl.append(np.mean(deg_kl_arr))
        all_mod_rel.append(np.mean(mod_rel_arr))
        all_evc_overlap.append(np.mean(evc_overlap_arr))
        all_evc_MAE.append(np.mean(evc_MAE_arr))
        all_diam_rel.append(np.mean(diam_rel_arr))

        print(f'all_index={ei + 1}/{len(eps)} 完成. {time.time() - ti:.2f}s\n')
    # 创建 DataFrame 存储平均结果
    avg_results = pd.DataFrame({
        "epsilon": eps,
        "avg_nmi": all_nmi_arr,
        "avg_evc_overlap": all_evc_overlap,
        "avg_evc_MAE": all_evc_MAE,
        "avg_deg_kl": all_deg_kl,
        "avg_diam_rel": all_diam_rel,
        "avg_cc_rel": all_cc_rel,
        "avg_mod_rel": all_mod_rel
    })

    # 保存路径
    res_path = "./result"
    avg_save_name = f"{res_path}/{dataset_name}_avg_results.csv"

    # 确保目录存在
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # 保存到 CSV 文件
    avg_results.to_csv(avg_save_name, index=False)

    print(f"所有实验的平均结果已保存到: {avg_save_name}")
    # 将结果保存到CSV文件中
    res_path = './result'
    save_name = f'{res_path}/{dataset_name}_{N}_{t:.2f}_{e1_r:.2f}_{e2_r:.2f}_{exp_num}.csv'
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if save_csv:
        all_data.to_csv(save_name, index=False)

    print('-----------------------------')
    print(f'数据集: {dataset_name}')
    print(f'eps={eps}')
    print(f'all_nmi_arr={all_nmi_arr}')
    print(f'all_evc_overlap={all_evc_overlap}')
    print(f'all_evc_MAE={all_evc_MAE}')
    print(f'all_deg_kl={all_deg_kl}')
    print(f'all_diam_rel={all_diam_rel}')
    print(f'all_cc_rel={all_cc_rel}')
    print(f'all_mod_rel={all_mod_rel}')
    print(f'总时间: {time.time() - t_begin:.2f}s')


if __name__ == '__main__':
    # 设置参数
    dataset_name = 'Chamelon'
    eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    e1_r = 1 / 2
    e2_r = 0
    exp_num = 1
    n1 = 20
    t = 1.0

    # 运行主函数
    main_func(dataset_name=dataset_name, eps=eps, e1_r=e1_r, e2_r=e2_r, N=n1, t=t, exp_num=exp_num)
