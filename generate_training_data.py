from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import pickle


def load_adj_from_csv(csv_path, num_nodes, adjtype='doubletransition'):
    """
    从distance.csv构建邻接矩阵
    csv格式: from, to, cost
    返回的邻接矩阵使用距离倒数作为权重（越近权重越大）
    """
    dist_df = pd.read_csv(csv_path)
    
    # 构建稀疏邻接矩阵（使用距离倒数作为权重）
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in dist_df.iterrows():
        i, j, w = int(row['from']), int(row['to']), float(row['cost'])
        if w > 0:
            adj[i, j] = 1.0 / w
            adj[j, i] = 1.0 / w  # 无向图
    
    # 对于METR-LA等数据集，部分边可能只单向记录，补全另一方向
    adj = np.maximum(adj, adj.T)
    
    # 处理孤立节点：没有连接的节点使用自连接
    row_sum = adj.sum(axis=1)
    isolated_nodes = row_sum == 0
    if isolated_nodes.any():
        print(f"警告: 发现 {isolated_nodes.sum()} 个孤立节点")
        np.fill_diagonal(adj, 1.0)  # 自连接权重为1
    
    return adj


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, num_nodes, 
        add_time_in_day=True, add_day_in_week=False, 
        time_interval=300  # 5分钟一个时间步
):
    """
    Generate samples from npz/npy data
    :param data: (num_samples, num_nodes, 1) 速度数据
    :param x_offsets: 输入序列的偏移
    :param y_offsets: 输出序列的偏移
    :param num_nodes: 节点数
    :param add_time_in_day: 是否添加日内时间特征
    :param add_day_in_week: 是否添加周几特征
    :param time_interval: 时间间隔（秒），默认5分钟=300秒
    :return:
        x: (num_samples, input_length, num_nodes, input_dim)
        y: (num_samples, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes_data, feature_dim = data.shape
    
    # 确保节点数一致
    assert num_nodes == num_nodes_data, f"节点数不匹配: {num_nodes} vs {num_nodes_data}"
    
    # data: (num_samples, num_nodes, feature_dim) - 直接使用原始速度数据
    feature_list = [data]
    
    if add_time_in_day:
        # 计算一天内的时间编码 (0-1之间)
        time_steps_per_day = 86400 // time_interval  # 288个时间步/天 (5分钟间隔)
        time_in_day = np.tile(
            np.arange(num_samples) % time_steps_per_day / time_steps_per_day,
            [num_nodes, 1]
        ).T.reshape(num_samples, num_nodes, 1).astype(np.float32)
        feature_list.append(time_in_day)
    
    if add_day_in_week:
        # 计算周几特征 (0-6)
        time_steps_per_week = 7 * (86400 // time_interval)
        day_in_week = np.tile(
            np.arange(num_samples) % time_steps_per_week // (86400 // time_interval) % 7,
            [num_nodes, 1]
        ).T.reshape(num_samples, num_nodes, 1).astype(np.float32)
        feature_list.append(day_in_week)
    
    data_with_time = np.concatenate(feature_list, axis=-1)  # (num_samples, num_nodes, feature_dim+1or2)
    
    # 生成序列样本
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    
    for t in range(min_t, max_t):
        x.append(data_with_time[t + x_offsets, ...])
        y.append(data_with_time[t + y_offsets, ...])
    
    x = np.stack(x, axis=0).astype(np.float32)  # (num_samples, input_len, num_nodes, dim)
    y = np.stack(y, axis=0).astype(np.float32)  # (num_samples, output_len, num_nodes, dim)
    
    return x, y


def generate_train_val_test(args):
    """生成训练、验证、测试数据"""
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    
    # 加载npz数据
    print(f"加载数据: {args.data_file}")
    data_npz = np.load(args.data_file, allow_pickle=True)
    data = data_npz['data']  # shape: (num_samples, num_nodes, 1)
    
    num_samples, num_nodes, feature_dim = data.shape
    print(f"数据形状: (时间步={num_samples}, 节点数={num_nodes}, 特征维度={feature_dim})")
    print(f"速度范围: [{data.min():.2f}, {data.max():.2f}] mph")
    
    # 定义时间偏移
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next seq_length_y steps
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    
    # 生成序列数据
    x, y = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        num_nodes=num_nodes,
        add_time_in_day=True,
        add_day_in_week=args.dow,
        time_interval=args.time_interval,
    )
    
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    
    # 划分训练/验证/测试集 (70% / 10% / 20%)
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    print(f"训练集: {x_train.shape[0]} 样本")
    print(f"验证集: {x_val.shape[0]} 样本")
    print(f"测试集: {x_test.shape[0]} 样本")
    
    # 保存训练数据
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        print(f"已保存: {cat}.npz")
    
    # 生成并保存邻接矩阵
    if args.distance_file and os.path.exists(args.distance_file):
        print(f"\n生成邻接矩阵: {args.distance_file}")
        adj = load_adj_from_csv(args.distance_file, num_nodes)
        
        # 保存为pkl格式（兼容原代码）
        sensor_ids = list(range(num_nodes))
        sensor_id_to_ind = {i: i for i in range(num_nodes)}
        
        adj_pkl_path = os.path.join(args.output_dir, 'adj_mx.pkl')
        with open(adj_pkl_path, 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj], f)
        print(f"已保存: adj_mx.pkl (形状: {adj.shape})")
        
        # 同时保存npy格式方便查看
        np.save(os.path.join(args.output_dir, 'adj_mx.npy'), adj)
        
        # 统计邻接矩阵信息
        non_zero = np.count_nonzero(adj)
        print(f"邻接矩阵: {non_zero} 个非零元素, 稀疏度: {1 - non_zero/(num_nodes*num_nodes):.2%}")
    else:
        print("\n警告: 未提供distance.csv，将使用单位矩阵作为邻接矩阵")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/PEMS04", help="输出目录")
    parser.add_argument("--data_file", type=str, required=True, help="npz/npy数据文件路径")
    parser.add_argument("--distance_file", type=str, default=None, help="distance.csv文件路径")
    parser.add_argument("--seq_length_x", type=int, default=12, help="输入序列长度")
    parser.add_argument("--seq_length_y", type=int, default=12, help="输出序列长度")
    parser.add_argument("--y_start", type=int, default=1, help="预测起始偏移")
    parser.add_argument("--time_interval", type=int, default=300, help="时间间隔(秒)，默认5分钟")
    parser.add_argument("--dow", action='store_true', help="是否添加星期特征")
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir):
        reply = str(input(f"{args.output_dir} 已存在，是否覆盖? (y/n): ")).lower().strip()
        if reply[0] != 'y': 
            print("已取消")
            exit()
    else:
        os.makedirs(args.output_dir)
    
    generate_train_val_test(args)
    print("\n数据预处理完成!")
