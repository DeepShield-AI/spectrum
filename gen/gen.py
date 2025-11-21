# generate_synthetic_dataset.py
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any
import os

def generate_base_signal(length: int, frequency: float = 0.01, amplitude: float = 1.0) -> np.ndarray:
    """
    生成基础正弦波信号
    """
    time = np.arange(length)
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    return signal

def add_noise(signal: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    添加高斯噪声
    """
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def add_small_spikes(signal: np.ndarray, spike_percentage: float = 0.05, spike_height: float = 0.5) -> np.ndarray:
    """
    添加小毛刺
    """
    signal_with_spikes = signal.copy()
    num_spikes = int(len(signal) * spike_percentage)
    spike_positions = np.random.choice(len(signal), num_spikes, replace=False)
    
    for pos in spike_positions:

        signal_with_spikes[pos] += spike_height * np.random.uniform(-1.5, 1.5)
    
    return signal_with_spikes

def generate_anomaly_positions(train_length: int, test_length: int, total_length: int, 
                              dimensions: int, anomaly_percentage: float = 0.1):
    """
    生成满足条件的异常位置：
    1. 90%异常点是连续的
    2. 10%异常点是离散的
    3. 同一异常点在至少70%维度上是异常的
    """
    num_anomaly_points = int(test_length * anomaly_percentage)
    
    # 90%连续异常，10%离散异常
    continuous_anomaly_count = int(num_anomaly_points * 0.9)
    discrete_anomaly_count = num_anomaly_points - continuous_anomaly_count
    
    all_anomaly_positions = []
    anomal_start_pos=5
    anomal_end_pos=50
    # 生成连续异常段
    remaining_continuous = continuous_anomaly_count
    while remaining_continuous > 0:
        # 随机选择起始位置（在测试集范围内）
        start_pos = np.random.randint(train_length, total_length - 10)
        # 连续异常段长度
        segment_length = min(np.random.randint(anomal_start_pos, anomal_end_pos), remaining_continuous)
        
        segment = list(range(start_pos, min(start_pos + segment_length, total_length)))
        all_anomaly_positions.extend(segment)
        remaining_continuous -= segment_length
    
    # 生成离散异常点
    available_positions = list(range(train_length, total_length))
    available_positions = [pos for pos in available_positions if pos not in all_anomaly_positions]
    
    if len(available_positions) >= discrete_anomaly_count:
        discrete_positions = np.random.choice(available_positions, discrete_anomaly_count, replace=False)
        all_anomaly_positions.extend(discrete_positions)
    
    # 去重并排序
    all_anomaly_positions = sorted(list(set(all_anomaly_positions)))

    # 转换为测试集内索引
    consolidated_positions = np.array(all_anomaly_positions) - train_length
    
    # 为每个维度分配异常点（确保同一位置至少70%的维度有异常）
    anomaly_positions_per_dim = {}
    
    for dim in range(dimensions):
        # 复制所有异常位置
        dim_anomaly_positions = consolidated_positions.copy()
        
        #随机去掉20%的异常位置
        num_to_remove = int(len(dim_anomaly_positions) * np.random.uniform(0, 0.2))
        if num_to_remove > 0:
            # 随机选择要去掉的位置索引
            remove_indices = np.random.choice(len(dim_anomaly_positions), num_to_remove, replace=False)
            # 使用布尔索引删除选定的位置
            dim_anomaly_positions = np.delete(dim_anomaly_positions, remove_indices)
        
        anomaly_positions_per_dim[dim] = np.sort(dim_anomaly_positions)
    
    # 返回每个维度的异常位置
    return anomaly_positions_per_dim

def add_correlated_anomalies(signal: np.ndarray, anomaly_positions: np.ndarray, 
                            spike_height: float = 25.0) -> np.ndarray:
    """
    在指定位置添加相关异常
    """
    signal_with_anomalies = signal.copy()
    
    for pos in anomaly_positions:
        if 0 <= pos < len(signal):
            # 添加向上的大毛刺，数值范围25-30
            signal_with_anomalies[pos] += spike_height * np.random.uniform(0.5, 1.3)
    
    return signal_with_anomalies

def generate_synthetic_dataset(config_file: str = None):
    """
    生成合成数据集
    """
    # 数据集参数
    anomaly_rate=0.07
    total_length = 10000
    dimensions = 10
    anomaly_base=25.0
    train_length = total_length // 2
    test_length = total_length - train_length
    
    # 生成多维时间序列数据
    data = []
    base_amplitudes=[]
    base_frequencies=[]
    for i in range(dimensions):
        # 频率在0.005到0.02之间随机生成
        frequency = np.random.uniform(0.005, 0.02)
        # 振幅在0.5到1.5之间随机生成
        amplitude = np.random.uniform(0.5, 1.5)
        
        base_frequencies.append(frequency)
        base_amplitudes.append(amplitude)
    
    # 生成满足条件的异常位置
    anomaly_positions_per_dim = generate_anomaly_positions(
        train_length, test_length, total_length, dimensions, anomaly_rate
    )
    
    for dim in range(dimensions):
        # 生成基础信号（值域大约在-1.5到1.5之间）
        base_signal = generate_base_signal(total_length, base_frequencies[dim], base_amplitudes[dim])
        
        # 添加噪声
        noisy_signal = add_noise(base_signal, 0.1)
        
        # 添加小毛刺（整个数据集，值域仍然在-1到1左右）
        signal_with_small_spikes = add_small_spikes(noisy_signal, 0.05, 0.5)
        
        # 添加满足条件的大毛刺（仅测试集部分，值域跳升到20-30）
        final_signal = add_correlated_anomalies(signal_with_small_spikes, 
                                              anomaly_positions_per_dim[dim] + train_length, 
                                              anomaly_base)
        
        data.append(final_signal)
    
    # 转换为numpy数组
    data_array = np.array(data).T  # shape: (10000, 10)
    
    # 创建训练集和测试集
    train_data = data_array[:train_length]  # 前50%作为训练集
    test_data = data_array[train_length:]   # 后50%作为测试集
    
    # 创建标签（训练集全为0，测试集中异常点为1）
    train_labels = np.zeros(train_length)
    
    # 测试集标签初始化为0
    test_labels = np.zeros(test_length)
    
    # 根据记录的异常位置设置测试集标签（只要任一维度在该时间点异常，则标记为异常）
    for dim in range(dimensions):
        anomaly_positions = anomaly_positions_per_dim[dim]
        test_labels[anomaly_positions] = 1
    
    # 保存数据
    output_dir = "../processed_datasets/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集（不包含标签）
    train_df = pd.DataFrame(train_data, columns=[f"value_{i}" for i in range(dimensions)])
    train_df.to_csv(os.path.join(output_dir, "synthetic_train.csv"), index=False)
    
    # 保存测试集（包含标签）
    test_df = pd.DataFrame(test_data, columns=[f"value_{i}" for i in range(dimensions)])
    test_df["label"] = test_labels
    test_df.to_csv(os.path.join(output_dir, "synthetic_test.csv"), index=False)
    
    print(f"数据集生成完成:")
    print(f"  - 总长度: {total_length}")
    print(f"  - 维度数: {dimensions}")
    print(f"  - 训练集形状: {train_data.shape}")
    print(f"  - 测试集形状: {test_data.shape}")
    print(f"  - 正常值范围: 约-1.5到1.5")
    print(f"  - 异常值范围: 约20到30")
    print(f"  - 训练集保存至: {os.path.join(output_dir, 'synthetic_train.csv')}")
    print(f"  - 测试集保存至: {os.path.join(output_dir, 'synthetic_test.csv')}")
    
    # 打印一些统计数据
    normal_min, normal_max = np.min(train_data), np.max(train_data)
    anomaly_min, anomaly_max = np.min(test_data), np.max(test_data)
    print(f"  - 实际正常值范围: {normal_min:.2f} 到 {normal_max:.2f}")
    print(f"  - 实际包含异常值范围: {anomaly_min:.2f} 到 {anomaly_max:.2f}")
    print(f"  - 测试集中异常点数量: {np.sum(test_labels)}")
    
    # 显示异常点分布统计
    unique_anomaly_times = np.sum(test_labels)
    print(f"  - 异常时间点数量: {unique_anomaly_times}")
    
    # 显示各维度异常点数量
    print("  - 各维度异常点数量:")
    for dim in range(dimensions):
        count = len(anomaly_positions_per_dim[dim])
        print(f"    维度{dim}: {count}个异常点")
    
    return train_df, test_df

def visualize_generated_dataset():
    """
    可视化生成的数据集
    """
    import sys
    import os
    # 添加父目录和notebooks/preprocess到Python路径
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_dir = os.path.join(parent_dir, 'notebooks', 'preprocess')
    sys.path.append(preprocess_dir)
    
    from draw import plot_multivariate_timeseries_separate
    import polars as pl
    
    # 读取生成的数据集
    dataset_dir = "../processed_datasets/synthetic"
    train_path = os.path.join(dataset_dir, "synthetic_train.csv")
    test_path = os.path.join(dataset_dir, "synthetic_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("数据集文件不存在，请先运行generate_synthetic_dataset()")
        return
    
    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    
    # 绘制图形
    plot_multivariate_timeseries_separate(train_df, test_df, "synthetic")

if __name__ == "__main__":
    # 生成数据集
    np.random.seed(42)
    train_df, test_df = generate_synthetic_dataset()
    
    # 可视化数据集
    visualize_generated_dataset()