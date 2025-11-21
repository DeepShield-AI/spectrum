# Import necessary libraries
import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "legend.frameon": False,
})

def plot_multivariate_timeseries_separate(train_df: pl.DataFrame, test_df: pl.DataFrame, 
                                        dataset_name: str, figsize=(14, 6)):
    """
    绘制训练集和测试集的时间序列数据，每个维度一张图
    使用数据点索引作为横坐标，分别保存训练集和测试集图像
    
    Args:
        train_df: 训练数据DataFrame
        test_df: 测试数据DataFrame
        dataset_name: 数据集名称
        figsize: 图形大小
    """
    
    # 创建输出目录
    train_output_dir = f"../../figures/datasets/{dataset_name}/train"
    test_output_dir = f"../../figures/datasets/{dataset_name}/test"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 获取所有value列
    value_cols = [col for col in train_df.columns if col.startswith("value_")]
    
    # 为每个维度绘制训练集图形
    for i, col in enumerate(value_cols):
        if col not in train_df.columns:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 创建索引作为横坐标
        train_indices = np.arange(len(train_df))
        
        # 绘制训练集（蓝色）
        train_values = train_df[col].to_numpy()
        ax.plot(
            train_indices,
            train_values,
            color="#2E86AB",
            linewidth=1.5,
            alpha=0.8,
            label="Train",
        )
        
        # 设置标题和轴标签
        title = f"{dataset_name} Dataset - Dimension {col.split('_')[1]} (Train)"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Data Point Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        
        # 美化图形
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        
        # 保存训练集图形
        filename = f"{dataset_name}-{col.split('_')[1]}-train.png"
        fig.savefig(os.path.join(train_output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logging.info(f"Saved train plot for {dataset_name} dimension {col.split('_')[1]} as {filename}")
    
    # 为每个维度绘制测试集图形
    for i, col in enumerate(value_cols):
        if col not in test_df.columns:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 创建索引作为横坐标
        test_indices = np.arange(len(test_df))
        
        # 绘制测试集（绿色）
        test_values = test_df[col].to_numpy()
        ax.plot(
            test_indices,
            test_values,
            color="#06A77D",
            linewidth=1.5,
            alpha=0.8,
            label="Test",
        )
        
        # 如果有标签列，标记测试集异常点
        if 'label' in test_df.columns:
            test_labels = test_df["label"].to_numpy()
            test_anomaly_mask = test_labels == 1
            
            if np.sum(test_anomaly_mask) > 0:
                ax.scatter(
                    test_indices[test_anomaly_mask],
                    test_values[test_anomaly_mask],
                    color="#E63946",
                    s=40,
                    marker="o",
                    zorder=5,
                    edgecolors="darkred",
                    linewidths=1.2,
                    label="Anomaly",
                )
        
        # 设置标题和轴标签
        title = f"{dataset_name} Dataset - Dimension {col.split('_')[1]} (Test)"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Data Point Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        
        # 美化图形
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        
        # 保存测试集图形
        filename = f"{dataset_name}-{col.split('_')[1]}-test.png"
        fig.savefig(os.path.join(test_output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logging.info(f"Saved test plot for {dataset_name} dimension {col.split('_')[1]} as {filename}")

def plot_npy_dataset_separate(train_data: np.ndarray, test_data: np.ndarray, 
                            dataset_name: str, figsize=(14, 6)):
    """
    分别绘制.npy格式训练集和测试集的时间序列数据，每个维度一张图
    使用数据点索引作为横坐标，分别保存训练集和测试集图像
    
    Args:
        train_data: 训练数据numpy数组
        test_data: 测试数据numpy数组（最后一列是标签）
        dataset_name: 数据集名称
        figsize: 图形大小
    """
    
    # 创建输出目录
    train_output_dir = f"../../figures/datasets/{dataset_name}/train"
    test_output_dir = f"../../figures/datasets/{dataset_name}/test"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 计算维度数量（不包括标签列）
    n_dims = test_data.shape[1] - 1
    
    # 为每个维度绘制训练集图形
    for dim in range(min(n_dims, train_data.shape[1])):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 创建索引作为横坐标
        train_indices = np.arange(len(train_data))
        
        # 绘制训练集（蓝色）
        ax.plot(
            train_indices,
            train_data[:, dim],
            color="#2E86AB",
            linewidth=1.5,
            alpha=0.8,
            label="Train",
        )
        
        # 设置标题和轴标签
        title = f"{dataset_name} Dataset - Dimension {dim} (Train)"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Data Point Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        
        # 美化图形
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        
        # 保存训练集图形
        filename = f"{dataset_name}-{dim}-train.png"
        fig.savefig(os.path.join(train_output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logging.info(f"Saved train plot for {dataset_name} dimension {dim} as {filename}")
    
    # 为每个维度绘制测试集图形
    for dim in range(n_dims):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 创建索引作为横坐标
        test_indices = np.arange(len(test_data))
        
        # 绘制测试集（绿色）
        ax.plot(
            test_indices,
            test_data[:, dim],
            color="#06A77D",
            linewidth=1.5,
            alpha=0.8,
            label="Test",
        )
        
        # 标记测试集异常点
        test_labels = test_data[:, -1]  # 最后一列是标签
        test_anomaly_mask = test_labels == 1
        
        if np.sum(test_anomaly_mask) > 0:
            ax.scatter(
                test_indices[test_anomaly_mask],
                test_data[test_anomaly_mask, dim],
                color="#E63946",
                s=40,
                marker="o",
                zorder=5,
                edgecolors="darkred",
                linewidths=1.2,
                label="Anomaly",
            )
        
        # 设置标题和轴标签
        title = f"{dataset_name} Dataset - Dimension {dim} (Test)"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Data Point Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        
        # 美化图形
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        
        # 保存测试集图形
        filename = f"{dataset_name}-{dim}-test.png"
        fig.savefig(os.path.join(test_output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logging.info(f"Saved test plot for {dataset_name} dimension {dim} as {filename}")

def visualize_psm_dataset():
    """
    可视化PSM数据集
    """
    dataset_name = "PSM"
    dataset_dir = "../../processed_datasets/PSM"
    
    if not os.path.exists(dataset_dir):
        print(f"处理后的PSM数据集目录 {dataset_dir} 不存在")
        return
    
    train_path = os.path.join(dataset_dir, "psm_train.csv")
    test_path = os.path.join(dataset_dir, "psm_test.csv")
    
    missing_files = []
    for path, name in [(train_path, "训练集"), (test_path, "测试集")]:
        if not os.path.exists(path):
            missing_files.append(name)
            
    if missing_files:
        print(f"处理后的PSM数据集缺少以下文件: {', '.join(missing_files)}")
        return
    
    # 读取训练和测试数据
    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    
    print(f"PSM训练数据形状: {train_df.shape}")
    print(f"PSM测试数据形状: {test_df.shape}")
    
    # 绘制图形
    plot_multivariate_timeseries_separate(train_df, test_df, dataset_name)

def visualize_swat_dataset():
    """
    可视化SWAT数据集
    """
    dataset_name = "SWAT"
    dataset_dir = "../../processed_datasets/SWAT"
    
    if not os.path.exists(dataset_dir):
        print(f"处理后的SWAT数据集目录 {dataset_dir} 不存在")
        return
    
    train_path = os.path.join(dataset_dir, "swat_train.csv")
    test_path = os.path.join(dataset_dir, "swat_test.csv")
    
    missing_files = []
    for path, name in [(train_path, "训练集"), (test_path, "测试集")]:
        if not os.path.exists(path):
            missing_files.append(name)
            
    if missing_files:
        print(f"处理后的SWAT数据集缺少以下文件: {', '.join(missing_files)}")
        return
    
    # 读取训练和测试数据
    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    
    print(f"SWAT训练数据形状: {train_df.shape}")
    print(f"SWAT测试数据形状: {test_df.shape}")
    
    # 绘制图形
    plot_multivariate_timeseries_separate(train_df, test_df, dataset_name)

def visualize_npy_dataset(dataset_name):
    """
    可视化.npy格式的数据集(MSL, SMD, SMAP)
    """
    dataset_dir = f"../../processed_datasets/{dataset_name}"
    
    if not os.path.exists(dataset_dir):
        print(f"处理后的{dataset_name}数据集目录 {dataset_dir} 不存在")
        return
    
    train_path = os.path.join(dataset_dir, f"{dataset_name}_train.npy")
    test_path = os.path.join(dataset_dir, f"{dataset_name}_test.npy")
    
    missing_files = []
    for path, name in [(train_path, "训练集"), (test_path, "测试集")]:
        if not os.path.exists(path):
            missing_files.append(name)
            
    if missing_files:
        print(f"处理后的{dataset_name}数据集缺少以下文件: {', '.join(missing_files)}")
        return
    
    # 加载数据
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    print(f"{dataset_name}训练数据形状: {train_data.shape}")
    print(f"{dataset_name}测试数据形状: {test_data.shape}")
    
    # 绘制图形
    plot_npy_dataset_separate(train_data, test_data, dataset_name)

if __name__ == "__main__":
    # 可视化PSM和SWAT数据集
    visualize_psm_dataset()
    visualize_swat_dataset()
    
    # # 可视化.npy格式的数据集
    for dataset in ['MSL', 'SMD', 'SMAP']:
         visualize_npy_dataset(dataset)