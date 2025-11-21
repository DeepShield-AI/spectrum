import numpy as np
import os

def preprocess_npy_dataset(dataset_name):
    """
    预处理.npy格式的数据集，并保存为.npy格式（不添加时间戳列）
    
    Parameters:
    dataset_name (str): 数据集名称 ('MSL', 'SMD', 'SMAP')
    """
    # 构建数据集路径
    dataset_dir = f"../../datasets/{dataset_name}"
    
    # 检查数据集目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"数据集目录 {dataset_dir} 不存在")
        return
    
    # 构建文件路径
    train_path = os.path.join(dataset_dir, f"{dataset_name}_train.npy")
    test_path = os.path.join(dataset_dir, f"{dataset_name}_test.npy")
    test_label_path = os.path.join(dataset_dir, f"{dataset_name}_test_label.npy")
    
    # 检查文件是否存在
    missing_files = []
    for path, name in [(train_path, "训练集"), (test_path, "测试集"), (test_label_path, "测试标签")]:
        if not os.path.exists(path):
            missing_files.append(name)
            
    if missing_files:
        print(f"数据集 {dataset_name} 缺少以下文件: {', '.join(missing_files)}")
        return
    
    # 加载数据
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    test_labels = np.load(test_label_path)
    
    print(f"=== {dataset_name} 数据集信息 ===")
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    
    # 不添加时间戳列，直接使用原始训练数据
    processed_train_data = train_data
    
    print(f"\n处理后训练数据形状: {processed_train_data.shape}")
    
    # 将标签列添加到测试数据的最后一列
    # 确保标签是一列向量
    if test_labels.ndim == 1:
        test_labels = test_labels.reshape(-1, 1)
    processed_test_data = np.hstack([test_data, test_labels])
    
    print(f"处理后测试数据形状: {processed_test_data.shape}")
    
    # 保存预处理后的数据为.npy格式
    output_dir = f"../../processed_datasets/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为.npy格式
    np.save(os.path.join(output_dir, f"{dataset_name}_train.npy"), processed_train_data)
    np.save(os.path.join(output_dir, f"{dataset_name}_test.npy"), processed_test_data)
    
    print(f"\n预处理完成，数据已保存至 {output_dir}")
    print("="*40)
    
    return processed_train_data, processed_test_data

# 处理所有数据集
datasets = ['MSL', 'SMD', 'SMAP']
for dataset in datasets:
    preprocess_npy_dataset(dataset)