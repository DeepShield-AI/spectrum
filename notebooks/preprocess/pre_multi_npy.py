import polars as pl
import os

def preprocess_psm_dataset():
    """
    处理PSM数据集，按照psm.ipynb中的方式
    """
    # PSM数据集路径
    dataset_dir = "../../datasets/PSM"
    
    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"PSM数据集目录 {dataset_dir} 不存在")
        return
    
    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")
    test_label_path = os.path.join(dataset_dir, "test_label.csv")
    
    # 检查文件是否存在
    missing_files = []
    for path, name in [(train_path, "train.csv"), (test_path, "test.csv"), (test_label_path, "test_label.csv")]:
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        print(f"PSM数据集缺少以下文件: {', '.join(missing_files)}")
        return
    
    # 读取训练数据
    train_df = pl.read_csv(train_path)
    
    # 重命名列: timestamp_(min) -> timestamp, feature_X -> value_X
    old_cols = train_df.columns
    new_cols = []
    for col in old_cols:
        if col == "timestamp_(min)":
            new_cols.append("timestamp")
        elif col.startswith("feature_"):
            # 将"feature_"替换为"value_"
            new_cols.append(col.replace("feature_", "value_"))
        else:
            new_cols.append(col)
    
    train_df = train_df.rename(dict(zip(old_cols, new_cols)))
    
    print("=== PSM 数据集信息 ===")
    print(f"训练数据形状: {train_df.shape}")
    print(f"训练数据列名: {train_df.columns}")
    
    # 读取测试数据
    test_df = pl.read_csv(test_path)
    
    # 应用相同的列重命名
    old_cols = test_df.columns
    new_cols = []
    for col in old_cols:
        if col == "timestamp_(min)":
            new_cols.append("timestamp")
        elif col.startswith("feature_"):
            new_cols.append(col.replace("feature_", "value_"))
        else:
            new_cols.append(col)
    
    test_df = test_df.rename(dict(zip(old_cols, new_cols)))
    
    print(f"测试数据形状: {test_df.shape}")
    print(f"测试数据列名: {test_df.columns}")
    
    # 读取测试标签
    test_label_df = pl.read_csv(test_label_path)
    print(f"测试标签形状: {test_label_df.shape}")
    print(f"测试标签列名: {test_label_df.columns}")
    
    # 将标签列添加到测试数据的最后
    # 假设标签列名为'label'
    if 'label' in test_label_df.columns:
        labels = test_label_df.select('label')
    else:
        # 如果没有'label'列，则使用第二列作为标签
        label_col = test_label_df.columns[1] if len(test_label_df.columns) > 1 else test_label_df.columns[0]
        labels = test_label_df.select(label_col).rename({label_col: 'label'})
    
    # 将标签添加到测试数据的最后一列
    test_with_label_df = test_df.hstack(labels)
    
    print(f"处理后测试数据形状: {test_with_label_df.shape}")
    print(f"处理后测试数据列名: {test_with_label_df.columns}")
    
    # 保存处理后的数据
    output_dir = "../../processed_datasets/PSM"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.write_csv(os.path.join(output_dir, "psm_train.csv"))
    test_with_label_df.write_csv(os.path.join(output_dir, "psm_test.csv"))
    
    print(f"PSM数据集预处理完成，数据已保存至 {output_dir}")
    print("="*50)

def preprocess_swat_dataset():
    """
    处理SWAT数据集，标签已经在测试数据的最后一列，列名与PSM保持一致
    """
    # SWAT数据集路径
    dataset_dir = "../../datasets/SWAT"
    
    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"SWAT数据集目录 {dataset_dir} 不存在")
        return
    
    train_path = os.path.join(dataset_dir, "swat_train2.csv")
    test_path = os.path.join(dataset_dir, "swat2.csv")
    
    # 检查文件是否存在
    missing_files = []
    for path, name in [(train_path, "swat_train2.csv"), (test_path, "swat2.csv")]:
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        print(f"SWAT数据集缺少以下文件: {', '.join(missing_files)}")
        return
    
    # 读取训练数据
    try:
        train_df = pl.read_csv(train_path)
        print("=== SWAT 数据集信息 ===")
        print(f"训练数据形状: {train_df.shape}")
        train_columns=train_df.columns
        train_df=train_df.select(train_columns[:-1])
        # 对训练数据进行列重命名，使其与PSM格式一致
        old_cols = train_df.columns
        new_cols = []
        for i, col in enumerate(old_cols):
            
                # 其他列作为特征值
                new_cols.append(f"value_{i}")
        
        train_df = train_df.rename(dict(zip(old_cols, new_cols)))
        print(f"训练数据列名: {train_df.columns}")
        
        # 读取测试数据
        test_df = pl.read_csv(test_path)
        print(f"测试数据形状: {test_df.shape}")
        
        # 对测试数据进行列重命名，使其与PSM格式一致
        old_cols = test_df.columns
        new_cols = []
        for i, col in enumerate(old_cols):
            if i == 0:
                pass
            else:
                # 其他列作为特征值
                new_cols.append(f"value_{i-1}")
        
        test_df = test_df.rename(dict(zip(old_cols, new_cols)))
        
        # 将测试数据的最后一列重命名为'label'
        old_columns = test_df.columns
        new_columns = list(old_columns[:-1]) + ['label']  # 将最后一列重命名为'label'
        test_df_renamed = test_df.rename(dict(zip(old_columns, new_columns)))
        
        print(f"处理后测试数据形状: {test_df_renamed.shape}")
        print(f"处理后测试数据列名: {test_df_renamed.columns}")
        
        # 保存处理后的数据
        output_dir = "../../processed_datasets/SWAT"
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.write_csv(os.path.join(output_dir, "swat_train.csv"))
        test_df_renamed.write_csv(os.path.join(output_dir, "swat_test.csv"))
        
        print(f"SWAT数据集预处理完成，数据已保存至 {output_dir}")
        
    except Exception as e:
        print(f"处理SWAT数据集时发生错误: {e}")
    
    print("="*50)

# 处理PSM和SWAT数据集
if __name__ == "__main__":
    # preprocess_psm_dataset()
    preprocess_swat_dataset()