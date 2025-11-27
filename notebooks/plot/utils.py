import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl
import seaborn as sns
from joblib import Parallel, delayed

# Shared Style
def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.edgecolor": "0.3",
            "axes.linewidth": 0.8,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "legend.frameon": False,
        }
    )

def _plot_chunk(timestamps_chunk, values_chunk, labels_chunk, dataset_name, split, col_name, segment_index, save_dir):
    """Helper function to plot a single chunk in parallel."""
    # Ensure Agg backend for thread safety and speed in parallel execution
    plt.switch_backend('Agg')
    set_style()
    
    plt.figure(figsize=(10, 4))
    
    # Plot Signal
    plt.plot(timestamps_chunk, values_chunk, color="#2E86AB", linewidth=1.0, alpha=0.8, label="Value")
    
    # Plot Anomalies if available
    if labels_chunk is not None:
        anomaly_mask = labels_chunk == 1
        if anomaly_mask.sum() > 0:
                anom_x = timestamps_chunk[anomaly_mask]
                anom_y = values_chunk[anomaly_mask]
                plt.scatter(anom_x, anom_y, color="#E63946", s=20, marker="x", zorder=5, linewidths=1.5, label="Anomaly")
                
        plt.title(f"{dataset_name} ({split}) | {col_name} | Segment {segment_index} | Anomalies: {anomaly_mask.sum()}")
    else:
        plt.title(f"{dataset_name} ({split}) | {col_name} | Segment {segment_index}")

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(save_dir, f"{col_name}-{segment_index}.png")
    plt.savefig(output_path, dpi=120)
    plt.close()

def plot_dataset_dimensions(df: pl.DataFrame, dataset_name: str, split: str, output_base_dir: str = "../../figures/datasets"):
    """
    Plots each dimension of the dataset in a separate figure and saves it to the specified directory.
    
    Args:
        df: Polars DataFrame containing the data.
        dataset_name: Name of the dataset (e.g., 'synthetic', 'PSM').
        split: 'train' or 'test'.
        output_base_dir: Base directory for saving figures.
    """
    set_style()
    
    # Create output directory
    save_dir = os.path.join(output_base_dir, dataset_name, split)
    os.makedirs(save_dir, exist_ok=True)
    
    # Identify feature columns (value-0, value-1 or value_0, value_1)
    # We assume format value-* or value_*
    value_cols = [col for col in df.columns if col != "timestamp" and col != "label"]
    
    # Try to sort them if they have indices
    try:
        # Extract numbers and sort
        col_indices = []
        for col in value_cols:
            # Handle value-0 and value_0
            separator = "-" if "-" in col else "_"
            idx = int(col.split(separator)[1])
            col_indices.append((idx, col))
        
        col_indices.sort()
        value_cols = [x[1] for x in col_indices]
    except:
        # Fallback if naming is irregular
        value_cols.sort()

    # Timestamps
    timestamps = np.arange(len(df))
    if "timestamp" in df.columns:
        # Only use timestamp column if it is numeric
        if df["timestamp"].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
             timestamps = df["timestamp"].to_numpy()
    
    label_col = "label" if "label" in df.columns else "is_anomaly"
    has_labels = label_col in df.columns
    
    print(f"Plotting {len(value_cols)} dimensions for {dataset_name} ({split})...")
    
    chunk_size = 5000
    total_points = len(df)
    num_chunks = (total_points + chunk_size - 1) // chunk_size

    def task_generator():
        for col_name in value_cols:
            full_values = df[col_name].to_numpy()
            if has_labels:
                full_labels = df[label_col].to_numpy()
            else:
                full_labels = None

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_points)
                
                # Slice data
                timestamps_chunk = timestamps[start_idx:end_idx]
                values_chunk = full_values[start_idx:end_idx]
                labels_chunk = full_labels[start_idx:end_idx] if full_labels is not None else None
                
                yield delayed(_plot_chunk)(
                    timestamps_chunk, values_chunk, labels_chunk,
                    dataset_name, split, col_name, i, save_dir
                )

    # Run in parallel
    Parallel(n_jobs=-1)(task_generator())
        
    print(f"✓ Saved plots for {len(value_cols)} dimensions (split into {num_chunks} segments each) to {save_dir}")
