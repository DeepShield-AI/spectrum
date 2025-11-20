import os
from pathlib import Path

import polars as pl
from torch.utils.data import Dataset

from ..config import WINDOW_SIZE


class KPIDataset(Dataset):
    def __init__(
            self,
            kpi_id: str,
            path: str = "./datasets/KPI/train/",
            window_size: int = WINDOW_SIZE,
            step: int = 1,
    ):
        path = Path(path) / f"{kpi_id}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

        ts = pl.read_csv(
            path,
            schema={"timestamp": pl.UInt64, "value": pl.Float64, "label": pl.UInt8},
            truncate_ragged_lines=True,
        )

        self.window_size = window_size
        self.step = step
        self.values = ts["value"]
        self.labels = ts["label"]

        n = len(self.values)
        if n < window_size:
            raise IndexError("Empty KPIDataset: no windows available (len(values) < window_size)")
        else:
            self.indices = list(range(0, n - window_size + 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = idx % len(self.indices)
        start = self.indices[idx]
        end = start + self.window_size

        return self.values[start:end], self.labels[start:end]
