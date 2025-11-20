import os
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from ..config import WINDOW_SIZE
from ..models.sr_cnn.spectral_residual import spectral_residual, average_filter


class YahooDataset(Dataset):
    def __init__(
            self,
            benchmark: str = "A1",
            _id: int = 1,
            path: str = "../../datasets/Yahoo/train/",
            window_size: int = WINDOW_SIZE,
            step: int = 1,
    ):
        path = Path(path) / f"{benchmark}/{_id}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

        ts = pl.read_csv(
            path,
            schema={"timestamp": pl.UInt64, "value": pl.Float64, "label": pl.UInt8},
            truncate_ragged_lines=True,
        )
        scaler = MinMaxScaler()
        scaler.fit(ts['value'].reshape((-1, 1)))
        values = scaler.transform(ts['value'].reshape((-1, 1))).flatten().astype('float32')
        self.window_size = window_size
        self.step = step
        self.values = values
        self.labels = ts['label'].to_numpy()

        n = len(self.values)
        if n < window_size:
            raise IndexError("Empty YahooDataset: no windows available (len(values) < window_size)")
        else:
            self.indices = list(range(0, n - window_size + 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = idx % len(self.indices)
        start = self.indices[idx]
        end = start + self.window_size
        return self.values[start:end], self.labels[start:end]


class SRYaHooDataset(Dataset):
    def __init__(
            self,
            benchmark: str = "A1",
            _id: int = 1,
            path: str = "../../datasets/Yahoo/train/",
            window_size: int = WINDOW_SIZE,
            step: int = WINDOW_SIZE // 2,
    ):
        path = Path(path) / f"{benchmark}/{_id}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

        ts = pl.read_csv(
            path,
            schema={"timestamp": pl.UInt64, "value": pl.Float64, "label": pl.UInt8},
            truncate_ragged_lines=True,
        )
        self.control = 0
        self.window_size = window_size
        self.step = step
        v = []
        l = []
        length = len(ts['value'])

        for pt in range(window_size, length, self.step):
            head = max(0, pt - window_size)
            tail = min(length, pt)
            values = np.array(ts['value'][head:tail], dtype=np.float64)
            values = MinMaxScaler().fit_transform(values.reshape((-1, 1))).flatten().astype('float32') * 3
            num = np.random.randint(1, 10)
            ids = np.random.choice(window_size, num, replace=False)
            # labels = np.array(ts['label'][head:tail], dtype=np.int64)
            labels = np.zeros(self.window_size, dtype=np.int64)
            if (self.window_size - 6) not in ids:
                self.control += np.random.random()
            else:
                self.control = 0
            if self.control > 100:
                ids[0] = self.window_size - 6
                self.control = 0
            mean = np.mean(values)
            data_avg = average_filter(values)
            var = np.var(values)
            for i in ids:
                values[i] += (data_avg[i] + mean) * np.random.randn() * min((1 + var), 10)
                labels[i] = 1
            v.append([values.tolist()])
            l.append([labels.tolist()])
        self.values = np.array(v).squeeze()
        self.labels = np.array(l).squeeze()
        self.length = len(self.values)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        values = self.values[idx]
        labels = self.labels[idx]
        wave = spectral_residual(values)
        wave_avg = average_filter(wave)
        for i in range(self.window_size):
            if wave[i] < 0.001 and wave_avg[i] < 0.001:
                labels[i] = 0
                continue
            ratio = wave[i] / wave_avg[i]
            if ratio < 1.0 and labels[i] == 1:
                labels[i] = 0
            if ratio > 5.0:
                labels[i] = 1
        srscore = abs(wave - wave_avg) / (wave_avg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                labels[idx] = 1
        resdata = torch.from_numpy(100 * abs(wave - wave_avg))
        reslb = torch.from_numpy(labels)
        return resdata, reslb


class LSTMYahooDataset(Dataset):
    def __init__(
            self,
            values,
            window_size: int = WINDOW_SIZE,
            step: int = 1,
    ):
        self.window_size = window_size
        self.step = step
        self.values = values

    def __len__(self):
        return max(len(self.values) - (self.window_size - 1) - self.step, 0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = idx + self.window_size
        x = self.values[idx:end_idx]
        y = self.values[end_idx:end_idx + self.step]
        x = torch.tensor(x).unsqueeze(-1)
        y = torch.tensor(y)
        return x, y
