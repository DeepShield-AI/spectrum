import time

import numpy as np
import polars as pl
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import Anomaly
from .spectral_residual import spectral_residual, extend_series, average_filter
from ...config import WINDOW_SIZE
from ...utils import device


class SRCNN:
    def __init__(
            self,
            window_size: int = WINDOW_SIZE,
            learn_rate: float = 1e-3,
            epochs: int = 50,
            batch_size: int = 32,
            back=0,
            backaddnum: int = 5,
    ):
        self.window_size = window_size
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.back = back
        self.backaddnum = backaddnum
        self.model = Anomaly(window_size=window_size).to(device())

    def fit(self, dataset: Dataset):
        train_loader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True
        )
        bp_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.SGD(
            bp_parameters, lr=self.learn_rate, momentum=0.9, weight_decay=0.0
        )
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            loop = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{self.epochs}",
                leave=True
            )
            for idx, (values, labels) in loop:
                optimizer.zero_grad()
                values, labels = values.float().to(device()), labels.float().to(device())
                output = self.model(values)
                loss1 = self.loss_function(output, labels)
                loss1.backward()
                train_loss += loss1.to(device()).item()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                loop.set_postfix(loss=train_loss / (idx + 1))
            self.adjust_lr(optimizer, epoch)
        total_time = time.time() - start_time
        print(f"Training time: {total_time:.2f} seconds")

    def predict(self, values: pl.Series) -> pl.Series:
        def modelwork(x):
            with torch.no_grad():
                x = torch.from_numpy(100 * x).float()
                x = torch.unsqueeze(x, 0)
                x = x.to(device())
                output = self.model(x)
            return output.detach().cpu().numpy().reshape(-1)

        step = 1
        window_size = self.window_size
        back = self.back
        backaddnum = self.backaddnum
        length = len(values)
        scores = [0] * (window_size - backaddnum)

        for pt in range(window_size - backaddnum + back + step, length - back, step):
            head = max(0, pt - (window_size - backaddnum))
            tail = min(length, pt)
            wave = np.array(extend_series(values[head:tail + back]))
            mag = spectral_residual(wave)
            # wave_avg = average_filter(mag)
            # Use absolute difference like Microsoft implementation
            # input_data = np.abs(mag - wave_avg) / (wave_avg + 0.01)
            rawout = modelwork(mag)

            for ipt in range(pt - step, pt):
                scores.append(rawout[ipt - head].item())
        scores += [0] * (length - len(scores))

        return pl.Series(scores, dtype=pl.Float64)

    def adjust_lr(self, optimizer: optim.Optimizer, epoch: int):
        base_lr = self.learn_rate
        cur_lr = base_lr * (0.5 ** ((epoch + 10) // 10))
        for param in optimizer.param_groups:
            param["lr"] = cur_lr

    def loss_function(self, x, labels):
        l2_reg = 0.
        l2_weight = 0.
        for W in self.model.parameters():
            l2_reg = l2_reg + W.norm(2)

        weight = torch.ones(labels.shape)
        weight[labels == 1] = self.window_size // 100
        weight = weight.to(device())

        # Use 'sum' reduction as in Microsoft implementation
        BCE = F.binary_cross_entropy(x, labels, weight=weight, reduction="mean")
        return l2_reg * l2_weight + BCE
