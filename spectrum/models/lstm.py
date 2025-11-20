from typing import Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..config import WINDOW_SIZE
from ..dataset import LSTMYahooDataset
from ..utils import EarlyStopping
from ..utils import device


class LSTM:
    def __init__(
            self,
            input_size: int = 1,
            lstm_layers: int = 2,
            window_size: int = WINDOW_SIZE,
            prediction_window_size: int = 1,
            batch_size: int = 32,
            epochs: int = 50,
            split=0.9,
            learning_rate: float = 0.001,
    ):
        self.model = _LSTM(
            input_size=input_size,
            lstm_layers=lstm_layers,
            window_size=window_size,
            prediction_window_size=prediction_window_size,
        ).to(device())
        self.window_size = window_size
        self.prediction_window_size = prediction_window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.split = split
        self.anomaly_scorer = AnomalyScorer()

    def fit(self, values: pl.Series):
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        train_dataloader, valid_dataloader = self._split_data(values.to_numpy())
        early_stopping = EarlyStopping(
            epochs=self.epochs,
        )

        for epoch in early_stopping:
            self.model.train()
            losses = []
            for x, y in train_dataloader:
                self.model.zero_grad()
                loss = self._predict(x, y, criterion)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            self.model.eval()
            valid_losses = []
            for x, y in valid_dataloader:
                loss = self._predict(x, y, criterion)
                valid_losses.append(loss.item())
            validation_loss = sum(valid_losses)
            early_stopping.update(validation_loss)
        self._estimate_normal_distribution(valid_dataloader)

    def _estimate_normal_distribution(self, dl: DataLoader):
        self.model.eval()
        errors = []
        for x, y in dl:
            y_hat = self.model.forward(x)
            e = torch.abs(y.reshape(*y_hat.shape) - y_hat)
            errors.append(e)
        self.anomaly_scorer.find_distribution(torch.cat(errors))

    def _predict(self, x, y, criterion) -> torch.Tensor:
        y = y.reshape(-1, self.prediction_window_size * self.model.hidden_units)
        y_hat = self.model.forward(x)
        loss = criterion(y_hat, y)
        return loss

    def predict(self, values: pl.Series) -> np.ndarray:
        self.model.eval()
        dataloader = DataLoader(
            LSTMYahooDataset(
                values.to_numpy().astype(np.float32),
                window_size=self.window_size,
                step=self.prediction_window_size,
            ),
            batch_size=self.batch_size,
        )
        errors = []
        for x, y in dataloader:
            y_hat = self.model.forward(x)
            e = torch.abs(y.reshape(*y_hat.shape) - y_hat)
            errors.append(e)
        errors = torch.cat(errors)
        return self.anomaly_scorer.forward(errors.mean(dim=1)).detach().numpy()

    def _split_data(self, ts: np.array) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * self.split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = LSTMYahooDataset(train_ts.astype(np.float32), window_size=self.window_size,
                                    step=self.prediction_window_size)
        valid_ds = LSTMYahooDataset(valid_ts.astype(np.float32), window_size=self.window_size,
                                    step=self.prediction_window_size)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=4 * self.batch_size)


class AnomalyScorer:
    def __init__(self):
        super().__init__()

        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        mean_diff = errors - self.mean
        return torch.mul(torch.mul(mean_diff, self.var ** -1), mean_diff)

    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=[0, 1])
        self.var = errors.var(dim=[0, 1])


class _LSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            lstm_layers: int = 2,
            window_size: int = WINDOW_SIZE,
            prediction_window_size: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.window_size = window_size
        self.prediction_length = prediction_window_size
        self.hidden_units = input_size
        self.lstms = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_units * self.prediction_length,
            batch_first=True,
            num_layers=lstm_layers,
        )
        self.dense = nn.Linear(
            in_features=self.window_size * self.hidden_units * self.prediction_length,
            out_features=self.hidden_units * self.prediction_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden = self.lstms(x)
        x = x.reshape(-1, self.window_size * self.hidden_units * self.prediction_length)
        x = self.dense(x)
        return x
