import polars as pl
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA as _ARIMA

from ..config import WINDOW_SIZE


class ARIMA:
    def __init__(
            self,
            window_size: int = WINDOW_SIZE,
            max_p: int = 5,
            max_q: int = 5,
            max_d: int = 50,
    ):
        self.window_size = window_size
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.p, self.q, self.d = 0, 0, 0
        self.model = None

    def fit(self, values: pl.Series):
        model = auto_arima(
            values.to_numpy(),
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d)
        self.p, self.q, self.d = model.order
        self.model = _ARIMA(values.to_numpy(), order=(self.p, self.q, self.d)).fit()

    def predict(self, values: pl.Series) -> pl.Series:
        values = values.to_numpy()
        residuals = []
        pre = self.model.forecast(self.window_size)
        residuals.extend([abs(values[i] - pre[i]) for i in range(self.window_size)])
        for i in range(self.window_size, len(values)):
            window_data = values[i - self.window_size: i]
            true_y = values[i]

            model = _ARIMA(window_data, order=(self.p, self.d, self.q))
            res = model.fit()

            pred_y = res.forecast()[0]
            residual = abs(true_y - pred_y)
            residuals.append(residual)

        return pl.Series(residuals, dtype=pl.Float64)
