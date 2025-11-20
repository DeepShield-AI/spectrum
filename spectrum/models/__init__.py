import polars as pl

from .arima import ARIMA
from .lstm import LSTM
from .savae_sr.savae_sr import SaVAE_SR
from .sr import Saliency
from .sr_cnn.srcnn import SRCNN
from ..config import WINDOW_SIZE
from .sr.sr import SpectralResidual

class BaseModel:
    def __init__(
            self,
            window_size: int = WINDOW_SIZE,
    ):
        self.window_size = window_size

    def fit(self, values: pl.Series, labels: pl.Series):
        raise NotImplementedError

    def predict(self, values: pl.Series) -> pl.Series:
        raise NotImplementedError
