import polars as pl

from ...config import WINDOW_SIZE
from .saliency import Saliency

import pandas as pd
import numpy as np

class SpectralResidual:
    def __init__(self, window_size=WINDOW_SIZE, mag_window_size: int = 3, score_window_size: int = 40):
        self.window_size = window_size
        self.mag_window_size = mag_window_size
        self.score_window_size = score_window_size

    def predict(self, values: pl.Series) -> pl.Series:
        sr = Saliency(
            amp_window_size=self.mag_window_size,
            series_window_size=self.window_size,
            score_window_size=self.score_window_size)
        return sr.generate_anomaly_score(values)

# EPS = 1e-6


# class SpectralResidual:
#     def __init__(self, threshold: float = 0.3, mag_window: int = 3, window_size: int = WINDOW_SIZE, score_window: int = 40):
#         self.__threshold__ = threshold
#         self.__mag_window = mag_window
#         self.__score_window = score_window
#         self.__anomaly_frame = None
#         self.__batch_size = window_size

#     def predict(self, values: pl.Series):
#         return self.__detect(values)

#     def __detect(self, values: pl.Series):
#         anomaly_frames = []
#         for i in range(0, len(values), self.__batch_size):
#             start = i
#             end = i + self.__batch_size
#             end = min(end, len(values))
#             if end - start >= 12:
#                 scores = self.__detect_core(values[start:end])
#                 anomaly_frames.append(pd.Series(scores))
#             else:
#                 ext_start = max(0, end - self.__batch_size)
#                 ext_frame = self.__detect_core(values[ext_start:end])
#                 anomaly_frames.append(pd.Series(ext_frame[start-ext_start:]))

#         return pd.concat(anomaly_frames, axis=0, ignore_index=True)

#     def __detect_core(self, series):
#         values = series
#         extended_series = SpectralResidual.extend_series(values)
#         mags = self.spectral_residual_transform(extended_series)
#         anomaly_scores = self.generate_spectral_score(mags)
#         return anomaly_scores[:len(values)]

#     def generate_spectral_score(self, mags):
#         ave_mag = average_filter(mags, n=self.__score_window)
#         safeDivisors = np.clip(ave_mag, EPS, ave_mag.max())

#         raw_scores = np.abs(mags - ave_mag) / safeDivisors
#         scores = np.clip(raw_scores / 10.0, 0, 1.0)

#         return scores

#     def spectral_residual_transform(self, values):
#         """
#         This method transform a time series into spectral residual series
#         :param values: list.
#             a list of float values.
#         :return: mag: list.
#             a list of float values as the spectral residual values
#         """

#         trans = np.fft.fft(values)
#         mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
#         eps_index = np.where(mag <= EPS)[0]
#         mag[eps_index] = EPS

#         mag_log = np.log(mag)
#         mag_log[eps_index] = 0

#         spectral = np.exp(mag_log - average_filter(mag_log, n=self.__mag_window))

#         trans.real = trans.real * spectral / mag
#         trans.imag = trans.imag * spectral / mag
#         trans.real[eps_index] = 0
#         trans.imag[eps_index] = 0

#         wave_r = np.fft.ifft(trans)
#         mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
#         return mag

#     @staticmethod
#     def predict_next(values):
#         """
#         Predicts the next value by sum up the slope of the last value with previous values.
#         Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
#         where g(x_i,x_j) = (x_i - x_j) / (i - j)
#         :param values: list.
#             a list of float numbers.
#         :return : float.
#             the predicted next value.
#         """

#         if len(values) <= 1:
#             raise ValueError(f'data should contain at least 2 numbers')

#         v_last = values[-1]
#         n = len(values)

#         slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

#         return values[1] + sum(slopes)

#     @staticmethod
#     def extend_series(values, extend_num=5, look_ahead=5):
#         """
#         extend the array data by the predicted next value
#         :param values: list.
#             a list of float numbers.
#         :param extend_num: int, default 5.
#             number of values added to the back of data.
#         :param look_ahead: int, default 5.
#             number of previous values used in prediction.
#         :return: list.
#             The result array.
#         """

#         if look_ahead < 1:
#             raise ValueError('look_ahead must be at least 1')

#         extension = [SpectralResidual.predict_next(values[-look_ahead - 2:-1])] * extend_num
#         return np.concatenate((values, extension), axis=0)

#     @staticmethod
#     def calculate_expected_value(values, anomaly_index):
#         values = deanomaly_entire(values, anomaly_index)
#         length = len(values)
#         fft_coef = np.fft.fft(values)
#         fft_coef.real = [v if length * 3 / 8 >= i or i >= length * 5 / 8 else 0 for i, v in enumerate(fft_coef.real)]
#         fft_coef.imag = [v if length * 3 / 8 >= i or i >= length * 5 / 8 else 0 for i, v in enumerate(fft_coef.imag)]
#         exps = np.fft.ifft(fft_coef)
#         return exps.real

# def average_filter(values, n=21):
#     """
#     Calculate the sliding window average for the give time series.
#     Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
#     :param values: list.
#         a list of float numbers
#     :param n: int, default 3.
#         window size.
#     :return res: list.
#         a list of value after the average_filter process.
#     """

#     if n >= len(values):
#         n = len(values)

#     res = np.cumsum(values, dtype=float)
#     res[n:] = res[n:] - res[:-n]
#     res[n:] = res[n:] / n

#     for i in range(1, n):
#         res[i] /= i + 1

#     return res
