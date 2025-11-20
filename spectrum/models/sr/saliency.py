import numpy as np
from scipy import stats

from .utils import marge_series, series_filter


def transform_saliency_map_phase(values: np.ndarray) -> np.ndarray:
    """
    Transform a time-series into spectral residual, which is method in computer vision.

    :param values: a list or numpy array of float values.
    :return: saliency map and spectral residual
    """

    freq = np.fft.fft(values)
    phase = np.angle(freq)
    saliency_map = np.fft.ifft(np.exp(1j * phase))
    return saliency_map


def transform_spectral_residual_phase(values):
    saliency_map = transform_saliency_map_phase(values)
    spectral_residual = np.sqrt(saliency_map.real ** 2 + saliency_map.imag ** 2)  # type: ignore
    return spectral_residual


class Saliency(object):
    def __init__(self, amp_window_size: int, series_window_size: int, score_window_size: int):
        self.amp_window_size = amp_window_size
        self.series_window_size = series_window_size
        self.score_window_size = score_window_size

    def transform_saliency_map(self, values: np.ndarray) -> np.ndarray:
        """
        Transform a time-series into spectral residual, which is method in computer vision.
        
        :param values: a list or numpy array of float values.
        :return: saliency map and spectral residual
        """

        freq = np.fft.fft(values)
        magnitude = np.sqrt(freq.real ** 2 + freq.imag ** 2)
        spectral_residual = np.exp(np.log(magnitude) - series_filter(np.log(magnitude), self.amp_window_size))

        freq.real = freq.real * spectral_residual / magnitude
        freq.imag = freq.imag * spectral_residual / magnitude

        saliency_map = np.fft.ifft(freq)
        return saliency_map

    def transform_spectral_residual(self, values):
        """
        Transform a time-series into spectral residual, which is method in computer vision.
        
        :param values: a list or numpy array of float values.
        :return: spectral residual
        """
        saliency_map = self.transform_saliency_map(values)
        spectral_residual = np.sqrt(saliency_map.real ** 2 + saliency_map.imag ** 2)  # type: ignore
        return spectral_residual

    def compute_indicator(self, values):
        sr = self.transform_spectral_residual(values)
        d = (sr - np.mean(sr)) / np.mean(sr)
        indicator = 1 - 1 / (1 + np.exp(-d))
        return indicator

    def generate_anomaly_score(self, values, type="abs"):
        """
        Generate anomaly score by spectral residual.
        :param values:
        :param type:
        :return:
        """

        extended_series = marge_series(values, self.series_window_size, self.series_window_size)
        mag = self.transform_spectral_residual(extended_series)[: len(values)]

        # if type == "avg":
        #     ave_filter = series_filter(mag, self.score_window_size)
        #     score = (mag - ave_filter) / ave_filter
        # elif type == "abs":
        #     ave_filter = series_filter(mag, self.score_window_size)
        #     score = np.abs(mag - ave_filter) / ave_filter
        # elif type == "chisq":
        #     score = stats.chi2.cdf((mag - np.mean(mag)) ** 2 / np.var(mag), df=1)
        # else:
        #     raise ValueError("No type!")
        return mag
