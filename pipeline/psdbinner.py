import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import List, Tuple


class PSDBinner(TransformerMixin, BaseEstimator):
    def __init__(self, bins: List, sfreq: int, select_bins: List = []):
        self.bins: List = bins
        self.sfreq: int = sfreq
        self.select_bins: List = select_bins

    def fit(self, *args, **kwargs) -> "PSDBinner":
        return self

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        input_shape: Tuple = x.shape
        output_shape: List = list(input_shape[:-1]) + [len(self.bins)]
        x = x.reshape(-1, input_shape[-1])
        nfreq: int = input_shape[-1]
        freqs: np.ndarray = np.linspace(0, self.sfreq / 2, nfreq)
        freq_idxs: List = [
            [np.where((freqs >= lo) & (freqs < h))[0] for lo, h in self.bins] for i in x
        ]
        x = np.array([[i[f].mean() for f in fi] for i, fi in zip(x, freq_idxs)])
        x = x.reshape(output_shape)
        if len(self.select_bins) > 0:
            x = x[..., self.select_bins]
        return x
