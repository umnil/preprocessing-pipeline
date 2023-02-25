import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import Callable, List, Tuple, Union, cast


class PSDBinner(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        bins: List,
        sfreq: int,
        select_bins: List = [],
        fn: Union[Callable, str] = "mean",
    ):
        self.bins: List = bins
        self.fn: Union[Callable, str] = fn
        self.sfreq: int = sfreq
        self.select_bins: List = select_bins
        self.freqs: Union[List, np.ndarray] = []

    def fit(self, *args, **kwargs) -> "PSDBinner":
        return self

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        input_shape: Tuple = x.shape
        output_shape: List = list(input_shape[:-1]) + [len(self.bins)]
        x = x.reshape(-1, input_shape[-1])
        nfreq: int = input_shape[-1]
        self.freqs = np.linspace(0, self.sfreq / 2, nfreq)
        freq_idxs: List = [
            [np.where((self.freqs >= lo) & (self.freqs < h))[0] for lo, h in self.bins]
            for i in x
        ]
        x = np.array(
            [
                [self._fn(xi[f], i) for f in fi]
                for i, (xi, fi) in enumerate(zip(x, freq_idxs))
            ]
        )
        x = x.reshape(output_shape)
        if len(self.select_bins) > 0:
            x = x[..., self.select_bins]
        return x

    def _fn(self, x: np.ndarray, i: int) -> np.ndarray:
        if type(self.fn) == str:
            return np.__dict__[self.fn](x)
        else:
            f: Callable = cast(Callable, self.fn)
            return f(x, i)
