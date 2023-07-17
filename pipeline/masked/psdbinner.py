import numpy as np

from ..psdbinner import PSDBinner
from typing import List, Tuple
from sklearn.base import BaseEstimator

class MaskedPSDBinner(PSDBinner,BaseEstimator):
    def transform(  # type: ignore
        self, x: np.ma.core.MaskedArray, *args, **kwargs  # type: ignore
    ) -> np.ndarray:  # type: ignore
        input_shape: Tuple = x.shape
        output_shape: List = list(input_shape[:-1]) + [len(self.bins)]
        x = x.reshape(-1, input_shape[-1])
        list_x = [i[~i.mask] for i in x]
        dt: List = [i.shape[-1] for i in list_x]
        self.freqs = [np.linspace(0, self.sfreq / 2, i) for i in dt]
        freq_idxs: List = [
            [np.where((f >= lo) & (f < h))[0] for lo, h in self.bins]
            for i, f in zip(x, self.freqs)
        ]
        np_x = np.array(
            [
                [self._fn(xi[f], i) for f in fi]
                for i, (xi, fi) in enumerate(zip(x, freq_idxs))
            ]
        )
        np_x = np_x.reshape(output_shape)
        if len(self.select_bins) > 0:
            np_x = np_x[..., self.select_bins]
        return np_x
