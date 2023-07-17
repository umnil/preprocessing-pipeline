import numpy as np

from mne.decoding import PSDEstimator  # type: ignore
from typing import List, Tuple
from sklearn.base import BaseEstimator

class MaskedPSDEstimator(PSDEstimator,BaseEstimator):
    def transform(self, x: np.ma.core.MaskedArray, **kwargs) -> np.ma.core.MaskedArray:
        input_shape: Tuple = x.shape
        t: int = input_shape[-1]
        df: int = int(t / 2 + 1)
        output_shape: List = list(input_shape[:-1]) + [df]
        x = x.reshape(-1, t)
        list_x: List = [
            super(MaskedPSDEstimator, self).transform(i[~i.mask]) for i in x
        ]
        np_x = np.array([np.concatenate([i, [np.nan] * (df - i.size)]) for i in list_x])
        x = np.ma.masked_invalid(np_x)
        x = x.reshape(output_shape)
        return x
