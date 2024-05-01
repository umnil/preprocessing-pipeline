import numpy as np

from mne.decoding import PSDEstimator  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from typing import List, Tuple


class MaskedPSDEstimator(PSDEstimator, BaseEstimator):
    # super().__init__()
    def __init__(
        self,
        sfreq=2 * np.pi,
        fmin=0,
        fmax=np.inf,
        bandwidth=None,
        adaptive=False,
        low_bias=True,
        n_jobs=None,
        normalization="length",
        *,
        demask: bool = True,
        verbose=None
    ):
        self.verbose = None
        super().__init__(
            sfreq,
            fmin,
            fmax,
            bandwidth,
            adaptive,
            low_bias,
            n_jobs,
            normalization,
            verbose=None,
        )
        self.demask: bool = demask

    def transform(self, x: np.ma.core.MaskedArray, **kwargs) -> np.ma.core.MaskedArray:
        """Transform the time series data into the frequency domain
        using power spectral density

        Parameters
        ----------
        x : np.ma.core.MaskedArray
            The input matrix. Transformed axis is the last
        """
        if not isinstance(x, np.ma.core.MaskedArray):
            return super(MaskedPSDEstimator, self).transform(x, **kwargs)

        if not self.demask:
            input_mask: np.ndarray = x.mask.copy()
            x = super(MaskedPSDEstimator, self).transform(x, **kwargs)
            output_mask: np.ndarray = input_mask[..., : x.shape[-1]]
            x = np.ma.MaskedArray(x, output_mask)
            return x

        input_shape: Tuple = x.shape
        t: int = input_shape[-1]
        df: int = int(t / 2 + 1)
        output_shape: List = list(input_shape[:-1]) + [df]
        x = x.reshape(-1, t)

        # unmaks x
        list_x: List = [i.data[~i.mask] for i in x]

        # transform
        window_lengths: np.ndarray = np.unique([i.size for i in list_x if i.size != 0])
        if len(window_lengths) == 1:
            np_x = super(MaskedPSDEstimator, self).transform(x)
        else:
            list_x = [super(MaskedPSDEstimator, self).transform(i[~i.mask]) for i in x]
            np_x = np.array(
                [np.concatenate([i, [np.nan] * (df - i.size)]) for i in list_x]
            )

        x = np.ma.masked_invalid(np_x)
        x = x.reshape(output_shape)
        return x
