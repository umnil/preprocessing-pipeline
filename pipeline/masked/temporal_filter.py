import numpy as np

from mne.decoding import TemporalFilter  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from typing import List, Tuple


class MaskedTemporalFilter(BaseEstimator, TemporalFilter):
    def __init__(
        self,
        l_freq=None,
        h_freq=None,
        sfreq=1.0,
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        n_jobs=None,
        method="fir",
        iir_params=None,
        fir_window="hamming",
        fir_design="firwin",
        *,
        verbose=None,
        strict_masking=False
    ):
        self.verbose = None
        self.strict_masking: bool = strict_masking
        super().__init__(
            l_freq,
            h_freq,
            sfreq,
            filter_length,
            l_trans_bandwidth,
            h_trans_bandwidth,
            n_jobs,
            method,
            iir_params,
            fir_window,
            fir_design,
            verbose=None,
        )

    def transform(self, x: np.ma.core.MaskedArray) -> np.ma.core.MaskedArray:
        """Filter data along the last dimension and account for masking

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times) or shape (n_channels, n_times)
            The data to be filtered over the last dimension. The channels
            dimension can be zero when passing a 2D array.

        Returns
        -------
        X : array
            The data after filtering.
        """
        input_shape: Tuple = x.shape
        t: int = input_shape[-1]
        x = x.reshape(-1, t)

        if not isinstance(x, np.ma.core.MaskedArray):
            x = super(MaskedTemporalFilter, self).transform(x)
            x = x.reshape(input_shape)
            return x

        # unmask
        list_x: List = [i.data[~i.mask] for i in x]

        # filter
        window_lengths: np.ndarray = np.unique([i.size for i in list_x if i.size != 0])
        if len(window_lengths) == 1 or not self.strict_masking:
            pre_mask = x.mask.copy()
            x = super(MaskedTemporalFilter, self).transform(x.copy())

            # The filter removes masking
            x = np.ma.MaskedArray(x, pre_mask)
        else:
            np_x = self.transform_indiv(list_x, t)
            x = np.ma.masked_invalid(np_x)

        x = x.reshape(input_shape)
        return x

    def transform_indiv(self, x_list: List, t: int) -> np.ndarray:
        """Filter each element in `x_list` in the event
        that each element may have a separate length of
        points

        Parameters
        ----------
        x_list : List
            A list of numpy arrays with data to pass through a filter
        t : int
            The maximum size that the data should be padded to

        Returns
        -------
        np.ndarray
            A stacked NxM array. Data are padded to M
        """
        x_list = [
            np.array([np.nan] * t)
            if len(i) == 0
            else super(MaskedTemporalFilter, self).transform(i)[0]
            for i in x_list
        ]
        x_list = [
            np.pad(i, [0, t - i.size], "constant", constant_values=[np.nan])
            for i in x_list
        ]
        np_x: np.ndarray = np.stack(x_list)
        return np_x
