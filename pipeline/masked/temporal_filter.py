import numpy as np

from mne.decoding import TemporalFilter  # type: ignore
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
#issue area

class MaskedTemporalFilter(TemporalFilter,TransformerMixin, BaseEstimator):
       # super().__init__()
    def __init__(self,l_freq=None, h_freq=None, sfreq=1.0, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=None, method='fir', iir_params=None, fir_window='hamming', fir_design='firwin', *, verbose=None):
        self.verbose = None
        super().__init__(l_freq, h_freq, sfreq, filter_length, l_trans_bandwidth, h_trans_bandwidth, n_jobs, method, iir_params, fir_window, fir_design,verbose=None)       
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
        """  # noqa: E501
        input_shape: Tuple = x.shape
        t: int = input_shape[-1]
        x = x.reshape(-1, t)
        list_x = [super(MaskedTemporalFilter, self).transform(i[~i.mask])[0] for i in x]
        np_x = np.array([np.concatenate([i, [np.nan] * (t - i.size)]) for i in list_x])
        x = np.ma.masked_invalid(np_x)
        x = x.reshape(input_shape)
        return x
