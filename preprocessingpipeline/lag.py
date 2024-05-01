import numpy as np
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import List, Optional, Tuple


class Lag(TransformerMixin, BaseEstimator):
    def __init__(self, delay: timedelta, sfreq: int, keep_padding: bool = False):
        """Induce a delay between labels and data in timepoints by `delay`
        time. Positive delay values shifts the data forward in time by shifting
        labels backward. Negative delays do the reverse

        Parameters
        ----------
        delay : timedelta
            The delay to induce in the data
        sfreq : int
            Sampling rate of the data
        keep_padding: bool
            If true, the padding added to the ends of the data is kept so that
            the shape of the output is the same as the input. Otherwise it is
            truncated so only valid data is kept.
        """
        self.delay: timedelta = delay
        self.sfreq: int = sfreq
        self.keep_padding: bool = keep_padding

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "Lag":
        return self

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x, y)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Induce lag in the data

        Parameters
        ----------
        x : np.ndarray
            shape (..., n_channels, n_time)
        y : np.ndarray
            shape (..., n_time)

        Returns
        -------
        np.ndarray
            Shifted data
        """
        delay_seconds: float = self.delay.total_seconds()
        shift_forward: bool = delay_seconds >= 0
        n_samples: int = int(np.round(np.abs(delay_seconds) * self.sfreq))
        if n_samples == 0:
            self._y_hat: np.ndarray = y
            return x

        # Pad
        time_pad_width: Tuple = (n_samples, 0) if shift_forward else (0, n_samples)
        label_pad_width: Tuple = (0, n_samples) if shift_forward else (n_samples, 0)
        pad_widths: List = [
            (0, 0) if i != x.ndim - 1 else time_pad_width for i in range(x.ndim)
        ]
        label_pad_widths: List = [
            (0, 0) if i != y.ndim - 1 else label_pad_width for i in range(y.ndim)
        ]
        x_pad: np.ndarray = np.pad(x, pad_widths)
        y_pad: np.ndarray = np.pad(y, label_pad_widths, constant_values=-1)

        # Truncate
        if shift_forward:
            xt: np.ndarray = x_pad[..., :-n_samples]
            yt: np.ndarray = y_pad[..., :-n_samples]
        else:
            xt = x_pad[..., n_samples:]
            yt = y_pad[..., n_samples:]

        # Keep Padding
        if self.keep_padding:
            self._y_hat = yt
            return xt

        if shift_forward:
            xt = xt[..., n_samples:]
            yt = yt[..., n_samples:]
        else:
            xt = xt[..., :-n_samples]
            yt = yt[..., :-n_samples]

        self._y_hat = yt
        return xt
