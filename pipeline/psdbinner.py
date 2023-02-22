import mne  # type: ignore
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
from typing import List, Optional, cast


class PSDBinner(TransformerMixin, BaseEstimator):
    """
    This class takes results from a power spectral density and formats them
    into bins
    """

    def __init__(
        self,
        bins: List[List],
        raw: Optional[mne.Epochs] = None,
        sfreq: Optional[int] = None,
        slen: Optional[int] = None,
        freqs: Optional[np.ndarray] = None,
        window_channel_sizes: Optional[List] = None,
        psd_channels: List[int] = [0, 2],
        method: str = "mean",
        **kwargs,
    ):
        """initialize the binner

        Parameters
        ----------
        bins : List[List]
            A list of pairs of frequency values that define the lower and upper
            bounds of each bin
        raw : Optional[mne.Epochs]
            An instance of mne Epochs that provides the input for determining
            frequency values. This is not needed if sample length and
            frequencies are provided.
        sfreq : Optional[int]
            The sampling frequency of the signal. Must be provided if raw is
            None
        slen : Optional[int]
            The sample length of the signal. Must be provided if raw is None
        freqs : Optional[np.ndarray]
            A list of freqs. Only needed if raw, sfreq, or slen are None
        window_channel_sizes : Optional[List]
            If provided, then freqs is a flattened array of channels. This
            provides the length of each channel within the flattened array
        psd_channels : List[int]
            A list of channel indexes that should be considdered for binning
        method : str
            A method for binning, can be any method associated with numpy
            (e.g., mean, median, max, min)
        **kwargs
            Extra arguments to feed to the multitapering method (e.g., fmax,
            fmin, sfreq)
        """
        self.bins: List[List] = bins
        self._raw: Optional[mne.Epochs] = raw
        self.sfreq: Optional[int] = sfreq
        self._slen: Optional[int] = slen
        self.freqs: np.ndarray
        self.window_channel_sizes: List = window_channel_sizes
        self.psd_channels: List[int] = psd_channels
        self.method: str = method
        self.psd_channels: List = psd_channels

        psds: np.ndarray
        if raw is not None:
            self.raw = raw
        elif (sfreq is not None) and (slen is not None):
            self.slen = slen
        elif freqs is not None:
            freqs = freqs
        else:
            raise Exception(
                "Either the raw parameter musted be supplied or the sfreq, slen "
                "parameters must be"
            )

        if window_channel_sizes is not None:
            self.window_channel_sizes = window_channel_sizes

        self._bin_freq_idxs: List = []
        self._freq_idx: np.ndarray = np.array([])

    def compute_bins(self, psds: np.ndarray) -> np.ndarray:
        """
        Compute the bins for the given frequency set

        Parameters
        ----------
        psds : np.ndarray
            Power Spectral Density values

        Return
        ------
        np.ndarray
            The aggregated frequency bin values
        """
        return np.array(
            [np.__dict__[self.method](psds[bfi]) for bfi in self.bin_freq_idxs]
        )

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "PSDBinner":
        """
        Fit the model. Simply here for convention
        """
        self.input = X
        return self

    def flatten(self, x: List) -> np.ndarray:
        """
        Given a list of channel psd data, flatten the data

        Parameters
        ----------
        x : List
            A list of where len(x) = n_channels

        Returns
        -------
        np.ndarray
            The NxM matrix where M is a concatenated list of psd binned
            features for each channel
        """
        stacked = np.hstack(x)
        sh: List[int] = list(self.input.shape[:-1]) + [stacked.shape[-1]]
        return stacked.reshape(sh)

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        transforms the matrix of raw frequency values into aggregated bins

        Parameters
        ----------
        X : np.ndarray
            The feature matrix of shape (..., n_frequency_features)
        y : Optional[np.ndarray]
            The labels

        Returns
        -------
        np.ndarray
            The transformed matrix (..., n_bins)
        """
        ch_x: List = []
        if self.n_channels > 1:
            ch_x = self.unflatten(X)
        else:
            ch_x = [X]

        ch_x = [
            self.transform_channel(c, i) if i in self.psd_channels else c
            for i, c in enumerate(ch_x)
        ]
        return self.flatten(ch_x)

    def transform_channel(self, X: np.ndarray, ch: int, **kwargs) -> np.ndarray:
        """
        transforms the matrix of raw frequency values into aggregated bins

        Parameters
        ----------
        X : np.ndarray
            The feature matrix of shape (n_samples, n_frequency_features)
        ch : int
            The channel index

        Returns
        -------
        np.ndarray
            The transformed matrix
        """
        if X.shape[-1] != len(self.freqs):
            raise Exception(
                f"PSD dimension is {X.shape[-1]} "
                + f"but was expected to be {len(self.freqs)}"
            )
        return np.apply_along_axis(self.compute_bins, -1, X)

    def unflatten(self, x: np.ndarray) -> List:
        """When the input data is a flattened array of features, this function
        unflattens the array for easier processing

        Parameters
        ----------
        x : np.ndarray
            Matrix to be unflattened

        Retunrs
        -------
        List
            A list of psd data per channel
        """
        wcs: np.ndarray = np.cumsum([0] + self.window_channel_sizes)
        channel_bounds: List = [(i, f) for i, f in zip(wcs[:-1], wcs[1:])]
        return [x[..., i:f] for i, f in channel_bounds]

    @property
    def bin_freq_idxs(self) -> List:
        """Return a list of indicies that point to the frequency indicies
        associated with each bin"""
        self._bin_freq_idxs = [
            self.freq_idx[(self.freqs >= binl) & (self.freqs < binh)]
            for binl, binh in self.bins
        ]
        return self._bin_freq_idxs

    @property
    def freq_idx(self) -> np.ndarray:
        self._freq_idx = np.arange(len(self.freqs))
        return self._freq_idx

    @property
    def n_channels(self) -> int:
        return (
            1 if self.window_channel_sizes is None else len(self.window_channel_sizes)
        )

    @property
    def raw(self) -> Optional[mne.Epochs]:
        return self._raw

    @raw.setter
    def raw(self, r: mne.Epochs):
        self._raw = r
        psds, freqs = mne.time_frequency.psd_multitaper(self._raw.copy(), **kwargs)
        self.freqs = fres

    @property
    def slen(self) -> Optional[int]:
        return self._slen

    @slen.setter
    def slen(self, slen: Optional[int]):
        self._slen = slen
        x: np.ndarray = np.ones((1, slen))
        psds, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=self.sfreq)
        self.freqs = freqs
