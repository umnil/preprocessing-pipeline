import numpy as np

from typing import List, Dict, Optional, cast
from mne.filter import filter_data  # type: ignore
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore


class Filterer(TransformerMixin, BaseEstimator):
    """Transforms data by running through a signal filter"""

    def __init__(
        self,
        window_channel_sizes: List[int] = [640, 16, 640, 16],
        selected_channels: List[int] = [0, 2],
        artifact_threshold: Optional[float] = 10.0,
        filter_args: Dict = {}
    ):
        """Initialize the Filter transformer

        Parameters
        ----------
        window_channel_sizes : List[int]
            This assumes that the input data `X` has rows of data that are
            aggregated channel data. This tells the filter class how many
            channels, and how many data points each channel has per window
        selected_channels : List[int]
            Which channels to include in the filtering
        artifact_detection_threshold : float
            This value is used to detect whether a given window of data has an
            artifact. This is done by checking if the window contains data
            points that are greater than x number of standard deviations away
            from the mean where x is the artifact detection threshold. Windows
            in that are deamed to contain artifacts are excluded from the
            transformed data and stored in the `_artifacts` variable.

            If None, then no artifaction dection will be done
        filter_args : Dict
            Keyword arguments to be passed to the filter
        """
        self.window_channel_sizes: List[int] = window_channel_sizes
        self.selected_channels: List[int] = selected_channels
        self.artifact_threshold: Optional[float] = artifact_threshold
        self.filter_args: Dict = filter_args

        # Unitialized Properties
        self._channel_data: List
        self._X: np.ndarray
        self._y: np.ndarray
        self._X_hat: np.ndarray
        self._y_hat: np.ndarray
        self._artifacts: np.ndarray

    def _extract_channel_data(self, X: np.ndarray, channel_id: int) -> np.ndarray:
        """Given the input data `X`, extract the specified channel data into an
        NxM matrix where N is the number of time epohcs (windows or packets0
        and M is the number of data points for the given epoch)

        Parameters
        ----------
        X : np.ndarray
            NxM array that matches the input
        channel_id : int
            The ID of the channel to select
        """
        pre_channel_sizes: List[int] = self.window_channel_sizes[:channel_id]
        inc_channel_sizes: List[int] = self.window_channel_sizes[: channel_id + 1]
        pre_n_samples: int = sum(pre_channel_sizes)
        inc_n_samples: int = sum(inc_channel_sizes)
        channel_data: np.ndarray = X[:, pre_n_samples:inc_n_samples]
        return channel_data

    def _detect_channel_artifacts(self, X: np.ndarray) -> np.ndarray:
        """Given an a matrix of data from one channel, detect whether there are
        any artifacts in the channel

        Parameters
        ----------
        X : np.ndarray
            An NxM matrix (N = n windows, M = n timepoints)

        Returns
        -------
        np.ndarray
            A vector of True or False values, where True indicates that the
            window was detected to have an artifact
        """
        channel_data: np.ndarray = np.hstack(X.tolist())
        m: int = channel_data.mean()
        s: float = channel_data.std()
        t: float = s * cast(float, self.artifact_threshold)
        upper_limit: float = m + t
        lower_limit: float = m - t
        return ((X > upper_limit) | (X < lower_limit)).any(axis=1)

    def _remove_artifact_windows(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Given a data set of X, detect if any of the selected channels
        contain an artifact. Separate the windows that do in any channel then
        pass the data on

        Parameters
        ----------
        X : np.ndarray
            An NxM data set (N = n windows, M = concatenated windows of data
            for all channels)
        y : np.ndarray
            The Nx1 target lables which will be modified as X is

        Returns
        -------
        np.ndarray
            The data set without the windows that contain artifacts. These
            windows are moved to `self.artifacts`
        """
        n_channels: int = len(self.window_channel_sizes)
        channel_artifacts: np.ndarray = np.array([])
        for channel_idx in range(n_channels):
            channel_data: np.ndarray = self._extract_channel_data(X, channel_idx)

            if channel_idx not in self.selected_channels:
                continue

            cur_channel_artifacts: np.ndarray = self._detect_channel_artifacts(
                channel_data
            )
            channel_artifacts = (
                np.c_[channel_artifacts, cur_channel_artifacts]
                if channel_artifacts.size > 0
                else cur_channel_artifacts
            )

        window_artifact_mask: np.ndarray = channel_artifacts.any(axis=1)
        self._window_artifact_mask: np.ndarray = window_artifact_mask
        self._artifacts = X[window_artifact_mask, :]
        self._y = y[~window_artifact_mask]
        self._y_hat = self._y
        return X[~window_artifact_mask, :]

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> "Filterer":
        """Fit the data... Essentially a place holder

        Parameters
        ----------
        X : np.ndarray
            The input data
        y : np.ndarray
            The target labels

        Returns
        -------
        Filterer
            The same class
        """
        assert X.shape[0] == y.shape[0], ("Input data is not aligned: "
                                          + f"X{X.shape}, y{y.shape}")

        self._y = y
        self._y_hat = y
        return self

    def transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Filter appropriate channel data

        Parameters
        ----------
        X : np.ndarray
            The input matrix

        Returns
        -------
        np.ndarray
            The same data but passed through a filter
        """
        n_channels: int = len(self.window_channel_sizes)
        self._channel_data = [
            self._extract_channel_data(X, ch_idx).tolist()
            for ch_idx in range(n_channels)
        ]

        results: np.ndarray = np.array([])
        for ch_idx in range(n_channels):
            channel_data: np.ndarray = self._channel_data[ch_idx]

            filtered_data: np.ndarray
            if ch_idx in self.selected_channels:
                filtered_data = filter_data(
                    np.array(channel_data).astype(np.float64),
                    verbose="critical",
                    **self.filter_args
                )
            else:
                filtered_data = np.array(channel_data)

            self._channel_data[ch_idx] = filtered_data
            results = (
                np.c_[results, filtered_data]
                if results.shape[0] != 0
                else filtered_data
            )

        self._X = (
            results
            if self.artifact_threshold is None
            else self._remove_artifact_windows(results, self._y)
        )
        self._X_hat = self._X
        return self._X_hat
