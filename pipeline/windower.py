import logging
import numpy as np

from typing import List
from collections import Counter
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore

LOGGER: logging.Logger = logging.getLogger(__name__)


class Windower(TransformerMixin, BaseEstimator):
    """Provides the blueprint for dividing the data into various widths of
    data
    """

    def __init__(
        self,
        samples_per_window: int = 640,
        label_scheme: int = 0,
        window_step: int = 80,
        trial_size: int = 60000,
    ):
        """

        Parameters
        ----------
        samples_per_window : int
            The number of samples to group into a window of data.
            Ignored if `label_scheme == 4`
        label_scheme : int
            Indicates how the window should be labeled. Labeling schemes are as
            follows

            0 - Window is labeled with the same as the first packet
            1 - Window is labeled with the same label as the last packet
            2 - Window is labeled with the same as the majority of packets in
                    the window
            3 - Non ambiguous lableling, this is the same as 0 except if a
                    window ever has a mix of packets, then it is labeled with NaN
            4 - Windows are made by associated lables (and are not assumed fixed)
        window_step : int
            The number of windows to move forward each step. If
            `window_step == window_size`, then there is no overlapping data in
            each window
            Ignored if `label_scheme == 4`
        trial_size : int
            In case multiple trials were processed prior to receiving the
            input, this trial size value ensures that the data is appropriately
            divided prior to windowing. This is the number of packets expected
            per trial
            Ignored if `label_scheme == 4`
        """
        self.samples_per_window: int = samples_per_window
        self.window_size: int = window_size
        self.window_step: int = window_step
        self.trial_size: int = trial_size
        self.packet_channel_sizes = packet_channel_sizes
        self.packet_size: int = sum(packet_channel_sizes)
        self.label_scheme: int = label_scheme
        self._window_packet_size: int = self.window_size * self.packet_size
        self._window_channel_size: List[int] = [
            self.window_size * x for x in self.packet_channel_sizes
        ]
        self._window_packets: np.ndarray = np.array([])
        self._n_packets: int = 0

        # Uninitialized variables to be defined alter
        self._n_windows: int
        self._y_hat: np.ndarray
        self._y: np.ndarray
        self._X: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "Windower":
        """Uses the data given to determine appropriate window indicies and
        labelling. When running, transform is expected to be passed the same X
        array, otherwise the behavior is undefined

        Parameters
        ----------
        X : np.ndarray
            An NxCxT matrix array of data were N is the number of epochs and C
            is the number of channels and T is the number of data points per
            epoch.
        y : np.ndarray
            An Nx1 matrix array that contains the corresponding labels for X

        Retunrs
        -------
        Windower
            The current instance of the windowing object
        """
        self._y = y
        self._y_hat = self._make_labels(y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the dataframe of packets into a dataframe of windows

        Parameters
        ----------
        X : np.ndarray
            see `fit()`

        Returns
        -------
        np.ndarray
            WxD matrix where W is the number of windows and `D = packet_size *
            window_size`
        """
        result: np.ndarray = np.array([])
        n_channels: int = len(self.packet_channel_sizes)
        for channel_idx in range(n_channels):
            channel_data: np.ndarray = self._get_channel_packets(X, channel_idx)
            windowed_channel_data: np.ndarray = self._transform_channel(channel_data)
            result = (
                np.c_[result, windowed_channel_data]
                if result.shape[0] != 0
                else windowed_channel_data
            )
        self._X = result

        return self._X

    def _make_labels(self, y: np.ndarray) -> np.ndarray:
        """Generate new labels for each window

        Parameters
        ----------
        y : np.ndarray
            An Nx1 input array of the labels to be transformed

        Returns
        -------
        np.ndarray
            an Wx1 input array were W is the number of windows
        """
        self._y = y
        y_transformed: np.ndarray = np.array(
            [
                [
                    y[int(packet_idx)] if not np.isnan(packet_idx) else np.nan
                    for packet_idx in window
                ]
                for window in self.window_packets
            ],
            dtype=object if self.label_scheme == 4 else np.int64,
        )

        if self.label_scheme == 0:
            # Only use the first packet label
            y_transformed = np.array([x[0] for x in y_transformed])
        elif self.label_scheme == 1:
            # Only use the last packet label
            y_transformed = np.array([x[-1] for x in y_transformed])
        elif self.label_scheme == 2:
            # Most common label
            y_transformed = np.array(
                [Counter(x).most_common()[0][0] for x in y_transformed]
            )
        elif self.label_scheme == 3:
            # Non-ambiguous
            y_transformed = np.array(
                [np.nan if len(set(x)) > 1 else x[0] for x in y_transformed]
            )
        elif self.label_scheme == 4:
            n_packets: int = y_transformed.shape[1]
            self._window_channel_size = (
                np.array(self.packet_channel_sizes) * n_packets
            ).tolist()
            y_transformed = y_transformed[:, 0].flatten().astype(np.float64)
        self._n_windows = y_transformed.shape[0]
        return y_transformed

    def _get_channel_packets(self, X: np.ndarray, channel: int) -> np.ndarray:
        """Extracts the packeted data array from the input matrix.

        Parameters
        ----------
        X : np.ndarray
            The input data
        channel : int
            The n-th channel in the data

        Returns
        -------
        np.ndarray
            An NxM matrix. N is the number of data packets and M is the number
            of data points per packet for the specified channel
        """
        pre_channels: List[int] = self.packet_channel_sizes[:channel]
        inc_channels: List[int] = self.packet_channel_sizes[: channel + 1]
        pre_n_points: int = sum(pre_channels)
        inc_n_points: int = sum(inc_channels)
        channel_data: np.ndarray = X[:, pre_n_points:inc_n_points]
        return channel_data

    def _transform_channel(self, ch: np.ndarray) -> np.ndarray:
        """given packets of data for a signle channel, transform the data into
        windowed data

        Parameters
        ----------
        ch : np.ndarray
            an NxM array with N packets and M data points per packet

        Return
        ------
        np.ndarray
            An NxM array with N windows and M data points per window
        """
        result: np.ndarray = np.array([[]])
        packet_channel_size: int = ch.shape[1]
        for window_packet_idxs in self.window_packets:
            selected_packet_idxs: List[int] = [
                int(i) for i in window_packet_idxs if not np.isnan(i)
            ]
            n_nan_packets: int = len([i for i in window_packet_idxs if np.isnan(i)])
            window_packets: np.ndarray = ch[selected_packet_idxs]
            if self.label_scheme == 4:
                nan_data: np.ndarray = (
                    np.ones([n_nan_packets, packet_channel_size]) * self.filler
                )
                window_packets = np.r_[window_packets, nan_data]
            assert window_packets.shape == (
                len(window_packet_idxs),
                packet_channel_size,
            )
            window_data: np.ndarray = np.hstack(window_packets)  # type: ignore
            result = (
                np.append(result, [window_data], axis=0)
                if result.shape[1] > 0
                else np.array([window_data])
            )
        return result

    def get_trial_window_packets(self, trial_idx: int) -> np.ndarray:
        """
        Given a trial index, return a MxN matrix where M is the number of
        windows and N is the number of packets in each window. The values in
        the matrix are essentially an index pointer to `x` pointing to the
        packet belongs to the window

        Parameters
        ----------
        trial_idx : int
            The index of the trial to develop a window packet matrix

        Returns
        -------
        np.ndarray
            The window packet matrix for the given trial index
        """
        packet_idx_limits: np.ndarray = (
            np.array([trial_idx, trial_idx + 1]) * self.trial_size
        )
        base_idx: np.ndarray = np.arange(0, self.trial_size)
        packet_idxs: np.ndarray = np.arange(*packet_idx_limits)

        s: int = 0
        e: int = -(self.window_size - 1)
        window_packets: np.ndarray = np.array(
            [
                packet_idxs[x : x + self.window_size]
                for x in base_idx[s : e : self.window_step]
            ]
        )
        return window_packets

    def get_label_packets(self) -> np.ndarray:
        """
        Returns
        -------
        List
            MxN array where M is the number of windows and N is the number of
            packets in each window, NOTE that this is a ragid matrix. N may be
            different for each M.
        """
        y_diff: np.ndarray = self._y[1:] - self._y[:-1]
        y_change_idxs: np.ndarray = np.where(y_diff != 0)[0] + 1
        y_start_idxs: np.ndarray = np.insert(y_change_idxs, 0, 0)
        y_end_idxs: np.ndarray = np.append(y_change_idxs, len(self._y))
        base_idx: List = list(range(0, self._y.size))
        ragid_matrix: List = [base_idx[s:e] for s, e in zip(y_start_idxs, y_end_idxs)]
        row_max_size: int = max([len(i) for i in ragid_matrix])
        for i, row in enumerate(ragid_matrix):
            diff: int = row_max_size - len(row)
            ragid_matrix[i] = row + [np.nan] * diff

        return np.array(ragid_matrix)

    @property
    def window_packets(self) -> np.ndarray:
        """A list of packet indicies for each window. This is actually where
        the bulk of the logic for windowing is performed. All the other
        matrices are fit to this window packet formation.
        """
        # use cached value if
        if (
            # there is a value that's been cached
            (self._window_packets.size > 0)
            # and the windowing parameters haven't been changed
            and (self._window_packet_size == self.window_size * self.packet_size)
            # and the number of packets to transform hasn't changed
            and (self._n_packets == self._y.shape[0])
        ):
            return self._window_packets

        self._n_packets = self._y.shape[0]
        self.packet_size = sum(self.packet_channel_sizes)
        self._window_packet_size = self.window_size * self.packet_size

        trial_window_packets: List = []
        trial_window_lengths: List = []
        if self.label_scheme < 4:
            n_trials: int = self._n_packets // self.trial_size
            if n_trials == 0:
                LOGGER.warn(
                    f"No trials found, reshifiting trial size to {self._y.shape[0]}"
                )
                self._trial_size_original = self.trial_size
                self.trial_size = self._y.shape[0]
                n_trials = 1
            for trial_idx in range(n_trials):
                window_packets: np.ndarray = self.get_trial_window_packets(trial_idx)
                trial_window_lengths.append(window_packets.shape[0])
                trial_window_packets.append(window_packets)
            self.trial_size = (
                self.trial_size
                if not hasattr(self, "_trail_size_original")
                else self._trial_size_original
            )
        else:
            trial_window_packets = self.get_label_packets().tolist()
            trial_window_lengths = [len(trial_window_packets)]

        self._trial_window_lengths = trial_window_lengths
        self._window_packets = np.vstack(trial_window_packets)
        return self._window_packets
