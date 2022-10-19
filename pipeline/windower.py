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

    def __init__(self,
                 window_size: int = 8,
                 label_scheme: int = 0,
                 window_step: int = 1,
                 trial_size: int = 750,
                 packet_channel_sizes: List[int] = [80, 2, 80, 2]):
        """

        Parameters
        ----------
        window_size : int
            The number of packets to group into a window of data.
        label_scheme : int
            Indicates how the window should be labeled. Labeling schemes are as
            follows

            0 - Window is labeled with the same as the first packet
            1 - Window is labeled with the same label as the last packet
            2 - Window is labeled with the same as the majority of packets in
                    the window
            3 - Non ambiguous lableling, this is the same as 0 except if a
                    window ever has a mix of packets, then it is labeled with NaN
        window_step : int
            The number of windows to move forward each step. If
            `window_step == window_size`, then there is no overlapping data in
            each window
        trial_size : int
            In case multiple trials were processed prior to receiving the
            input, this trial size value ensures that the data is appropriately
            divided prior to windowing. This is the number of packets expected
            per trial
        packet_channel_sizes : List[int]
            This lise has a value for every channel with data in each packet.
            Each element of this list specifies that amount of data each
            channel has stored in a packet (based off it's sampling rate)
        """
        self.window_size: int = window_size
        self.window_step: int = window_step
        self.trial_size: int = trial_size
        self.packet_channel_sizes = packet_channel_sizes
        self.packet_size: int = sum(packet_channel_sizes)
        self.label_scheme: int = label_scheme
        self._window_packet_size: int = self.window_size * self.packet_size
        self._window_channel_size: List[int] = [
            self.window_size * x
            for x in self.packet_channel_sizes
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
            An NxM matrix array of data were N is the number of packets and M
            is the number of data points per packet. M maybe a concatenated
            array of multiple channels as defined by self.packet_channel_sizes
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
            windowed_channel_data: np.ndarray = self._transform_channel(
                channel_data
            )
            result = (
                np.c_[result, windowed_channel_data]
                if result.shape[0] != 0 else windowed_channel_data
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
        y_transformed: np.ndarray = np.array([
            [y[packet_idx] for packet_idx in window]
            for window in self.window_packets
        ])

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
        inc_channels: List[int] = self.packet_channel_sizes[:channel+1]
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
        return np.array([
            np.hstack(ch[packet_idxs])
            for packet_idxs in self.window_packets
        ])

    @property
    def window_packets(self) -> np.ndarray:
        """A list of packet indicies for each window. This is actually were the
        bulk of the logic for windowing is performed. All the other matrices
        are fit to this window packet formation.
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
        n_trials: int = self._n_packets // self.trial_size
        if n_trials == 0:
            LOGGER.warn("No trials, reshifiting")
            self.trial_size = self._y.shape[0]
            n_trials = 1
        trial_window_packets: List = []
        trial_window_lengths: List = []
        for trial_idx in range(n_trials):
            packet_idx_limits: np.ndarray = (
                np.array([trial_idx, trial_idx + 1]) * self.trial_size
            )
            base_idx: np.ndarray = np.arange(0, self.trial_size)
            packet_idxs: np.ndarray = np.arange(*packet_idx_limits)
            window_packets = np.array([
                packet_idxs[x : x + self.window_size]
                for x in base_idx[:-(self.window_size - 1):self.window_step]
            ])
            trial_window_lengths.append(window_packets.shape[0])
            trial_window_packets.append(window_packets)

        self._trial_window_lengths = trial_window_lengths
        self._window_packets = np.vstack(trial_window_packets)
        return self._window_packets
