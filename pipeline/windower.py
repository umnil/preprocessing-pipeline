import logging
import numpy as np

from typing import List, Tuple, Union
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
from . import utils

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
        axis: Union[int, List[int]] = 0,
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
        axis : Union[int, List[int]]
            The dimension across which to perform windowing. If a list of axes
            is provided, this transformer functions as a re-windowing function,
            in which the axes in the list are remerged and windowed over.
        """
        self.samples_per_window: int = samples_per_window
        self.window_step: int = window_step
        self.trial_size: int = trial_size
        self.label_scheme: int = label_scheme
        self._window_packets: np.ndarray = np.array([])
        self.axis: Union[int, List[int]] = axis

        # Uninitialized variables to be defined later
        self._n: int
        self._n_windows: int
        self._t: int
        self._y_hat: np.ndarray
        self._y: np.ndarray
        self._X: np.ndarray

    def _make_labels(self, y: np.ndarray) -> np.ndarray:
        """Generate new labels for each window

        Parameters
        ----------
        y : np.ndarray
            An multidimensional input array of the labels to be transformed

        Returns
        -------
        np.ndarray
            an Wx1 input array were W is the number of windows
        """
        t_axis: int = self.axis if isinstance(self.axis, int) else self.axis[0]
        self._n = y.shape[t_axis]
        y, win_idxs = self._window_transform(y, t_axis, True)
        self._window_idxs = win_idxs

        # Apply labelling scheme
        if self.label_scheme == 0:
            # Only use the first packet label
            y_transformed = y[..., 0]
        elif self.label_scheme == 1:
            # Only use the last packet label
            y_transformed = y[..., -1]
        elif self.label_scheme == 2:
            # Most common label
            flattened_y: np.ndarray = y.reshape(-1, self.samples_per_window)
            counts: List = [np.unique(i, return_counts=True) for i in flattened_y]
            flattened_y = np.stack([i[0][np.argmax(i[1])] for i in counts])
            y = flattened_y.reshape(*y.shape[:-1])
            y_transformed = y.squeeze()
        elif self.label_scheme == 3:
            # Non-ambiguous
            y_transformed = np.array([np.nan if len(set(x)) > 1 else x[0] for x in y])
        elif self.label_scheme == 4:
            # Windows by labels
            y = self._y
            if y.ndim > 1:
                y = utils.equalize_list_to_array([self._window_by_label(i) for i in y])
            else:
                y = self._window_by_label(y)
            y_transformed = y

        self._n_windows = y_transformed.shape[0]
        return y_transformed

    def _window_by_label(self, labels: np.ndarray) -> np.ndarray:
        """Create windows over labels such that each window is a unique set of
        values within labels. The order of the windows are as the values appear
        in labels. Labels is expected to be one dimensional

        Parameters
        ----------
        labels : np.ndarray
            The labels to window

        Return
        ------
        np.ndarray
            The 2d windowed array
        """
        assert (
            labels.ndim == 1
        ), "Windowing by label requiers that the labels are 1 dimensional"
        delta: np.ndarray = np.diff(labels)
        transitions: np.ndarray = np.where(delta != 0)[-1] + 1
        delta_idxs: np.ndarray = np.array([0] + transitions.tolist() + [labels.size])
        window_list_idxs: List = [
            np.arange(a, b) for a, b in zip(delta_idxs[:-1], delta_idxs[1:])
        ]
        labels_list: List = [labels[i] for i in window_list_idxs]
        labels = utils.equalize_list_to_array(labels_list)
        return labels[:, 0]

    def _window_transform(
        self, a: np.ndarray, axis: int = 0, return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Performs a windowing transformation on an array across the given
        axis

        Parameters
        ----------
        a : np.ndarray
            The input array to perform the windowing on
        axis : int
            The axis within `a` across which to perform windowing
        return_indices : bool
            If true, returns the indcies used to index the windows from the
            original input array

        Returns
        -------
        np.ndarray
            The windowed array
        """
        # Move working axis to the back
        a = np.moveaxis(a, axis, -1)

        # calculate the number of windows
        n_win: int = int(self._n / self.window_step)
        window_start_idxs = np.linspace(0, self._n, n_win + 1).astype(np.int32)
        window_idxs = np.array(
            [np.arange(s, s + self.samples_per_window) for s in window_start_idxs]
        )
        window_idxs = np.array(
            [idxs for idxs in window_idxs if ~np.any(idxs > self._n)]
        )
        a = np.stack([a[..., i] for i in window_idxs])
        # shape should now be (n_win, ..., n_time)
        # where n_time = samples_per_window
        assert (
            a.shape[0] == n_win - 1
        ), f"Failed to window the data. Expected {n_win-1}, but got {a.shape[0]}"
        assert a.shape[-1] == self.samples_per_window

        # replace time axis. Shape should now be (... n_windows, ..., n_time)
        a = np.moveaxis(a, 0, axis)
        return a if not return_indices else a, window_idxs

    def fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> "Windower":
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
        self._t = np.prod(np.array(x.shape)[self.axis])
        self._y = y
        self._y_hat = self._make_labels(y)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the dataframe of packets into a dataframe of windows

        Parameters
        ----------
        x : np.ndarray
            see `fit()`

        Returns
        -------
        np.ndarray
            WxD matrix where W is the number of windows and `D = packet_size *
            window_size`
        """
        # place all time axes in the back and all other axes up front
        # Flatten the back so that time is linear across all other axes
        axis_restore: np.ndarray = np.arange(x.ndim)
        if isinstance(self.axis, int):
            x = np.moveaxis(x, self.axis, -1)
            x = x.reshape(*x.shape[:-1], -1)
            axis_restore[[-1, self.axis]] = axis_restore[[self.axis, -1]]
        elif isinstance(self.axis, list):
            x = np.moveaxis(x, self.axis, range(-len(self.axis), 0))
            x = x.reshape(*x.shape[: -len(self.axis)], -1)
            axis_restore[[np.arange(-len(self.axis), 0), self.axis]] = axis_restore[
                [self.axis, np.arange(-len(self.axis), 0)]
            ]

        full_t: int = x.shape[-1]
        n_windows: int = int(full_t / self.window_step)
        window_start_idxs: np.ndarray = np.linspace(0, full_t, n_windows + 1).astype(
            np.int32
        )
        window_idxs: np.ndarray = np.array(
            [np.arange(s, s + self.samples_per_window) for s in window_start_idxs]
        )
        window_idxs = np.array([idxs for idxs in window_idxs if ~np.any(idxs > full_t)])

        if self.label_scheme != 4:
            x = x[..., window_idxs]
            x = np.moveaxis(x, axis_restore, np.arange(axis_restore.size))
        else:
            y: np.array = self._y
            y = np.array(y.tolist() * self._t).reshape(self._t, -1).T
            y = y.flatten()
            yd = np.array([b - a for a, b in zip(y[1:], y[:-1])])
            yd = np.array([0] + (np.where(yd != 0)[0] + 1).tolist() + [y.size])
            window_idxs_list: List = [
                list(range(a, b)) for a, b in zip(yd[:-1], yd[1:])
            ]
            window_lengths = [len(i) for i in window_idxs_list]
            max_win_len = max(window_lengths)
            list_x = [x[i] for i in window_idxs_list]
            x = np.stack(
                [
                    np.r_[i, np.ones((max_win_len - i.shape[0], n_channels)) * np.nan]
                    for i in list_x
                ]
            )
            x = np.swapaxes(x, 1, 2)

        self._X = x

        return self._X

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
