import logging
import numpy as np

from typing import Callable, List, Tuple, Union, cast
from scipy import stats
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
        return_masked: bool = False,
        axis: int = -1,
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
        return_masked : bool
            When True the labels and data are returned as a masked array. This
            logic only applies to labeling shemes that either remove (e.g.,
            labal_scheme == 3) or add to (e.g., label_scheme == 4)
        axis : int
            The dimension across which to perform windowing.
        """
        self.samples_per_window: int = samples_per_window
        self.window_step: int = window_step
        self.trial_size: int = trial_size
        self.label_scheme: int = label_scheme
        self._window_packets: np.ndarray = np.array([])
        self.return_masked: bool = return_masked
        self.axis: int = axis
        self.window_fn: Callable = (
            self._window_transform if self.label_scheme < 4 else self._split_transform
        )

        # Uninitialized variables to be defined later
        self._n: int
        self._n_windows: int
        self._t: int
        self._y_hat: np.ndarray
        self._y: np.ndarray
        self._x: np.ndarray

    @staticmethod
    def _compute_split_indices(a: np.ndarray, axis: int = -1) -> List:
        """Return a list of window indices to split the data against. The
        return type is a list, instead of a numpy array, because splitting does
        not constrain the windowing to a fixed size, so the numpy matrix would
        not be square

        Parameters
        ----------
        a : np.ndarray
            The input array with data to split by. Data will be split such that
            each data split will have samples that are equivlant. In other
            words, data are only split across time a points when sample values
            change
        axis : int
            The axis to consider as the time axis
        """
        assert (
            a.ndim == 1
        ), "Windowing by label requiers that the labels are 1 dimensional"
        delta: np.ndarray = np.diff(a)
        transitions: np.ndarray = np.where((delta != 0) * (~np.isnan(delta)))[-1] + 1
        delta_idxs: np.ndarray = np.array([0] + transitions.tolist() + [a.size])
        window_list_idxs: List = [
            np.arange(a, b) for a, b in zip(delta_idxs[:-1], delta_idxs[1:])
        ]
        return window_list_idxs

    @staticmethod
    def _compute_window_indices(
        a: np.ndarray, step_size: int, window_size: int, axis: int = -1
    ) -> np.ndarray:
        """Return a matrix of indices that can be applyed to a to result in a windowed
        version of a

        Parameters
        ----------
        a : np.ndarray
            The input matrix
        step_size : int
            The number of samples to move the window forward for each window
        window_size : int
            The number of samples per window
        axis : int
            The axis along which to window the data
        """
        # normalize axis
        axis = axis if axis >= 0 else a.ndim + axis

        # calculate the number of windows
        t: int = a.shape[axis]
        n_win: int = int(t / step_size)

        # n_win +1 to ensure we go *over* the time. We'll cut off later
        window_start_idxs = np.linspace(0, t, n_win + 1).astype(np.int32)
        window_idxs = np.array(
            [np.arange(s, s + window_size) for s in window_start_idxs]
        )

        # Here we cut back any access beyond the time
        window_idxs = np.array([idxs for idxs in window_idxs if ~np.any(idxs > t)])
        return window_idxs

    @staticmethod
    def _compute_lengths(a: np.ndarray) -> np.ndarray:
        """Compute the lengths. The input should be a 3D array of shape (runs,
        windows, time). The resulting output is an array of n_runs, where each
        element lists the number of windows. Used for time-series models such
        as hidden markov models"""
        if not isinstance(a, np.ma.core.MaskedArray):
            length_list: List = [len(i) for i in a]
            return np.array(length_list)
        else:
            length_list = [np.sum(~run.mask) for run in a]
            return np.array(length_list)

    @staticmethod
    def _mask_heterogenous_axis(
        a: np.ndarray, axis: int = -1
    ) -> np.ma.core.MaskedArray:
        """Given a multi-dimension array, mask the array along the axis if the
        values along that axis for each instance are not homogenous

        Parameters
        ----------
        a : np.ndarray
            The input array

        Returns
        -------
        np.ndarray
            The masked array
        """
        # move axis to the back
        a_ax: np.ndarray = np.moveaxis(a, axis, -1)

        # flatten
        n: int = a_ax.shape[-1]
        a_fl: np.ndarray = a_ax.reshape(-1, n)

        # generate mask
        mask_fl: np.ndarray = np.array([[~np.all(i[0] == i)] * i.size for i in a_fl])

        # mask
        m_fl: np.ma.core.MaskedArray = np.ma.array(a_fl, mask=mask_fl)

        # reshape
        m_ax: np.ndarray = m_fl.reshape(*a_ax.shape)

        # reset
        m: np.ndarray = np.moveaxis(m_ax, -1, axis)
        return m

    def _split_transform(self, *a) -> List:
        """Create windows over labels such that each window is a unique set of
        values within labels. The order of the windows are as the values appear
        in labels. Labels is expected to be one dimensional

        Parameters
        ----------
        x : np.ndarray
            The data to window
        ... : np.ndarray
            Other arrays to apply the windowing to
        y : np.ndarray
            The array which will determine the indexing

        Return
        ------
        xt, yt, ...
            A list of transformed datasets
        """
        labels: np.ndarray = a[-1]
        window_list_idxs: List = [self._compute_split_indices(run) for run in labels]
        results = []
        for ix, x in enumerate(a):
            assert x.ndim <= 3, "Only 3 dimensional (run, channel, time), is supported"
            assert x.shape[0] == len(
                window_list_idxs
            ), "mismatch in number of runs in datasets"
            x_list: List = [
                utils.equalize_list_to_array([runx[..., i] for i in runi])
                for runx, runi in zip(x, window_list_idxs)
            ]
            # Shape is now (run, window, ..., time)
            xt: np.ma.core.MaskedArray = np.ma.masked_invalid(
                utils.equalize_list_to_array(x_list, axes=[0, -1])
            )
            xt = np.moveaxis(xt, 1, -2)
            results.append(xt)
        return results

    def _window_transform(self, *a, axis: int = -1) -> np.ndarray:
        """Performs a windowing transformation on an array across the given
        axis. Note that the resulting array will have `a.ndim + 1` number of
        dimensions to account for the windows. So if `a` has a shape of (5, 10,
        2000, 8), and the time axis is `axis=2`, the resulting shape will be
        (5, 10, n_win, n_time, 8).

        Parameters
        ----------
        x : np.ndarray
            The input array to perform the windowing on
        ...
        y : np.ndarray
            Final input array to perform windowing on
        axis : int
            The axis within `a` across which to perform windowing

        Returns
        -------
        List
            The list of windowed arrays, plust the lengths of runs
        """
        results: List = []
        for x in a:
            # normalize axis
            target_axis = axis if axis >= 0 else x.ndim + axis
            # Move working axis to the back
            x = np.moveaxis(x, target_axis, -1)
            t: int = x.shape[-1]

            # compute the window indices to window the data
            window_idxs: np.ndarray = self._compute_window_indices(
                x, self.window_step, self.samples_per_window, target_axis
            )

            # Apply the indices and check for masking
            data: np.ndarray
            mask: np.ndarray
            if isinstance(a, np.ma.core.MaskedArray):
                data = x.data
                mask = x.mask
                data = np.stack([data[..., i] for i in window_idxs])
                mask = np.stack([mask[..., i] for i in window_idxs])
                x = np.masked_array(data, mask)
            else:
                data = x
                data = np.stack([data[..., i] for i in window_idxs])
                x = data

            # shape should now be (n_win, ..., n_time)
            expected_n_windows: int = int(
                (t - self.samples_per_window) / self.window_step + 1
            )
            # where n_time = samples_per_window
            assert x.shape[0] == expected_n_windows, (
                "Failed to window the data."
                f"Expected {expected_n_windows}, but got {x.shape[0]}"
            )
            assert x.shape[-1] == self.samples_per_window

            # replace time axis. Shape should now be (... n_windows, n_time)
            x = np.moveaxis(x, 0, target_axis)
            results.append(x)

        return results

    def fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> "Windower":
        """Given an n-dimensional matrix `x`, the time-dimension, indicated by
        `axis`, is windowed over creating a new dimension (i.e., from time ->
        windows, time). The way the windows are labelled is determined by the
        selected `label_scheme`

        Parameters
        ----------
        X : np.ndarray
            An matrix array of data with a time dimension
        y : np.ndarray
            An NxM matrix array that contains the corresponding labels for X.
            The size of the first dimension must match X.

        Retunrs
        -------
        Windower
            The current instance of the windowing object
        """
        return self

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.transform(x, y)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Transforms the dataframe of packets into a dataframe of windows

        Parameters
        ----------
        x : np.ndarray
            see `fit()`

        y : np.ndarray

        Returns
        -------
        np.ndarray
            WxD matrix where W is the number of windows and `D = packet_size *
            window_size`
        """
        # Process y data
        assert y.ndim > 1, "y labels must be at least two dimensions"
        assert x.ndim > 1, "x labels must be at least two dimensions"
        self._t = np.prod(np.array(x.shape)[self.axis])
        self._y = y

        # Apply windowing
        xt, yt_prelabel = self.window_fn(x, y)
        assert yt_prelabel.ndim == 3

        # Apply Labeling Scheme
        if self.label_scheme == 0:
            yt: np.ndarray = yt_prelabel[..., 0]
        elif self.label_scheme == 1:
            yt = yt_prelabel[..., -1]
        elif self.label_scheme == 2:
            yt = stats.mode(yt_prelabel, axis=-1)
        elif self.label_scheme == 3:
            yt_masked: np.ma.core.MaskedArray = self._mask_heterogenous_axis(
                yt_prelabel, self.axis
            )
            n_ch: int = xt.shape[1]
            xt_mask_shift: np.array = np.array([yt_masked.mask] * n_ch)
            xt_mask: np.array = np.moveaxis(xt_mask_shift, 0, 1)
            xt_masked: np.ma.core.MaskedArray = np.ma.array(xt, mask=xt_mask)

            yt = yt_masked[..., 0]
            xt = xt_masked
        elif self.label_scheme == 4:
            yt = yt_prelabel[..., 0]

        # Compute lengths
        self._y_lengths = self._compute_lengths(yt)

        self._x_hat = xt
        self._y_hat = yt
        return self._x_hat
