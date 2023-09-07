import logging
import numpy as np

from typing import List, Tuple, Union, cast
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

        # Uninitialized variables to be defined later
        self._n: int
        self._n_windows: int
        self._t: int
        self._y_hat: np.ndarray
        self._y: np.ndarray
        self._x: np.ndarray

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
        self._n = y.shape[self.axis]
        y, win_idxs = self._window_transform(y, self.axis, True)
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
            if self.return_masked:
                mask: np.ndarray = np.zeros(y.shape)
                flattened_y = y.reshape(-1, self.samples_per_window)
                mask = mask.reshape(flattened_y.shape)
                i: np.ndarray = np.array(
                    [i for i, a, in enumerate(flattened_y) if len(set(a)) < 2]
                )
                j: np.ndarray = np.zeros(i.size).astype(np.int32)
                mask[i, j] = 1
                mask = mask.reshape(y.shape).astype(bool)
                y = np.ma.MaskedArray(data=y, mask=~mask)[..., 0]
            else:
                y = utils.mask_list(
                    [[win[0] for win in inst if len(set(win)) < 2] for inst in y]
                )

            y_transformed = y.squeeze()
        elif self.label_scheme == 4:
            # Windows by labels
            y = self._y
            y = utils.mask_list([cast(np.ndarray, self._window_by_label(i)) for i in y])
            y_transformed = y

        self._n_windows = y_transformed.shape[0]
        return y_transformed

    @staticmethod
    def _window_by_label(
        labels: np.ndarray, return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
        """Create windows over labels such that each window is a unique set of
        values within labels. The order of the windows are as the values appear
        in labels. Labels is expected to be one dimensional

        Parameters
        ----------
        labels : np.ndarray
            The labels to window
        return_indices : bool
            If True, return the windowing indices that can be used to rewindow
            the original data

        Return
        ------
        np.ndarray
            The 2d windowed array
        List
            A list of windowing indices. Only returned if `return_indices` is true
        """
        assert (
            labels.ndim == 1
        ), "Windowing by label requiers that the labels are 1 dimensional"
        delta: np.ndarray = np.diff(labels)
        transitions: np.ndarray = np.where((delta != 0) * (~np.isnan(delta)))[-1] + 1
        delta_idxs: np.ndarray = np.array([0] + transitions.tolist() + [labels.size])
        window_list_idxs: List = [
            np.arange(a, b) for a, b in zip(delta_idxs[:-1], delta_idxs[1:])
        ]
        labels_list: List = [labels[i] for i in window_list_idxs]
        labels = utils.mask_list(labels_list)
        return labels[:, 0] if not return_indices else (labels[:, 0], window_list_idxs)

    def _window_transform(
        self, a: np.ndarray, axis: int = -1, return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Performs a windowing transformation on an array across the given
        axis. Note that the resulting array will have `a.ndim + 1` number of
        dimensions to account for the windows. So if `a` has a shape of (5, 10,
        2000, 8), and the time axis is `axis=2`, the resulting shape will be
        (5, 10, n_win, n_time, 8).

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
        # normalize axis
        axis = axis if axis >= 0 else a.ndim + axis

        # Move working axis to the back
        a = np.moveaxis(a, axis, -1)

        # calculate the number of windows
        t: int = a.shape[-1]
        n_win: int = int(t / self.window_step)

        # n_win +1 to ensure we go *over* the time. We'll cut off later
        window_start_idxs = np.linspace(0, t, n_win + 1).astype(np.int32)
        window_idxs = np.array(
            [np.arange(s, s + self.samples_per_window) for s in window_start_idxs]
        )

        # Here we cut back any access beyond the time
        window_idxs = np.array([idxs for idxs in window_idxs if ~np.any(idxs > t)])
        a = np.stack([a[..., i] for i in window_idxs])
        # shape should now be (n_win, ..., n_time)
        # where n_time = samples_per_window
        expected_n_windows: int = int(
            (t - self.samples_per_window) / self.window_step + 1
        )
        assert a.shape[0] == expected_n_windows, (
            "Failed to window the data."
            f"Expected {expected_n_windows}, but got {a.shape[0]}"
        )
        assert a.shape[-1] == self.samples_per_window

        # replace time axis. Shape should now be (... n_windows, n_time)
        a = np.moveaxis(a, 0, axis)
        return a if not return_indices else (a, window_idxs)

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
        self._y_hat = self._make_labels(y)
        self._y_lengths = [
            i.data[~i.mask].shape[0]
            for i in np.ma.masked_invalid(
                self._y_hat if self._y_hat.ndim > 1 else self._y_hat[None, ...]
            )
        ]
        # place all time axes in the back and all other axes up front
        # Flatten the back so that time is linear across all other axes
        dim_list: np.ndarray = np.arange(x.ndim)
        x = np.moveaxis(x, self.axis, -1)
        x = x.reshape(*x.shape[:-1], -1)
        dim_list[[-1, self.axis]] = dim_list[[self.axis, -1]]

        if self.label_scheme < 4:
            x = cast(np.ndarray, self._window_transform(x))
            x = np.moveaxis(x, dim_list, np.arange(dim_list.size))
            if self.label_scheme == 3:
                # Calculate label transformation
                y = cast(np.ndarray, self._window_transform(self._y, self.axis))

                # Reshape to known axes (n, n_win, ...)
                x = np.moveaxis(x, self.axis - 1, 1)
                if self.return_masked:
                    flattened_x: np.ndarray = x.reshape(-1, *x.shape[2:])
                    flattened_y: np.ndarray = y.reshape(-1, self.samples_per_window)
                    mask: np.ndarray = np.zeros(flattened_x.shape)
                    i: np.ndarray = np.array(
                        [i for i, a in enumerate(flattened_y) if len(set(a)) < 2]
                    )
                    mask[i, ...] = 1
                    mask = mask.reshape(x.shape).astype(bool)
                    x = np.ma.MaskedArray(data=x, mask=~mask)
                else:
                    x = utils.mask_list(
                        [
                            [xwin for xwin, ywin in zip(xi, yi) if len(set(ywin)) < 2]
                            for xi, yi in zip(x, y)
                        ]
                    )

                # Reseparate the n and windowing axes
                x = np.moveaxis(x, 1, self.axis - 1)
        else:
            y = self._y
            idxs = [self._window_by_label(i, True)[1] for i in y]
            # Apply windows
            x_win = [[a[..., i] for i in idx] for a, idx in zip(x, idxs)]
            x_time = [utils.equalize_list_to_array(a) for a in x_win]
            x = utils.equalize_list_to_array(x_time, axes=[0, -1])
            x = np.moveaxis(x, 1, self.axis - 1)

        self._x = x

        return self._x
