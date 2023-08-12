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
            flattened_y = y.reshape(-1, self.samples_per_window)
            flattened_y = np.array([i[0] for i in flattened_y if len(set(i)) < 2])
            y = flattened_y if y.ndim < 3 else flattened_y.reshape(y.shape[0], -1)
            y_transformed = y.squeeze()
        elif self.label_scheme == 4:
            # Windows by labels
            y = self._y
            if y.ndim > 1:
                y = utils.equalize_list_to_array(
                    [cast(np.ndarray, self._window_by_label(i)) for i in y]
                )
            else:
                y = cast(np.ndarray, self._window_by_label(y))
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
        labels = utils.equalize_list_to_array(labels_list)
        return labels[:, 0] if not return_indices else (labels[:, 0], window_list_idxs)

    def _window_transform(
        self, a: np.ndarray, axis: int = -1, return_indices: bool = False
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
        assert (
            a.shape[0] == expected_n_windows
        ), f"Failed to window the data. Expected {expected_n_windows}, but got {a.shape[0]}"
        assert a.shape[-1] == self.samples_per_window

        # replace time axis. Shape should now be (... n_windows, ..., n_time)
        a = np.moveaxis(a, 0, axis)
        return a if not return_indices else (a, window_idxs)

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
        dim_list: np.ndarray = np.arange(x.ndim)
        if isinstance(self.axis, int):
            x = np.moveaxis(x, self.axis, -1)
            x = x.reshape(*x.shape[:-1], -1)
            dim_list[[-1, self.axis]] = dim_list[[self.axis, -1]]
        elif isinstance(self.axis, list):
            x = np.moveaxis(x, self.axis, range(-len(self.axis), 0))
            x = x.reshape(*x.shape[: -len(self.axis)], -1)
            dim_list[[np.arange(-len(self.axis), 0), self.axis]] = dim_list[
                [self.axis, np.arange(-len(self.axis), 0)]  # type: ignore
            ]

        if self.label_scheme < 4:
            x = cast(np.ndarray, self._window_transform(x))
            x = np.moveaxis(x, dim_list, np.arange(dim_list.size)).squeeze()
            if self.label_scheme == 3:
                # Remove windows with mixed labelling
                t_axis: int = self.axis if isinstance(self.axis, int) else self.axis[0]
                y, win_idxs = self._window_transform(self._y, t_axis, True)
                n_axis: int = 1 if isinstance(self.axis, int) else len(self.axis)
                flatten_y: np.ndarray = y.reshape(-1, *y.shape[n_axis:])
                flatten_x: np.ndarray = x.reshape(-1, *x.shape[n_axis:])
                flatten_x = np.array(
                    [i for i, j in zip(flatten_x, flatten_y) if len(set(j)) == 1]
                )
                if n_axis > 1:
                    x = flatten_x.reshape(x.shape[0], -1, *x.shape[2:])
                else:
                    x = flatten_x
        else:
            y = self._y
            if y.ndim > 1:
                idxs = [self._window_by_label(i, True)[1] for i in y]
                print(type(idxs[0]))
                x = np.array(
                    [
                        utils.equalize_list_to_array([a[..., i] for i in idx])
                        for a, idx in zip(x, idxs)
                    ]
                )
            else:
                _, idxs = self._window_by_label(self._y, True)
                x = utils.equalize_list_to_array([x[..., i] for i in idxs]).squeeze()

        self._x = x

        return self._x
