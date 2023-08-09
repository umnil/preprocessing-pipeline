import mne  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import List, Optional, Union

from .. import utils


class Labeler(TransformerMixin, BaseEstimator):
    """Input is an mne.Raw object and output is a numpy array in the shape of
    (sample, channels, time), time will have a length of one so that the
    Windowing object can perform the appropraite windowing
    """

    def __init__(
        self,
        labels: Optional[List] = None,
        channels: Optional[List] = None,
        concatenate: bool = True,
    ):
        """Initialize a labeler that will search for the labels specified in
        `labels`

        Parameters
        ----------
        labels : Optional[List]
            Labels. If unlabeled sections should be included, be sure to add `None` to the list
        channels : Optional[List]
            Channels to pick
        concatenate : bool
            When transform is passed a list of Raws, if `concatenate` is set to
            `False`, the transform will return a 3 dimensional matrix
        """
        self.labels: Optional[List[str]] = labels
        self.channels: Optional[List[int]] = channels
        self.concatenate: bool = concatenate
        self._x_hat: np.ndarray = np.empty([])
        self._y_hat: np.ndarray = np.empty([])
        self._x_lengths: List[int] = []
        self._y_lengths: List[int] = []

    def fit(self, x: Union[mne.io.Raw, List[mne.io.Raw]], *args, **kwargs) -> "Labeler":
        """Fit the labeler to the raw data

        Parameters
        ----------
        x : Union[mne.io.Raw, List[mne.io.Raw]]
            The mne raw object with annotations or a list of them

        Returns
        -------
        Labeler
            self
        """
        if isinstance(x, List):
            y_hat_list: List = [self.load_labels(i) for i in x]
            self._y_lengths = [i.shape[-1] for i in y_hat_list]
            if not self.concatenate:
                self._y_hat = utils.equalize_list_to_array(y_hat_list)
            else:
                self._y_hat = np.concatenate(y_hat_list)
        else:
            self._y_hat = self.load_labels(x)
            self._y_lengths = [self._y_hat.shape[-1]]
        return self

    def load_labels(self, raw: mne.io.Raw) -> np.ndarray:
        """Load the labels from an individual raw object. If the `Labeler` was
        initialized with a set of labels, the labels are replaced with the
        index value of the label as they are provided. The result is always a
        list of integers. Values not found in the labels are marked as NaN.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw mne object containing the data

        Returns
        -------
        np.ndarray
            A list of labels for each data point
        """
        sfreq: int = raw.info["sfreq"]
        data: np.ndarray = raw.get_data()
        n_samples: int = data.shape[-1]
        y_labels: List = ["None"] * n_samples
        n_annotations: int = len(raw.annotations.description)
        for i in range(n_annotations):
            onset: float = raw.annotations.onset[i]
            duration: float = raw.annotations.duration[i]
            description: str = raw.annotations.description[i]

            onset_samples: int = int(np.round(onset * sfreq))
            duration_samples: int = int(np.round(duration * sfreq))
            begin: int = onset_samples
            end: int = onset_samples + duration_samples
            n_label_samples: int = end - begin
            label_seg: List[str] = [description] * n_label_samples
            y_labels[begin:end] = label_seg

        retval: np.ndarray
        if self.labels is not None:
            labels = self.labels
            if None in self.labels:
                labels = [label for label in self.labels if label is not None]
            retval = pd.Categorical(y_labels, labels).codes
            retval = retval.astype(np.float64)
            if None not in self.labels:
                # We make time points that will be removed as NaN
                retval[retval == -1.0] = retval[retval == -1.0] * np.nan
        else:
            retval = np.unique(y_labels, return_inverse=True)[1]
            retval = retval.astype(np.float64)

        retval = retval.astype(np.float64)
        return retval

    def transform(self, x: Union[mne.io.Raw, List[mne.io.Raw]]) -> np.ndarray:
        """Transform an mne.Raw object into individual data and labels

        Parameters
        ----------
        x : Union[mne.io.Raw, List[mne.io.Raw]]
            The mne Rae object that contains data and annotations

        Returns
        -------
        np.ndarry
            Shape of (time, channels, 1)
        """
        data: np.ndarray
        if isinstance(x, List):
            data_list: List[np.ndarray] = []
            for i in x:
                i = i.copy() if self.channels is None else i.copy().pick(self.channels)
                data = i.get_data()
                data = np.moveaxis(np.expand_dims(data, -1), 0, 1)
                data_list.append(data)
            if not self.concatenate:
                self._x_hat = utils.equalize_list_to_array(data_list)
                self._x_hat = utils.equalize_list_to_array(
                    [i[~np.isnan(j)] for i, j in zip(self._x_hat, self._y_hat)]
                )
                self._y_hat = utils.equalize_list_to_array(
                    [i[~np.isnan(i)] for i in self._y_hat]
                )
            else:
                self._x_hat = np.concatenate(data_list)
            self._x_lengths = [i.shape[0] for i in data_list]
        else:
            x = x.copy() if self.channels is None else x.copy().pick(self.channels)
            data = x.get_data()
            self._x_hat = np.moveaxis(np.expand_dims(data, -1), 0, 1)
            self._x_hat = self._x_hat[~np.isnan(self._y_hat)]
            self._y_hat = self._y_hat[~np.isnan(self._y_hat)]

        # Remove NaN marked data
        return self._x_hat
