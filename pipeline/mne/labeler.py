import mne  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import List, Optional, Tuple, Union, cast

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
            Labels. If unlabeled sections should be included, be sure to add
            `None` to the list.
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
        self._mask: List = []

    @staticmethod
    def apply_mask(a: np.ndarray, mask: List) -> np.ndarray:
        """Applies each mask in list to each row element in a

        Parameters
        ----------
        a : np.ndarray
            The input array
        mask : List
            A list of masks for each row component in a

        Returns
        -------
        np.ndarray
            `a` with data masked
        """
        assert a.shape[0] == len(
            mask
        ), f"a has shape {a.shape} and there are {len(mask)} masks"
        a_list: List = utils.array_to_clean_list(a)
        a_list = [i[m] for i, m in zip(a_list, mask)]
        return utils.equalize_list_to_array(a_list, axis=0)

    def filter_labels(self, y_labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Given a list of string labels obtained from mne annotations convert
        these to numerical labels depending on whether they're provided to the
        labeler class. The result is always a list of integers. Data where
        labels are not requested will be removed. A mask is initialized on the
        class for use when transforming the x data to remove the corresponding
        data that occurs with the removed labels

        Parameters
        ----------
        y_labels : List[str]
            A list of labels. Each label corresponds to exactly one sampled
            data point

        Returns
        -------
        np.ndarray
            The filtered labels
        np.ndarray
            A mask for how the labels were filtered
        """
        cur_mask: np.ndarray = np.array([True] * len(y_labels))
        retval: np.ndarray
        if self.labels is not None:
            labels = self.labels
            if None in self.labels:
                labels = [
                    label if label is not None else "None" for label in self.labels
                ]
            # Data points with no label are marked as -1
            retval = pd.Categorical(y_labels, labels).codes
            # We make time points that will be removed as NaN
            cur_mask = retval != -1.0
            retval = retval[cur_mask]
        else:
            retval = np.unique(y_labels, return_inverse=True)[1]

        retval = retval.astype(np.float64)
        return retval, cur_mask

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
            filtered_labels: List = [self.filter_labels(self.load_labels(i)) for i in x]
            y_hat_list: List = [i[0] for i in filtered_labels]
            mask_list: List = [i[1] for i in filtered_labels]
            self._y_lengths = [i.shape[-1] for i in y_hat_list]
            self._y_hat = utils.equalize_list_to_array(y_hat_list)
            self._mask = mask_list
            if self.concatenate:
                self._y_hat = np.concatenate(y_hat_list)
        else:
            self._y_hat, mask = self.filter_labels(self.load_labels(x))
            self._mask = cast(List, mask)
            self._y_lengths = [self._y_hat.shape[-1]]
        return self

    @staticmethod
    def load_labels(raw: mne.io.Raw) -> List[str]:
        """Load the labels from an individual raw object. If the `Labeler` was
        initialized with a set of labels, the labels are replaced with the
        index value of the label as they are provided.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw mne object containing the data

        Returns
        -------
        List[str]
            A list of labels
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

        return y_labels

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

            self._x_hat = utils.equalize_list_to_array(data_list, axis=0)
            self._x_hat = self.apply_mask(self._x_hat, self._mask)

            if self.concatenate:
                self._x_hat = np.concatenate(self._x_hat)
                non_nan_idxs: np.bool_ = ~(np.isnan(self._x_hat)[:, 0].squeeze())
                self._x_hat = self._x_hat[non_nan_idxs]

            self._x_lengths = [i.shape[0] for i in data_list]
        else:
            x = x.copy() if self.channels is None else x.copy().pick(self.channels)
            data = x.get_data()
            self._x_hat = np.moveaxis(  # size = (time, channels, 1)
                np.expand_dims(data, -1), 0, 1
            )
            self._x_hat = self._x_hat[self._mask]

        return self._x_hat

    @property
    def is_masked(self) -> bool:
        """This property returns true if the `_x_hat` value contains nan values"""
        return np.isnan(self._x_hat).any()
