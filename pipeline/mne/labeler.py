import mne  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import List, Optional, Union


class Labeler(TransformerMixin, BaseEstimator):
    """Input is an mne.Raw object and output is a numpy array in the shape of
    (sample, channels, time), time will have a length of one so that the
    Windowing object can perform the appropraite windowing
    """

    def __init__(self, labels: Optional[List] = None, channels: Optional[List] = None):
        """Initialize a labeler that will search for the labels specified in
        `labels`

        Parameters
        ----------
        labels : Optional[List]
            Labels
        channels : Optional[List]
            Channels to pick
        """
        self.labels: Optional[List[str]] = labels
        self.channels: Optional[List[int]] = channels

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
            self._y_hat = np.concatenate([self.load_labels(i) for i in x])
        else:
            self._y_hat = self.load_labels(x)
        return self

    def load_labels(self, raw: mne.io.Raw) -> np.ndarray:
        """Load the labels from an individual raw object

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
            retval = pd.Categorical(y_labels, self.labels).codes
        else:
            retval = np.unique(y_labels, return_inverse=True)[1]

        retval = retval.astype(np.float64)
        retval[retval == -1] = np.nan
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
                data = (
                    i.copy() if self.channels is None else i.copy().pick(*self.channels)
                )
                data = i.get_data()
                data = np.moveaxis(np.expand_dims(data, -1), 0, 1)
                data_list.append(data)
            self._x_hat = np.concatenate(data_list)
        else:
            data = x.copy() if self.channels is None else x.copy().pick(*self.channels)
            data = data.get_data()
            self._x_hat = np.moveaxis(np.expand_dims(data, -1), 0, 1)
        return self._x_hat
