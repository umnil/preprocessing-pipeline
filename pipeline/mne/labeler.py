import mne
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class Labeler(TransformerMixin, BaseEstimator):
    """Input is an mne.Raw object and output is a numpy array in the shape of
    (sample, channels, time), time will have a length of one so that the
    Windowing object can perform the appropraite windowing
    """

    def fit(self, x: mne.io.Raw, *args, **kwargs) -> "Labeler":
        """Fit the labeler to the raw data

        Parameters
        ----------
        x : mne.io.Raw
            The mne raw object with annotations

        Returns
        -------
        Labeler
            self
        """
        self.sfreq: int = x.info["sfreq"]
        data: np.ndarray = x.get_data()
        n_samples: int = data.shape[-1]
        labels: np.ndarray = np.array(["None"] * n_samples)
        n_annotations: int = len(x.annotations.description)
        for i in range(n_annotations):
            onset: float = x.annotations.onset[i]
            duration: float = x.annotations.duration[i]
            description: str = x.annotations.description[i]

            onset_samples: int = int(np.round(onset * self.sfreq))
            duration_samples: int = int(np.round(duration * self.sfreq))
            begin: int = onset_samples
            end: int = onset_samples + duration_samples
            n_label_samples: int = end - begin
            label_seg: List[str] = [description] * n_label_samples
            labels[begin:end] = label_seg

        self._y_hat: np.ndarray = np.unique(labels, return_inverse=True)[1]
        return self

    def transform(self, x: mne.io.Raw) -> np.ndarray:
        """Transform an mne.Raw object into individual data and labels

        Parameters
        ----------
        x : mne.io.Raw
            The mne Rae object that contains data and annotations

        Returns
        -------
        np.ndarry
            Shape of (time, channels, 1)
        """
        data: np.ndarray = x.get_data()
        self._x_hat = np.moveaxis(np.expand_dims(data, -1), 0, 1)
        return self._x_hat
