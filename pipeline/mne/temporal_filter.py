import mne
import numpy as np
from typing import Tuple


class NDTemporalFilter(mne.decoding.TemporalFilter):
    """A mne TemporalFilter that is capable of filtering multidimension arrays
    beyond the restricted 3 dimensions
    """

    def transform(self, x: np.ndarray) -> np.ndarray:
        shape: Tuple = x.shape
        x = x.reshape(-1, shape[-1])
        x = super().transform(x)
        x = x.reshape(shape)
        return x
