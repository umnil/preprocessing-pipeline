import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import Optional


class ArtifactRemover(TransformerMixin, BaseEstimator):
    def __init__(self, std: int = 10):
        """Blunt artifact remover. This simply discards any incidents that have
        detectable artifacts which are defined as points above or below std
        standard deviations from the mean

        Parameters
        ----------
        std : int
            The number of standard of deviations to place the threshold detection
        """
        self.std: int = 10

    def fit(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> "ArtifactRemover":
        self.y: np.ndarray = y
        return self

    def transform(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, *args, **kwargs
    ) -> np.ndarray:
        mean: np.ndarray = x.mean(axis=-1)
        std: np.ndarray = x.std(axis=-1)
        lo_thresh: np.ndarray = mean - std * self.std
        hi_thresh: np.ndarray = mean + std * self.std

        lo_thresh = np.repeat(lo_thresh, x.shape[-1]).reshape(*x.shape[:-1], -1)
        hi_thresh = np.repeat(hi_thresh, x.shape[-1]).reshape(*x.shape[:-1], -1)

        lo_mask: np.ndarray = (x < lo_thresh).any(axis=(1, 2))
        hi_mask: np.ndarray = (x > hi_thresh).any(axis=(1, 2))
        mask = lo_mask | hi_mask

        x = x[~mask]
        self._y_hat = self.y[~mask]

        return x
