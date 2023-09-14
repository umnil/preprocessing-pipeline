import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore


class Vectorizer(BaseEstimator, TransformerMixin):
    def fit(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> "Vectorizer":
        self.preserved_dims: int = y.ndim
        return self

    def transform(self, x: np.ndarray):
        return x.reshape(*x.shape[: self.preserved_dims], -1)
