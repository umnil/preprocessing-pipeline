import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple


class NDPCA(PCA):
    def fit(x: np.ndarray, y: Optional[np.ndarray] = None) -> "NDPCA":
        x = self.prepare_inputs(x, x.shape)
        return super(NDPCA, self).fit(x, y)

    def prepare_inputs(x: np.ndarray, original_shape: Tuple) -> np.ndarray:
        n_features: int = original_shape[-1]
        x = x.reshape(-1, n_features)
        return x

    def prepare_outputs(x: np.ndarray, original_shape: Tuple) -> np.ndarray:
        n_components: int = x.shape[-1]
        x = x.reshape(*original_shape[:-1], n_components)
        return x

    def transform(x: np.ndarray, y: Optional[np.array] = None) -> np.ndarray:
        original_shape: Tuple = x.shape
        x = self.prepare_inputs(x, original_shape)
        x = super(NDPCA, self).transform(x)
        x = self.prepare_outputs(x, original_shape)
        return x
