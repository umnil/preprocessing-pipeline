import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple


class Pairer(TransformerMixin, BaseEstimator):
    def __init__(self, pair_values: List[int] = [0, 1], force_even: bool = True):
        assert len(pair_values) == 2, "Only two pair values can be listed for pairing"
        self.pair_values: List[int] = pair_values
        self.force_even: bool = force_even

    def fit(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        assert (
            x.shape[0] == y.shape[0]
        ), "x and y must have the same number of instances"

        if self.force_even:
            if y.shape[0] % 2 > 0:
                y = y[:-1]
        first_label, second_label = self.pair_values
        labels, counts = np.unique(y, return_counts=True)
        assert np.all(
            counts[0] == counts[:2]
        ), f"Motor imagery labels are unbalanced: {counts}"

        idxs1: np.ndarray = np.where(y == first_label)[0]
        assert np.all(y[idxs1 + 1] == second_label), (
            f"Data is not sequential. Expected the first label ({first_label}) to "
            f"precede the second ({second_label})"
        )
        idxs2: np.ndarray = idxs1 + 1
        self.pairs: np.ndarray = np.c_[idxs1, idxs2]
        self._y_hat = np.array([[y[p1], y[p2]] for p1, p2 in self.pairs])
        return self

    def transform(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.force_even:
            if x.shape[0] % 2 > 0:
                x = x[:-1]

        input_shape: Tuple = x.shape
        n_inst: int = input_shape[0]
        x = x.reshape(n_inst, -1)
        x = np.stack([[x[p1], x[p2]] for p1, p2 in self.pairs])
        output_shape: List = list(x.shape[:2]) + list(input_shape[-2:])
        x = x.reshape(output_shape)
        return x
