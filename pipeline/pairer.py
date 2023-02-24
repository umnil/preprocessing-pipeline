import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import List, Tuple


class Pairer(TransformerMixin, BaseEstimator):
    def __init__(self, pair_values: List[int] = [0, 1], force_even: bool = True):
        assert len(pair_values) == 2, "Only two pair values can be listed for pairing"
        self.pair_values: List[int] = pair_values
        self.force_even: bool = force_even
        self.cutoff: Optional[int] = None

    def fit(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        self.cutoff = None
        assert (
            x.shape[0] == y.shape[0]
        ), "x and y must have the same number of instances"

        first_label, second_label = self.pair_values
        labels, counts = np.unique(y, return_counts=True)
        idxs1: np.ndarray = np.where(y == first_label)[0]
        bounds: List = idxs1.tolist() + [y.size]
        segments: List = [y[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        idxs2_offsets: List = [np.where(s == second_label)[0] for s in segments]

        if self.force_even:
            if not np.all([i.size != 0 for i in idxs2_offsets]):
                self.cutoff = idxs1[-1]
                idxs1 = idxs1[:-1]
                idxs2_offsets = idxs2_offsets[:-1]

        assert np.all(
            [i.size != 0 for i in idxs2_offsets]
        ), f"Motor imagery labels are unbalanced: {counts}"

        np.all([i.size != 0 for i in idxs2_offsets]), (
            f"Data is not sequential. Expected the first label ({first_label}) to "
            f"precede the second ({second_label})"
        )

        idxs2: np.ndarray = np.array([a + b[0] for a, b in zip(idxs1, idxs2_offsets)])
        self.pairs: np.ndarray = np.c_[idxs1, idxs2]
        self._y_hat = np.array([[y[p1], y[p2]] for p1, p2 in self.pairs])

        return self

    def transform(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.cutoff is not None:
            i: int = self.cutoff
            x = x[:i]

        input_shape: Tuple = x.shape
        n_inst: int = input_shape[0]
        x = x.reshape(n_inst, -1)
        x = np.stack([[x[p1], x[p2]] for p1, p2 in self.pairs])
        output_shape: List = list(x.shape[:2]) + list(input_shape[-2:])
        x = x.reshape(output_shape)
        return x
