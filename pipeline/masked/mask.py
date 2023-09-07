import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore


class Masker(TransformerMixin, BaseEstimator):
    def __init__(self, filler=None):
        self.filler = filler

    def fit(self, *args, **fit_params):
        return self

    def fit_transform(self, x, y=None, **fit_params):
        return self.transform(x, y, **fit_params)

    def transform(self, x, y=None, *args, **kwargs):
        if self.filler is None:
            ret = np.ma.masked_invalid(x)
            ret.data[ret.mask] = 0
            if y is not None:
                self._y_hat = np.ma.masked_invalid(y)
        else:
            ret = np.ma.masked_values(x, self.filler)
            if y is not None:
                self._y_hat = np.ma.masked_values(y, self.filler)

        ret.harden_mask()
        if y is not None:
            self._y_hat.harden_mask()
        return ret
