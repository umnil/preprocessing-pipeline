import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore


class Masker(TransformerMixin, BaseEstimator):
    def __init__(self, filler=None):
        self.filler = filler

    def fit(self, *args, **kwargs):
        return self

    def transform(self, x, *args, **kwargs):
        if self.filler is None:
            ret = np.ma.masked_invalid(x)
            ret.data[ret.mask] = 0
        else:
            ret = np.ma.masked_values(x, self.filler)
        ret.harden_mask()

        def set_params(self, **params):
        # Set the parameters of the pipeline
            for filler, value in params.items():
                setattr(self, filler, value)
        return ret
