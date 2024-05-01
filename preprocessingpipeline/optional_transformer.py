import inspect
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import Sequence, Optional


class OptionalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator: BaseEstimator, active: bool = True):
        """An `OptionalTransformer` placed within a pipeline allows a grid
        search algorithm to selectively turn off or on the wrapped `estimator`.
        This enables pipelines to be optimized across pipeline features and
        explore preprocessing parameters as hyperparameters to the over all
        model of interest

        Parameters
        ----------
        estimator : BaseEstimator
            The scikit-learn estimator or transformer that will be optional
        active : bool
            When False, the estimator is simply the identify function in that
            the transformer acts as a passthrough
        """
        self.estimator: BaseEstimator = estimator
        self.active: bool = active

    def fit(
        self, x: Sequence, y: Optional[Sequence] = None, **fitparams
    ) -> "OptionalTransformer":
        """Fit the estimator if active"""
        if self.active:
            self.estimator.fit(x, y, **fitparams)

        return self

    def fit_transform(
        self, x: Sequence, y: Optional[Sequence] = None, **fit_params
    ) -> Sequence:
        if self.active:
            x = self.estimator.fit_transform(x, y)
            self._y_hat = (
                y
                if not hasattr(self.estimator, "_y_hat")
                else getattr(self.estimator, "_y_hat")
            )
            return x
        else:
            self._y_hat = y
            return x

    def transform(self, x: Sequence, y: Optional[Sequence] = None) -> Sequence:
        if self.active:
            has_y: bool = (
                "y" in inspect.signature(self.estimator.transform).parameters.keys()
            )
            if has_y:
                x = self.estimator.transform(x, y)
            else:
                x = self.estimator.transform(x)
            self._y_hat = (
                y
                if not hasattr(self.estimator, "_y_hat")
                else getattr(self.estimator, "_y_hat")
            )
            return x
        else:
            self._y_hat = (
                y
                if not hasattr(self.estimator, "_y_hat")
                else getattr(self.estimator, "_y_hat")
            )
            return x
