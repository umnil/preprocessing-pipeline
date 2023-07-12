import numpy as np

from sklearn.pipeline import FeatureUnion  # type: ignore
from .transform_pipeline import _transform_one, _fit_transform_one

try:
    from sklearn.utils.parallel import Parallel, delayed  # type: ignore
except ModuleNotFoundError:
    from joblib import Parallel, delayed  # type: ignore


class TransformFeatureUnion(FeatureUnion):
    def fit_transform(self, x, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        x : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        x_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        results = self._parallel_func(x, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            self._y_hat = y
            return np.zeros((x.shape[0], 0))

        xs, ys, transformers = zip(*results)
        self._update_transformer_list(transformers)

        self._y_hat = next(iter(ys))
        return self._hstack(xs)

    def transform(self, x, y=None):
        """Transform x separately by each transformer, concatenate results.

        Parameters
        ----------
        x : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : input

        Returns
        -------
        x_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.

        y_t : array-like
        """
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, x, y, weight)
            for name, trans, weight in self._iter()
        )
        xs = [r[0] for r in res]
        ys = [r[1] for r in res]
        if not xs:
            # All transformers are None
            self._y_hat = ys
            return np.zeros((x.shape[0], 0))

        self._y_hat = next(iter(ys))
        return self._hstack(xs)
