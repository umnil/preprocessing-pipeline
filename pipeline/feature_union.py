import numpy as np

from sklearn.pipeline import FeatureUnion  # type: ignore
from sklearn.utils.parallel import Parallel, delayed  # type: ignore
from .transform_pipeline import _transform_one


class TransformFeatureUnion(FeatureUnion):
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
            return np.zeros((x.shape[0], 0)), ys

        return self._hstack(xs), next(iter(ys))
