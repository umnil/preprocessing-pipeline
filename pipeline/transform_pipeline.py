import pickle
import numpy as np

from typing import Tuple, IO, Dict, Any, List, Optional
from datetime import datetime

from sklearn.base import clone  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.utils import _print_elapsed_time  # type: ignore
from sklearn.utils.metaestimators import available_if  # type: ignore
from sklearn.utils.validation import check_memory  # type: ignore


def _final_estimator_has(attr):
    """Check that final_estimator has `attr`.
    Used together with `avaliable_if` in `Pipeline`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


class TransformPipeline(Pipeline):
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps: List = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, y, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params):
        """Fit the model.
        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : np.ndarray
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : np.ndarray, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt: np.ndarray
        yt: np.ndarray
        Xt, yt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.
        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt, yt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                return last_step.fit(Xt, yt, **fit_params_last_step).transform(Xt)

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        """Transform the data, and apply `score` with the final estimator.
        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.
        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        Xt = X
        yt = y
        for _, name, transform in self._iter(with_final=False):
            if hasattr(transform, "_y_hat"):
                Xt = transform.fit_transform(Xt, yt)
            else:
                Xt = transform.transform(Xt)
            yt = (
                yt if not hasattr(transform, "_y_hat") else getattr(transform, "_y_hat")
            )
            # Check for nan
            if np.isnan(yt.astype(np.float64)).any():
                nan_mask = np.isnan(yt)
                Xt = Xt[~nan_mask]
                yt = yt[~nan_mask]

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.steps[-1][1].score(Xt, yt, **score_params)

    def transform(
        self, X: np.ndarray, y: np.ndarray, debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        preform a transformation on both the data and the labels. Useful for
        when labels need to be removed with data or when data is split up and
        labels can be removed or added.

        NOTE: for steps in the pipeline where the labels `y` is altered, it
        must store the value in a property called `_y_hat`

         Parameters
        ----------
        X : np.ndarray
            The input data set
        y : np.ndarray
            The original target values for the data set
        debug : bool
            if true, expections will save information suche as original data,
            current data, and the step at which the pipeline failed

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The transformed X and y values
        """
        self.X_original: np.ndarray = X
        self.y_original: np.ndarray = y
        self.results: List[Tuple] = []

        X_hat: np.ndarray
        y_hat: np.ndarray
        for name, step in self.steps:
            X_hat, y_hat = self._step_transform(step, X, y, debug)

            # Check for nan
            if np.isnan(y_hat.astype(np.float64)).any():
                nan_mask = np.isnan(y_hat)
                X_hat = X_hat[~nan_mask]
                y_hat = y_hat[~nan_mask]

            # Validation
            assert X_hat.shape[0] == y_hat.shape[0], (
                f"X_hat rows = {X_hat.shape[0]}"
                + f", while y_hat rows = {y_hat.shape[0]}"
                + f".\nStep: {name}"
                + f".\n\n preprocessor: {self}"
            )
            dims: List = list(X_hat.shape)
            for i, dim in enumerate(dims):
                assert dim > 0, (
                    f"The {name} step (step #{step}) of the pipeline "
                    + f"depleted the values in dimension {i} of the data "
                    + "matrix"
                )

            self.results.append((X_hat, y_hat))
            X, y = X_hat, y_hat

        return (X_hat, y_hat)

    def _step_transform(
        self, step: Any, X: np.ndarray, y: np.ndarray, debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preform the actual data transformation at the given step"""
        X_hat: np.ndarray
        y_hat: np.ndarray
        try:
            X_hat = step.fit_transform(X, y)
            y_hat = y if not hasattr(step, "_y_hat") else getattr(step, "_y_hat")

            return (X_hat, y_hat)
        except Exception as e:
            if debug:
                data_artifact: Dict = {
                    "initial_data": {"X": self.X_original, "y": self.y_original},
                    "input_data": {"X": X, "y": y},
                    "pipeline": self,
                    "step": step,
                }
                cur_time: datetime = datetime.now()
                cur_time_str: str = cur_time.strftime("%Y%m%d%H%M%S")
                data_filename: str = f"pipeline_error_{cur_time_str}.p"

                fh: IO[bytes]
                with open(data_filename, "wb") as fh:
                    pickle.dump(data_artifact, fh)

            raise e


def _fit_transform_one(
    transformer, X, y, weight, message_clsname="", message=None, **fit_params
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            res_x = transformer.fit_transform(X, y, **fit_params)
            res_y = (
                y
                if not hasattr(transformer, "_y_hat")
                else getattr(transformer, "_y_hat")
            )
        else:
            res_x = transformer.fit(X, y, **fit_params).transform(X)
            res_y = (
                y
                if not hasattr(transformer, "_y_hat")
                else getattr(transformer, "_y_hat")
            )

        # Check for nan
        if np.isnan(res_y.astype(np.float64)).any():
            nan_mask = np.isnan(res_y)
            res_x = res_x[~nan_mask]
            res_y = res_y[~nan_mask]

    if weight is None:
        return res_x, res_y, transformer
    return res_x * weight, res_y, transformer
