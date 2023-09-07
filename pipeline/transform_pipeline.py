import inspect
import pickle
import numpy as np

from typing import Dict, IO, Sequence, List, Optional, Tuple, cast
from datetime import datetime

from sklearn.base import TransformerMixin, clone  # type: ignore
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
    def _fit(self, X: Sequence, y: Optional[Sequence] = None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps: List = list(self.steps)
        self.results: List[Tuple] = []
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
            self.results.append((X, y))
            if hasattr(fitted_transformer, "_y_lengths"):
                self._y_lengths = getattr(fitted_transformer, "_y_lengths")
        return X, y

    def fit(self, X: Sequence, y: Optional[Sequence] = None, **fit_params):
        """Fit the model.
        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : Sequence
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : Sequence, default=None
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

    def fit_transform(
        self, X: Sequence, y: Optional[Sequence] = None, **fit_params
    ) -> Sequence:
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
        Xt : Sequence
            Typically ndarray of shape (n_samples, n_transformed_features)
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
                x_hat = last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                x_hat = last_step.fit(Xt, yt, **fit_params_last_step).transform(Xt)
            self._y_hat = (
                yt if not hasattr(last_step, "_y_hat") else getattr(last_step, "_y_hat")
            )
            return x_hat

    @available_if(_final_estimator_has("predict"))
    def predict(self, x: Sequence, y: Optional[Sequence] = None, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        xt = x
        yt = y
        for _, name, transform in self._iter(with_final=False):
            has_y_param: bool = "y" in inspect.signature(transform.transform).parameters
            if has_y_param and yt is not None:
                xt = transform.transform(xt, yt)
                yt = (
                    yt
                    if not hasattr(transform, "_y_hat")
                    else getattr(transform, "_y_hat")
                )
            else:
                xt = transform.transform(xt)
        self._y_hat = yt

        return self.steps[-1][1].predict(xt, **predict_params)

    @available_if(_final_estimator_has("score"))
    def score(self, X: Sequence, y=None, sample_weight=None):
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
            if yt is not None:
                Xt = transform.fit_transform(Xt, yt)
            else:
                Xt = transform.transform(Xt)
            yt = (
                yt if not hasattr(transform, "_y_hat") else getattr(transform, "_y_hat")
            )

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.steps[-1][1].score(Xt, yt, **score_params)

    def transform(
        self, x: Sequence, y: Sequence, debug: bool = False
    ) -> Tuple[Sequence, Sequence]:
        """
        preform a transformation on both the data and the labels. Useful for
        when labels need to be removed with data or when data is split up and
        labels can be removed or added.

        NOTE: for steps in the pipeline where the labels `y` is altered, it
        must store the value in a property called `_y_hat`

         Parameters
        ----------
        x : Sequence
            The input data set
        y : Sequence
            The original target values for the data set
        debug : bool
            if true, expections will save information suche as original data,
            current data, and the step at which the pipeline failed

        Returns
        -------
        Tuple[Sequence, Sequence]
            The transformed X and y values
        """
        # Store the original data
        self.x_original: Sequence = x
        self.y_original: Sequence = y
        self.results = []

        xt: Sequence = x
        yt: Sequence
        for idx, name, transform in self._iter():
            xt, yt = _transform_one(transform, x, y, debug=debug)

            # Validation
            assert len(xt) == len(yt), (
                f"xt rows = {len(xt)}"
                + f", while yt rows = {len(yt)}"
                + f".\nStep: {name}"
                + f".\n\n preprocessor: {self}"
            )

            # Validate dimensions
            if isinstance(xt, np.ndarray):
                xta: np.ndarray = cast(np.ndarray, xt)
                dims: List = list(xta.shape)
                for i, dim in enumerate(dims):
                    assert dim > 0, (
                        f"The {name} step (step #{idx}) of the pipeline "
                        + f"depleted the values in dimension {i} of the data "
                        + "matrix"
                    )

            self.results.append((xt, yt))
            x, y = xt, yt

        return (xt, yt)


def _transform_one(
    transformer: TransformerMixin,
    x: Sequence,
    y: Sequence,
    weight: Optional[float] = None,
    **fit_params,
) -> Tuple[Sequence, Sequence]:
    debug: bool = fit_params.get("debug", False)
    try:
        res_x: Sequence = transformer.transform(x, y)
        res_y: Sequence = (
            y if not hasattr(transformer, "_y_hat") else getattr(transformer, "_y_hat")
        )
        return (res_x, res_y)
    except Exception as e:
        if debug:
            data_artifact: Dict = {
                "input_data": {"x": x, "y": y},
                "transformer": transformer,
            }
            cur_time: datetime = datetime.now()
            cur_time_str: str = cur_time.strftime("%Y%m%d%H%M%S")
            data_filename: str = f"pipeline_error_{cur_time_str}.p"

            fh: IO[bytes]
            with open(data_filename, "wb") as fh:
                pickle.dump(data_artifact, fh)

        raise e

    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res_x, res_y
    return res_x * weight, res_y


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

    if weight is None:
        return res_x, res_y, transformer
    return res_x * weight, res_y, transformer
