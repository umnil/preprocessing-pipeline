import inspect
import pickle
import numpy as np

from datetime import datetime
from importlib.metadata import version
from typing import Dict, IO, Sequence, List, Optional, Tuple, cast

from sklearn.base import TransformerMixin, clone  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.utils.metaestimators import available_if  # type: ignore
from sklearn.utils.validation import check_memory  # type: ignore
from .utils import (
    split_mask_xy,
    reform_mask_xy,
    transformer_split_mask,
    transformer_reform_mask,
)

scikit_version = float(".".join(version("scikit-learn").split(".")[:2]))

if scikit_version > 1.4:
    from sklearn.utils.metadata_routing import (  # type: ignore
        _raise_for_params,
        _routing_enabled,
        process_routing,
    )
else:
    from sklearn.utils import _print_elapsed_time  # type: ignore

if scikit_version > 1.4:
    from sklearn.utils._user_interface import _print_elapsed_time  # type: ignore


def _final_estimator_has(attr):
    """Check that final_estimator has `attr`.
    Used together with `avaliable_if` in `Pipeline`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


class TransformPipeline(Pipeline):
    def _fit(
        self,
        X: Sequence,
        y: Optional[Sequence] = None,
        routed_params: Optional[Dict] = None,
    ) -> Tuple:
        # shallow copy of steps - this should really be steps_
        self.steps: List = list(self.steps)
        self.results: List[Tuple] = []
        self._y_lengths: List[int] = []
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

            cloned_transformer = transformer_split_mask(cloned_transformer)

            X, y = split_mask_xy(X, y)
            # Fit or load from cache the current transformer
            X, y, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=routed_params[name] if routed_params is not None else None,
            )
            X, y = reform_mask_xy(X, y)
            fitted_transformer = transformer_reform_mask(fitted_transformer)
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
            self.results.append((X, y))
            if hasattr(fitted_transformer, "_y_lengths"):
                self._y_lengths = getattr(fitted_transformer, "_y_lengths")
        return X, y

    def fit(self, X: Sequence, y: Optional[Sequence] = None, **params):
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
        **params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        routed_params = self._check_method_params(method="fit", props=params)
        Xt: np.ndarray
        yt: np.ndarray
        Xt, yt = self._fit(X, y, routed_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                if hasattr(self._final_estimator, "use_lengths"):
                    last_step_params.update({"lengths": self._y_lengths})
                self._final_estimator.fit(Xt, yt, **last_step_params["fit"])

        return self

    def fit_transform(
        self, X: Sequence, y: Optional[Sequence] = None, **params
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
        **params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : Sequence
            Typically ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        if scikit_version >= 1.4:
            routed_params = self._check_method_params(
                method="fit_transform", props=params
            )
        else:
            routed_params = self._check_fit_params(**params)

        Xt, yt = self._fit(X, y, routed_params)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt, yt
            last_step_params = routed_params[self.steps[-1][0]]
            if hasattr(self, "use_lengths"):
                if self.use_lengths:
                    last_step_params.update({"lengths": self._y_lengths})
            if hasattr(last_step, "fit_transform"):
                if scikit_version >= 1.4:
                    x_hat = last_step.fit_transform(Xt, yt, **last_step_params["fit"])
                else:
                    x_hat = last_step.fit_transform(Xt, yt, **last_step_params)
            else:
                x_hat = last_step.fit(Xt, yt, **last_step_params["fit"]).transform(
                    Xt, **last_step_params["fit"]
                )
            self._y_hat = (
                yt if not hasattr(last_step, "_y_hat") else getattr(last_step, "_y_hat")
            )
            return x_hat

    @available_if(_final_estimator_has("predict"))
    def predict(self, x: Sequence, y: Optional[Sequence] = None, **params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
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
        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                has_y_param: bool = (
                    "y" in inspect.signature(transform.transform).parameters
                )
                if has_y_param and yt is not None:
                    xt = transform.transform(xt, yt)
                    yt = (
                        yt
                        if not hasattr(transform, "_y_hat")
                        else getattr(transform, "_y_hat")
                    )
                    self._y_lengths = (
                        self._y_lengths
                        if not hasattr(transform, "_y_lengths")
                        else getattr(transform, "_y_lengths")
                    )
                else:
                    xt = transform.transform(xt)
            self._y_hat = yt

            if hasattr(self.steps[-1][1], "use_lengths"):
                params.update({"lengths": self._y_lengths})
            return self.steps[-1][1].predict(xt, **params)

        # metadata routing enabled
        routed_params = process_routing(self, "predict", **params)
        for _, name, transform in self._iter(with_final=False):
            has_y_param = "y" in inspect.signature(transform.transform).parameters
            if has_y_param is not None:
                xt = transform.transform(xt, yt, **routed_params[name].transform)
                yt = (
                    yt
                    if not hasattr(transform, "_y_hat")
                    else getattr(transform, "_y_hat")
                )
                self._y_lengths = (
                    self._y_lengths
                    if not hasattr(transform, "_y_lengths")
                    else getattr(transform, "_y_lengths")
                )
            else:
                xt = transform.transform(xt, **routed_params[name].transform)
        if hasattr(self.steps[-1][1], "use_lengths"):
            params.update({"lengths": self._y_lengths})
        return self.steps[-1][1].predict(xt, **routed_params[self.steps[-1][0]].predict)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, x: Sequence, y: Optional[Sequence] = None, **params):
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        xt = x
        yt = y

        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                has_y_param: bool = (
                    "y" in inspect.signature(transform.transform).parameters
                )
                if has_y_param:
                    xt = transform.transform(xt, yt)
                    yt = (
                        yt
                        if not hasattr(transform, "_y_hat")
                        else getattr(transform, "_y_hat")
                    )
                    self._y_lengths = (
                        self._y_lengths
                        if not hasattr(transform, "_y_lengths")
                        else getattr(transform, "_y_lengths")
                    )
                else:
                    xt = transform.transform(xt)

            self._y_hat = yt
            if hasattr(self.steps[-1][1], "use_lengths"):
                params.update({"lengths": self._y_lengths})
            return self.steps[-1][1].predict_proba(xt, **params)

        # metadata routing enabled
        routed_params = process_routing(self, "predict_proba", **params)
        for _, name, transform in self._iter(with_final=False):
            has_y_param = "y" in inspect.signature(transform.transform).parameters
            if has_y_param:
                xt = transform.transform(xt, yt, **routed_params[name].transform)
                yt = (
                    yt
                    if not hasattr(transform, "_y_hat")
                    else getattr(transform, "_y_hat")
                )
                self._y_lengths = (
                    self._y_lengths
                    if not hasattr(transform, "_y_lengths")
                    else getattr(transform, "_y_lengths")
                )
            else:
                xt = transform.transform(xt, **routed_params[name].transform)

        self._y_hat = yt
        if hasattr(self.steps[-1][1], "use_lengths"):
            params.update({"lengths": self._y_lengths})
        return self.steps[-1][1].predict_proba(
            xt, **routed_params[self.steps[-1][0]].predict_proba
        )

    @available_if(_final_estimator_has("score"))
    def score(self, X: Sequence, y=None, sample_weight=None, **params):
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

        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                if yt is not None:
                    Xt = transform.fit_transform(Xt, yt)
                else:
                    Xt = transform.transform(Xt)
                yt = (
                    yt
                    if not hasattr(transform, "_y_hat")
                    else getattr(transform, "_y_hat")
                )
                self._y_lengths = (
                    self._y_lengths
                    if not hasattr(transform, "_y_lengths")
                    else getattr(transform, "_y_lengths")
                )

            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            if hasattr(self.steps[-1][1], "use_lengths"):
                score_params.update({"lengths": self._y_lengths})
            return self.steps[-1][1].score(Xt, yt, **score_params)

        # metadata routing enabled
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )
        for _, name, transform in self._iter(with_final=False):
            if yt is not None:
                Xt = transform.fit_transform(Xt, yt, **routed_params[name].transform)
            else:
                Xt = transform.transform(Xt, **routed_params[name].transform)
            yt = (
                yt if not hasattr(transform, "_y_hat") else getattr(transform, "_y_hat")
            )
            self._y_lengths = (
                self._y_lengths
                if not hasattr(transform, "_y_lengths")
                else getattr(transform, "_y_lengths")
            )

        if hasattr(self.steps[-1][1], "use_lengths"):
            params.update({"lengths": self._y_lengths})
        return self.steps[-1][1].score(Xt, yt, **routed_params[self.steps[-1][0]].score)

    def transform(
        self, x: Sequence, y: Sequence, debug: bool = False, **params
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
        _raise_for_params(params, self, "transform")

        # Store the original data
        self.x_original: Sequence = x
        self.y_original: Sequence = y
        self.results = []
        self._y_lengths = []

        xt: Sequence = x
        yt: Sequence
        routed_params = process_routing(self, "transform", **params)
        for idx, name, transform in self._iter():
            xt, yt = _transform_one(transform, x, y, debug=debug)
            xt = transform.transform(x, **routed_params[name].transform)
            yt = (
                yt if not hasattr(transform, "_y_hat") else getattr(transform, "_y_hat")
            )
            self._y_lengths = (
                self._y_lengths
                if not hasattr(transform, "_y_lengths")
                else getattr(transform, "_y_lengths")
            )
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

        self._y_hat = yt
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
        has_y_param: bool = "y" in inspect.signature(transformer.transform).parameters
        if has_y_param:
            res_x: Sequence = transformer.transform(x, y)
            res_y: Sequence = (
                y
                if not hasattr(transformer, "_y_hat")
                else getattr(transformer, "_y_hat")
            )
        else:
            res_x = transformer.transform(x)
            res_y = y
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
    transformer, X, y, weight, message_clsname="", message=None, params=None
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    params = params or {}
    X, y = reform_mask_xy(X, y)
    transformer = transformer_reform_mask(transformer)
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            res_x = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
            res_y = (
                y
                if not hasattr(transformer, "_y_hat")
                else getattr(transformer, "_y_hat")
            )
        else:
            res_x = transformer.fit(X, y, **params.get("fit", {})).transform(
                X, **params.get("transform", {})
            )
            res_y = (
                y
                if not hasattr(transformer, "_y_hat")
                else getattr(transformer, "_y_hat")
            )

    if weight is not None:
        res_x *= weight

    res_x, res_y = split_mask_xy(res_x, res_y)
    transformer = transformer_split_mask(transformer)
    return res_x, res_y, transformer
