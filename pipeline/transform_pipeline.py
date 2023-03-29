import pickle
import numpy as np

from typing import Tuple, IO, Dict, Any, List
from datetime import datetime

from sklearn.pipeline import Pipeline  # type: ignore


class TransformPipeline(Pipeline):
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
