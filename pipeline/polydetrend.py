import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import PolynomialFeatures  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore


class PolyDetrend(TransformerMixin, BaseEstimator):
    def __init__(self, degree: int = 5):
        """Polynomial detrending

        Parameters
        ----------
        n : int
            degree of polynomial fit
        """
        self.degree: int = degree
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=True)),
                ("linear", LinearRegression(fit_intercept=False)),
            ]
        )

    def fit(self, x: np.ndarray, *args, **kwargs) -> "PolyDetrend":
        return self

    def transform(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        self.t: np.ndarray = np.arange(x.shape[-1])[..., None]
        input_shape: Tuple = x.shape
        x = x.reshape(-1, input_shape[-1])
        self.fs: List = []
        out: List = []
        for i in x:
            xi: np.ndarray = i[..., None]
            self.pipeline.fit(self.t, xi)
            f: np.ndarray = self.pipeline.predict(self.t)
            self.fs.append(f)
            out.append((xi - f).squeeze())

        x = np.stack(out)
        x = x.reshape(input_shape)
        return x
