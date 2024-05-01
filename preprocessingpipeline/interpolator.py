import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import Callable, List, Union, Tuple, cast


class Interpolator(TransformerMixin, BaseEstimator):
    def __init__(
        self, t_max: int, method: Union[Callable, str] = "mean", axis: int = -1
    ):
        """Initialize an interpolator object that will interpolate data along a
        given axis

        Parameters
        ----------
        t_max : int
            The maximum value that the dimension should expect to have
        method : Union[Callable, str]
            A function or string that defines how the masked axis will be
            conformed
        axis : int
            The axis dimension that will be interpolated
        """
        if isinstance(method, Callable):  # type: ignore
            self.method: Callable = cast(Callable, method)
        else:
            self.method = np.__getattribute__(cast(str, method))
        self.axis: int = axis
        self.t_max: int = t_max

    def fit(self, x: np.ma.core.MaskedArray, *args, **kwargs) -> "Interpolator":
        """Fit a set of data to the iterpolator

        Parameters
        ----------
        x : np.ma.core.MaskedArray
            A masked numpy array

        Returns
        -------
        Interpolator
            The fitted interpolator
        """
        if (self.axis != -1) or (self.axis != x.ndim - 1):
            x = cast(
                np.ma.core.MaskedArray, np.moveaxis(cast(np.ndarray, x), self.axis, -1)
            )
        return self

    def transform(self, x: np.ma.core.MaskedArray, *args, **kwargs) -> np.ndarray:
        """Interpolate

        Parameters
        ----------
        x : np.ma.core.MaskedArray
            The input masked array

        Returns
        -------
        np.ndarray
            The interpolated unmasked array
        """
        # Move the interpolation axis to the end
        moved_axis: bool = False
        if (self.axis != -1) or (self.axis != x.ndim - 1):
            moved_axis = True
            x = cast(
                np.ma.core.MaskedArray, np.moveaxis(cast(np.ndarray, x), self.axis, -1)
            )

        # Force to 2D
        original_shape: Tuple = x.shape
        x = x.reshape(-1, original_shape[-1])
        self.sizes: List[int] = [i.data[~i.mask].shape[self.axis] for i in x]
        self.dim_size: int = int(np.round(self.method(self.sizes)))

        x_hat: np.ndarray = np.array(
            [
                np.interp(
                    np.linspace(0, self.t_max, self.dim_size),
                    np.linspace(0, self.t_max, i.data[~i.mask].shape[-1]),
                    i.data[~i.mask],
                )
                for i in x
            ]
        )

        # Reshape back to original dimensions
        x_hat = x_hat.reshape(*original_shape[:-1], self.dim_size)

        # Move axis back
        if moved_axis:
            x_hat = np.moveaxis(x_hat, -1, self.axis)

        self._x_hat = x_hat
        return self._x_hat
