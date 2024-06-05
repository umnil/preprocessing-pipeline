import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore
from typing import Optional


class MaskedStandardScaler(StandardScaler):
    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        axis: int = -1
    ):
        super(MaskedStandardScaler, self).__init__(
            copy=copy, with_mean=with_mean, with_std=with_std
        )
        self.axis: int = axis

    def fit(self, x: np.ndarray, y: Optional[np.ndarray]) -> "MaskedStandardScaler":
        x = np.moveaxis(x, self.axis, -1)
        x = x.reshape(-1, x.shape[-1])
        return super(MaskedStandardScaler, self).fit(x, y)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.moveaxis(x, self.axis, -1)
        pre_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = super(MaskedStandardScaler, self).transform(x)
        x = x.reshape(pre_shape)
        x = np.moveaxis(x, -1, self.axis)
        return x
