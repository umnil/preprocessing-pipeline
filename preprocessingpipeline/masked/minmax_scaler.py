import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from typing import Optional, Tuple


class MaskedMinMaxScaler(MinMaxScaler):
    def __init__(
        self,
        feature_range: Tuple = (0, 1),
        *,
        copy: bool = True,
        clip: bool = False,
        axis: int = -1
    ):
        super(MaskedMinMaxScaler, self).__init__(
            feature_range=feature_range, copy=copy, clip=clip
        )
        self.axis = axis

    def fit(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "MaskedMinMaxScaler":
        x = np.moveaxis(x, self.axis, -1)
        x = x.reshape(-1, x.shape[-1])
        return super(MaskedMinMaxScaler, self).fit(x, y)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.moveaxis(x, self.axis, -1)
        pre_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = super(MaskedMinMaxScaler, self).transform(x)
        x = x.reshape(pre_shape)
        x = np.moveaxis(x, -1, self.axis)
        return x
