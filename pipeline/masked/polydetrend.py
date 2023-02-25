import numpy as np
from typing import List, Tuple
from ..polydetrend import PolyDetrend


class MaskedPolyDetrend(PolyDetrend):
    def transform(  # type: ignore
        self, x: np.ma.core.MaskedArray, *args, **kwargs
    ) -> np.ma.core.MaskedArray:
        input_shape: Tuple = x.shape
        t: int = input_shape[-1]
        x = x.reshape(-1, t)
        out: List = [
            super(MaskedPolyDetrend, self).transform(i[~i.mask].data[None, ...])
            for i in x
        ]
        out = [i.squeeze().tolist() + [np.nan] * (t - i.size) for i in out]
        x = np.stack(out)
        x = np.ma.masked_invalid(x)
        x = x.reshape(input_shape)
        return x
