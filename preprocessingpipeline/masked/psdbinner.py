import numpy as np

from ..psdbinner import PSDBinner
from typing import List, Tuple


class MaskedPSDBinner(PSDBinner):
    def transform(  # type: ignore
        self, x: np.ma.core.MaskedArray, *args, **kwargs  # type: ignore
    ) -> np.ndarray:  # type: ignore
        if not isinstance(x, np.ma.core.MaskedArray):
            return super(MaskedPSDBinner, self).transform(x, *args, **kwargs)

        # If only one dimension is raggid and no nans just reapply the mask
        n_diff_axes = sum([not x.mask.prod(axis=i).any() for i in range(x.mask.ndim)])
        if n_diff_axes < 2:
            input_mask: np.ndarray = x.mask.copy()
            x = super(MaskedPSDBinner, self).transform(x.data, **kwargs)
            output_mask: np.ndarray = input_mask[..., : x.shape[-1]]
            x = np.ma.MaskedArray(x, output_mask)
            return x

        input_shape: Tuple = x.shape
        output_shape: List = list(input_shape[:-1]) + [len(self.bins)]
        x = x.reshape(-1, input_shape[-1])
        list_x = [i[~i.mask] for i in x]
        dt: List = [i.shape[-1] for i in list_x]
        self.freqs = [np.linspace(0, self.sfreq / 2, i) for i in dt]
        freq_idxs: List = [
            [np.where((f >= lo) & (f < h))[0] for lo, h in self.bins]
            for i, f in zip(x, self.freqs)
        ]
        np_x = np.array(
            [
                [self._fn(xi[f], i, f=self.freqs[f]) for f in fi]
                for i, (xi, fi) in enumerate(zip(x, freq_idxs))
            ]
        )
        np_x = np_x.reshape(output_shape)
        if len(self.select_bins) > 0:
            np_x = np_x[..., self.select_bins]
        return np_x
