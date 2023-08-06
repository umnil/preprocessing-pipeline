import numpy as np
from typing import List, Optional


def list_to_masked(x: List[np.ndarray]) -> np.ma.core.MaskedArray:
    """Given a list of numpy arrays convert all arrays into masked arrays with
    shared dimensions and returning with a concatenated ragged masked array

    Parameters
    ----------
    x : List[np.ndarray]
        A list of numpy arrays that may have different shapes

    returns
    -------
    np.ma.core.MaskedArray
        A masked array where valeus are masked on arrays that had smaller
        dimensions
    """
    max_dims: int = max([i.ndim for i in x])
    dimmed_x: List = [ensure_dims(i, max_dims) for i in x]
    max_shape: List[int] = [max([s.shape[i] for s in x]) for i in range(max_dims)]

    for a, max_len in enumerate(max_shape):
        cur_len: List[int] = [i.shape[a] for i in x]
        diff_len: List[int] = [max_len - l for i, l in zip(x, cur_len)]

        # Place the axis of interest in the back
        x_hat: List = [np.moveaxis(i, a, -1) for i in x]
        sizes: List[int] = [
            int(np.prod(list(i.shape)[:-1]) * d) for i, d in zip(x_hat, diff_len)
        ]
        residules: List = [np.ones(s) * np.nan for s in sizes]
        shaped_residules: List = [
            r.reshape(list(i.shape[:-1]) + [-1]) for r, i in zip(residules, x_hat)
        ]
        x_hat: List = [
            np.concatenate([i, r], axis=-1) for i, r in zip(x_hat, shaped_residules)
        ]
        x_hat: List = [np.moveaxis(i, -1, a) for i in x_hat]
        x = x_hat

    return np.ma.masked_invalid(np.stack(x))


def ensure_dims(x: np.ndarray, dims: int) -> np.ndarray:
    """Ensures that the input value `x` has the expected number of
    dimensions
    """
    while x.ndim < dims:
        x = x[..., None]
    return x
