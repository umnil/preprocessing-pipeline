import numpy as np
from typing import List, Tuple


def array_to_clean_list(a: np.ndarray, axis=-1) -> List:
    """Given an array `a` that may or may not have been created using
    `equalize_list_to_array` convert to a list of arrays and remove any NaN
    Padding
    """
    masked_array: np.ma.core.MaskedArray = np.ma.masked_invalid(a)
    masked_list: List = [np.moveaxis(i, axis, 0) for i in masked_array]
    unmasked_lists = [i.data[~i.mask[:, 0].squeeze()] for i in masked_list]
    unmasked_lists = [np.moveaxis(i, 0, axis) for i in unmasked_lists]
    return unmasked_lists


def equalize_list_to_array(a: List[np.ndarray], axes: List[int] = [-1]) -> np.ndarray:
    """Given a list of ragged numpy arrays, this function fills the missing
    data with NaN to return an array with squre dimensions

    Parameters
    ----------
    a : List
        The list of other numpy arrays
    axes : List[int]
        The axes within the arrays along which NaNs are filled

    Return
    ------
    np.ndarray
        The filled and stacked array
    """
    max_lens: List[int] = [max([i.shape[axis] for i in a]) for axis in axes]
    a = [
        np.pad(
            i.astype(np.float32),
            generate_padding_param(i, max_lens, axes),
            "constant",
            constant_values=np.nan,
        )
        for i in a
    ]
    return np.stack(a)


def equalize_shape(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """This is a convenience function for ensuring that each element along
    a given axis of `a` is sized to the amount equal to the element with
    the largest size. NaN's are used as placeholders. Since this function
    can be called more than once, Nan's are first stripped if they exist.

    Parameters
    ----------
    a : np.ndarray
        The multidimensional input array

    axis : int
        The axis along which to modify & equalize

    Return
    ------
    np.ndarray
    """
    a = np.moveaxis(a, axis, -1)

    cur_len: int = a.shape[-1]
    preserved_sizes: Tuple = a.shape[:-1]
    a = a.reshape(-1, cur_len)

    list_a: List = [i[~np.isnan(i)] for i in a]
    a = equalize_list_to_array(list_a)

    a = a.reshape(*preserved_sizes, -1)
    a = np.moveaxis(a, -1, axis)
    return a


def generate_padding_param(
    a: np.ndarray, max_lens: List[int], axes: List = [-1]
) -> List:
    """Create a set of tuples for use in np.pad

    Parameters
    ----------
    a : np.ndarray
        The array to check the shape of
    max_lens : List[int]
        Maximum expected lengths for each axis provided
    axes : List
        List of axes to pad
    """
    axes = [i if i >= 0 else (a.ndim + i) for i in axes]
    assert np.array([a >= 0 for a in axes]).all()

    # Populate lens
    lens = [0] * a.ndim
    for i, axis in enumerate(axes):
        lens[axis] = max_lens[i] - a.shape[axis]

    return [[0, len] for i, len in enumerate(lens)]
