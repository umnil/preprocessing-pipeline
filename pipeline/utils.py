import numpy as np
from typing import List, Tuple


def equalize_list_to_array(a: List[np.ndarray], axis: int = 0) -> np.ndarray:
    """Given a list of ragged numpy arrays, this function fills the missing
    data with NaN to return an array with squre dimensions

    Parameters
    ----------
    a : List
        The list of other numpy arrays
    axis : int
        The axis within the arrays along which NaNs are filled

    Return
    ------
    np.ndarray
        The filled and stacked array
    """
    max_len: int = max([i.shape[axis] for i in a])
    a = [
        np.pad(
            i.astype(np.float32),
            generate_padding_param(i, max_len, axis),
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
    a = equalize_list_to_array(list_a, axis)

    a = a.reshape(*preserved_sizes, -1)
    a = np.moveaxis(a, -1, axis)
    return a


def generate_padding_param(a: np.ndarray, max_len: int, axis: int = 0) -> List:
    """Create a set of tuples for use in np.pad

    Parameters
    ----------
    a : np.ndarray
        The array to check the shape of
    max_len : int
        Maximum expected length
    axis : int
        The axis along which to create padding
    """
    axis = axis if axis >= 0 else (a.ndim + axis)
    assert axis >= 0
    return [[0, max_len - a.shape[axis] if i == axis else 0] for i in range(a.ndim)]
