import numpy as np
from sklearn.base import TransformerMixin
from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast


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


def get_frequency_bins() -> List[Tuple]:
    """Return a constant value of frequency bounds

    Returns
    -------
    List[Tuple]
    """
    return [
        (0, 4),  # Theta
        (4, 8),  # Delta
        (8, 12),  # Alpha
        (12, 21),  # Low Beta
        (21, 30),  # High Beta
        (30, 45),  # Low Gamma
        (45, 70),  # Mid Gamma
        (70, 100),  # High Gamma
    ]


def mask_list(input_list: List) -> np.ndarray:
    """Given a list of input data, merge them to shared axes dimensions and pad
    with NaN where needed

    Parameters
    ----------
    input_list : List
        A list of numpy arrays to be masked and merged

    Returns
    -------
    np.ndarray
        The merged data array
    """
    contains_list: bool = True in [isinstance(li, Iterable) for li in input_list]
    if contains_list:
        array_list = [mask_list(i) for i in input_list]
        n_dim = array_list[0].ndim
        ret = equalize_list_to_array(array_list, axes=list(range(n_dim)))
        return ret
    else:
        return np.array(input_list)


def split_mask(value: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray, Tuple, type]:
    """Convert a masked array into a tuple with details to reform it"""
    data: np.ndarray = value.data.flatten().view(np.uint8)
    mask: np.ndarray = value.mask.flatten().astype(np.uint8).view(np.uint8)
    return data, mask, value.shape, value.dtype


def reform_mask(
    data: np.ndarray, mask: np.ndarray, shape: Tuple, dtype: type
) -> np.ma.MaskedArray:
    """Convert details of a split masked array back into a masked array"""
    return np.ma.MaskedArray(
        data.astype(np.uint8).view(dtype), mask.astype(np.uint8)
    ).reshape(shape)


def is_split_mask(val) -> bool:
    """Determine whether a tuple value has sufficient data of a split mask"""
    if isinstance(val, tuple):
        if len(val) == 4:
            a, b, c, d = val
            at = isinstance(a, np.ndarray)
            bt = isinstance(b, np.ndarray)
            ct = isinstance(c, tuple)
            dt = isinstance(d, type)
            if at and bt and ct and dt:
                return True

    return False


def split_mask_xy(
    X: Sequence, y: Optional[Sequence] = None
) -> Tuple[Union[Tuple, Sequence], Optional[Union[Tuple, Sequence]]]:
    """Convenience function for simultaneous mask splitting"""
    x_ret: Union[Tuple, Sequence] = X
    y_ret: Optional[Union[Tuple, Sequence]] = y
    if isinstance(X, np.ma.MaskedArray):
        x_ret = split_mask(X)

    if isinstance(y, np.ma.MaskedArray):
        y_ret = split_mask(y)

    return x_ret, y_ret


def reform_mask_xy(
    X: Sequence, y: Optional[Sequence] = None
) -> Tuple[Sequence, Optional[Sequence]]:
    """Convenience function for simultaneous mask reform"""
    x_ret: Sequence = X
    y_ret: Optional[Sequence] = y
    if isinstance(X, tuple):
        x_ret = cast(Sequence, reform_mask(*X))

    if isinstance(y, tuple):
        y_ret = cast(Sequence, reform_mask(*y))

    return x_ret, y_ret


def transformer_split_mask(transformer: TransformerMixin) -> TransformerMixin:
    """Transform all mask attributes in an object"""
    attr_names: List[str] = [i for i in dir(transformer) if hasattr(transformer, i)]
    ma_attr_names: List[str] = [
        i for i in attr_names if isinstance(getattr(transformer, i), np.ma.MaskedArray)
    ]
    for attr_name in ma_attr_names:
        setattr(transformer, attr_name, split_mask(getattr(transformer, attr_name)))
    return transformer


def transformer_reform_mask(transformer: TransformerMixin) -> TransformerMixin:
    """Transform all split masks back into masks"""
    attr_names: List[str] = [i for i in dir(transformer) if hasattr(transformer, i)]
    ma_attr_names: List[str] = [
        i for i in attr_names if is_split_mask(getattr(transformer, i))
    ]
    for attr_name in ma_attr_names:
        setattr(transformer, attr_name, reform_mask(getattr(transformer, attr_name)))
    return transformer


def unmask_array(a: np.ndarray) -> List:
    """Given an array `a` that may or may not have been created using
    `equalize_list_to_array` convert to a list of arrays and remove any NaN
    Padding

    Parameters
    ----------
    a : np.ndarray
        The input array with masked NaN values

    Returns
    -------
    List
        A list of numpy arrays with the dimensions pulled out and masked Nans cleared
    """
    ma: np.ma.core.MaskedArray = np.ma.masked_invalid(a)
    ma_list: List = ma.tolist()
    return unmask_list(ma_list)


def unmask_list(input_list: List) -> List:
    """Recursively unmask all arrays in a list

    Parameters
    ----------
    input_list : List
        Input arrays

    Returns
    -------
    List
        List of list of unmasked data
    """
    result: List = []
    for i in input_list:
        if isinstance(i, list):
            i = unmask_list(i)
            if len(i) != 0:
                result.append(i)
        else:
            if i is not None:
                result.append(i)

    return result
