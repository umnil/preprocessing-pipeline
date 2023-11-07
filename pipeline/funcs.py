import mne  # type: ignore
import numpy as np
from functools import reduce
from mne import set_eeg_reference  # type: ignore
from typing import List, Optional, Tuple
from .masked import TemporalFilter
from .utils import get_frequency_bins, unmask_array


def channel_select(
    x: List[mne.io.Raw], channels: Optional[List] = None
) -> List[mne.io.Raw]:
    """Select only central channels from a set of EEG electrodes"""
    all_channel_names: List[str] = reduce(
        lambda a, b: np.intersect1d(a, b), [i.info.ch_names for i in x]
    ).tolist()
    eeg_channel_names: List[str] = [
        i for i in all_channel_names if i[0] in ["F", "C", "P", "T", "O"]
    ]
    selectable_channels: Optional[List] = (
        None if channels is None else [i for i in channels if i in all_channel_names]
    )

    return [
        i.copy().pick(
            picks=selectable_channels if channels is not None else eeg_channel_names
        )
        for i in x
    ]


def common_average_reference(x: List[mne.io.Raw]) -> List[mne.io.Raw]:
    """Apply common average reference to a list of mne raws"""
    return [set_eeg_reference(i.load_data())[0] for i in x]


def concat(x: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Special transformation for convenience when concatenating sessions and
    windows together in pipelines

    Parameters
    ----------
    x : np.ndarray
        The input data to concatenate
    y : np.ndarray
        The Label data to concatenate

    Returns
    -------
    x : np.ndarray
        The concatenated output data
    y : np.ndarray
        The concatenated output labels
    """
    active: bool = kwargs.get("active", False)
    keep_mask: bool = kwargs.get("keep_mask", False)
    x = np.moveaxis(x, 1, 2)
    if active:
        if keep_mask:
            x = x.reshape(-1, *x.shape[2:])
            y = y.flatten()
        else:
            x = np.concatenate(unmask_array(x)).astype(np.float64)
            y = np.hstack(unmask_array(y))
    return x.astype(np.float64), y.astype(np.float64)


def filter_bank(x: np.ndarray, bins: Optional[List[Tuple]] = None) -> np.ndarray:
    """Run the data, in parallel, through a set of band-pass filters

    Parameters
    ----------
    x : np.ndarray
        The input data
    bins : List[Tuple]
        A list of tuples that define the lower and high frequency bands of each
        band-pass filter

    Returns
    -------
    np.ndarray
        The filter bank output with a new dimension
    """
    _bins: List[Tuple] = get_frequency_bins() if bins is None else bins
    bank: List[TemporalFilter] = [
        TemporalFilter(sfreq=256, l_freq=lo, h_freq=hi) for lo, hi in _bins
    ]
    if isinstance(x, np.ma.core.MaskedArray):
        return np.ma.stack([f.fit_transform(x) for f in bank], axis=-3)
    else:
        return np.stack([f.fit_transform(x) for f in bank], axis=-3)


def good_channels(x: List[mne.io.Raw]) -> List[mne.io.Raw]:
    """A function to remove bad channels from a list of mne Raw data
    structures

    Parameters
    ----------
    x : List[mne.io.Raw]
        A list of mne data structures to remove bad channels from

    Returns
    -------
    List[mne.io.Raw]
        Same list as input but bad channels are removed
    """
    return [
        i.copy().pick(np.setdiff1d(i.info.ch_names, i.info["bads"]).tolist()) for i in x
    ]
