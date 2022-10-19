import numpy as np
import pandas as pd  # type: ignore

from typing import List, Dict, Callable, Tuple
from functools import partial
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
from mne.time_frequency import psd_array_multitaper  # type: ignore


def psd(*args, **kwargs) -> Tuple:
    result, freqs = psd_array_multitaper(*args, **kwargs)
    result = 20 * np.log10(result + 1e-15)
    return (result, freqs)


class Featurizer(TransformerMixin, BaseEstimator):
    """Create features for the channels of interest"""

    def __init__(
        self,
        spectral_channels: List[int] = [0, 2],
        static_channels: List[int] = [1, 3],
        window_channel_sizes: List[int] = [640, 16, 640, 16],
        spectral_func: Callable = psd,
        spectral_func_args: Dict = {
            "sfreq": 200,
            "bandwidth": 3 * 2 * 200 / 640,
            "adaptive": True,
            "low_bias": True,
            "normalization": "full",
            "verbose": 50,
        },
        static_func: Callable = np.median,
        static_func_args: Dict = {"axis": 1},
    ):
        """Initialize featurizer properties

        Parameters
        ----------
        spectral_channels : List[int]
            A list of channel indicies that should be treated for spectral
            featurization
        static_channels : List[int]
            A list of channel indicies that should be treated as static
            aggregate analysis
        window_channel_sizes : List[int]
            A list of channel window lengths
        spectral_func : Callable
            The function to call for spectral featurization, this function
            should return a tuple of frequencies and spectral estimates
        spectral_func_args : Dict
            Other function arguments for running the spectral function
        static_func : Callable
            Function to processes static channels
        statif_func_args : Dict
            A keyword arugment list to pass to static_func
        """
        self.spectral_channels: List[int] = spectral_channels
        self.static_channels: List[int] = static_channels
        self.window_channel_sizes: List[int] = window_channel_sizes
        self.spectral_func: Callable = spectral_func
        self.spectral_func_args: Dict = spectral_func_args
        self.static_func: Callable = static_func
        self.static_func_args: Dict = static_func_args

        # Uninitialized variables set elsewhere
        self._freqs: np.ndarray
        self._feature_name_list: List = [0] * len(self.window_channel_sizes)
        self._X: np.ndarray

    @property
    def f_spec(self) -> Callable:
        return partial(self.spectral_func, **self.spectral_func_args)

    @property
    def f_stat(self) -> Callable:
        return partial(self.static_func, **self.static_func_args)

    def fit(self, X: np.ndarray, *args, **kwargs) -> "Featurizer":
        """Does nothing

        Parameters
        ----------
        X : np.ndarray
            The input data

        Returns
        -------
        Featurizer
            The same object
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Runs the filter on the data"""
        n_channels: int = len(self.window_channel_sizes)
        result: np.ndarray = np.array([])
        for channel_idx in range(n_channels):
            channel_data: np.ndarray = self._extract_channel_data(X, channel_idx)

            transformed_data: np.ndarray = self._transform_channel(
                channel_idx, channel_data
            )

            result = (
                np.c_[result, transformed_data]
                if result.shape[0] != 0
                else transformed_data
            )

        self._X = result
        return self._X

    def _extract_channel_data(self, X: np.ndarray, channel_idx: int) -> np.ndarray:
        """Extract the appropraite channel data from the array

        Parameters
        ----------
        X : np.ndarray
            The input array

        channel_idx : int
            The index of the channel to extract from `X`

        Returns
        -------
        np.ndarray
            The channel data
        """
        prior_channel_lengths: List[int] = self.window_channel_sizes[:channel_idx]
        inclusive_channel_lengths: List[int] = self.window_channel_sizes[
            : channel_idx + 1
        ]
        channel_start_idx: int = sum(prior_channel_lengths)
        channel_end_idx: int = sum(inclusive_channel_lengths)
        return X[:, channel_start_idx:channel_end_idx]

    def _transform_channel(
        self, channel_idx: int, channel_data: np.ndarray
    ) -> np.ndarray:
        """Transform channel data and returned the transformed data based on
        whether the channel is apart of spectral or static processing

        Parameters
        ----------
        channel_idx : int
            The index of the channel
        channel_data : np.ndarray
            Current channel data

        Returns
        -------
        np.ndarray
            Transformed channel data
        """
        result: np.ndarray
        feature_names: List[str] = []
        if channel_idx in self.spectral_channels:
            result, self._freqs = self.f_spec(channel_data)
            feature_names = [f"ch{channel_idx+1}_{x:0.3}" for x in self._freqs]
        elif channel_idx in self.static_channels:
            result = self.f_stat(channel_data)
            feature_names = [f"ch{channel_idx+1}_static"]

        self._feature_name_list[channel_idx] = feature_names
        return result

    def to_long(self, y: np.ndarray) -> pd.DataFrame:
        """
        Convert the transformed featurized data of this featurizer into a long
        format data frame

        Parameters
        ----------
        y : np.ndarray
            The labels to assign the current output rows of this featurizer

        Returns
        ----------
        pd.DataFrame
            A long format data frame
        """
        feature_names: List = np.hstack(self._feature_name_list).tolist()
        columns: List = ["prompt"] + feature_names
        main_data_frame: pd.DataFrame = pd.DataFrame(np.c_[y, self._X], columns=columns)
        main_long: pd.DataFrame = main_data_frame.reset_index().melt(
            id_vars=["index", "prompt"], var_name="freq"
        )
        main_long["channel"] = main_long["freq"].apply(lambda x: int(x[2:3]))
        main_long["freq"] = main_long["freq"].apply(
            lambda x: float(x[4:]) if "static" not in x else -1
        )
        return main_long
