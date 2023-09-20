import numpy as np
import pandas as pd  # type: ignore

from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
from typing import List, Optional, Union, cast


class Extractor(TransformerMixin, BaseEstimator):
    """Data packets from the BCI system are collected in dataframes. This class
    assists in the pipeline by extracting the raw data from these dataframes
    for use in scikit-learn and MNE pipelines
    """

    def __init__(self, picks: List[int] = [], y_column: str = "prompt"):
        super(Extractor, self).__init__()
        self._data_column_name: str = "data"
        self.picks: List[int] = picks
        self.y_column: str = y_column

        # Uninitialized Variables
        self._X: np.ndarray
        self._y_hat: np.ndarray

    def fit(self, X: Union[List, pd.DataFrame], *args, **kwargs) -> "Extractor":
        """Fit the data to be transfomred

        Parameters
        ----------
        X : Union[List, pd.DataFrame]
            A single or list of NxM data frame(s) where N is the number of
            packets, and M are the columns of the dataframe from a collected
            BCI session

        Returns
        -------
        Extractor
            The object with the data fit
        """
        return self

    def transform(
        self, X: Union[List, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """transform an input of trial data in the form of a DataFrame into a
        multi dimensional array of (N x C x T) where:
          - N = Number of Epochs or Instances
          - C = Number of Channels
          - T = Number of time points

        If the sampling rates for the channels are different, then channels
        with lower sampling frequencies are appended with np.nan

        Parameters
        ----------
        X : Union[List, pd.DataFrame]
            A single or list of NXM table(s). N = number of packets. M = number
            of packet features

        y : Optional[np.ndarray]
            Optional input. Ignored

        Returns
        -------
        np.ndarray
            an NxCXxT matrix as described above
        """
        if type(X) is list:
            self._X = self.transform_list(X, y)
        else:
            self._X = self.transform_dataframe(X, y)

        return self._X

    def transform_dataframe(
        self, x: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Performs transformation of a dataframe"""
        self._data_column_name = self._resolve_data_column_name(x)
        self._channel_info = self._resolve_channel_info(x)
        self._picked_channel_info = [self._channel_info[i] for i in self.picks]
        to_ragged: bool = np.unique(self._picked_channel_info).size > 1
        data = cast(pd.DataFrame, x)["data"].values
        if to_ragged:
            max_size = np.max(self._picked_channel_info)
            diff_channel_info = [
                max_size - self._channel_info[i] for i in range(len(self._channel_info))
            ]
            ret = np.array(
                [
                    [
                        ch + [np.nan] * diff_channel_info[i]
                        for i, ch in enumerate(pkt)
                        if i in self.picks
                    ]
                    for pkt in data
                ]
            )
        else:
            ret = np.array(
                [[ch for i, ch in enumerate(pkt) if i in self.picks] for pkt in data]
            )
        self._X = ret

        # Ensure appropriate shape
        if y is None:
            y = x[self.y_column].values

        self._y_hat = np.repeat(y[..., None], self._X.shape[-1], axis=-1).flatten()[
            None, ...
        ]
        self._X = np.swapaxes(self._X, 0, 1)
        self._X = self._X.reshape(self._X.shape[0], -1)[None, ...]

        return self._X

    def transform_list(
        self, x: List[pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Performs the transformation on a list of dataframes"""
        trial_results: List = []
        for i in x:
            xt: np.ndarray = self.transform_dataframe(i)
            yt: np.ndarray = self._y_hat
            trial_results.append((xt, yt))

        self._X = np.concatenate([i[0] for i in trial_results])
        self._y_hat = np.concatenate([i[1] for i in trial_results])

        return self._X

    def _resolve_data_column_name(self, X: pd.DataFrame) -> str:
        """Using the data frame provided, determine the name of the column that
        is storing the data for each packet.

        Parameters
        ----------
        X : pd.DataFrame
            NxM table, where N is the number of packets, M is the number of
            features about the packet, one of which is the raw data.

        Returns
        -------
        str
            The name of the column containing the data
        """
        possible_names: List[str] = ["data", "raw_data"]
        for possible_name in possible_names:
            if possible_name in X.columns:
                return possible_name

        raise Exception("Could not determine the appropriate column name for the data")

    def _resolve_channel_info(self, X: pd.DataFrame) -> np.ndarray:
        """Determine the channel info in the data of the dataframe

        Parameters
        ----------
        X : pd.DataFrame
            NxM table as used in other methods

        Returns
        -------
        np.ndarray
            A 1xN array where N is the number channels, and each element is
            equal to the number of data points in that channel for a packet
        """
        packet: pd.Series = X.iloc[0, :]
        data: List = packet[self._data_column_name]
        result: List[int] = [len(ch) for ch in data]

        return np.array(result)
