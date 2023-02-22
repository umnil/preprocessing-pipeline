import numpy as np
import pandas as pd  # type: ignore

from typing import List, Union, cast
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore


class ChExtractor(TransformerMixin, BaseEstimator):
    """Data packets from the BCI system are collected in dataframes. This class
    assists in the pipeline
    """

    def __init__(self, picks=[]):
        super(ChExtractor, self).__init__()
        self._data_column_name: str = "data"
        self.picks = picks

        # Uninitialized Variables
        self._X: np.ndarray

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

    def transform(self, X: Union[List, pd.DataFrame]) -> np.ndarray:
        """transform an input of trial data in the form of a DataFrame into a
        time-series feature set

        Parameters
        ----------
        X : Union[List, pd.DataFrame]
            A single or list of NXM table(s). N = number of packets. M = number
            of packet features

        Returns
        -------
        np.ndarray
            an NxM matrix where N = number of packets and M is the number of
            concatenated data points. M is the concatenated data from all
            channels in sequential order. Thus by using `self._channel_info`
            the data can be reseparated.
        """
        if type(X) is list:
            trial_results: List = [self.transform(x) for x in X]
            self._X = np.vstack(trial_results)
        else:
            self._data_column_name = self._resolve_data_column_name(X)
            self._channel_info = self._resolve_channel_info(X)
            data: pd.Series = cast(pd.DataFrame, X).loc[:, self._data_column_name]
            data = data.apply(lambda x: np.array([x[i] for i in self.picks]))
            self._X = np.stack(data.values)
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
