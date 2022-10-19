import os
import numpy as np
import pandas as pd  # type: ignore

from typing import List
from pipeline.extractor import Extractor
from pipeline.windower import Windower
from pipeline.filter import Filterer


class TestFilterer:

    pipeline_dir: str = os.path.dirname(__file__)
    test_dir: str = os.path.join(pipeline_dir, "..")
    data_dir: str = os.path.join(test_dir, "data")
    data_file_name: str = "sample.p"
    data_file_path: str = os.path.join(data_dir, data_file_name)
    artifact_file_name: str = "artifact.p"
    artifact_file_path: str = os.path.join(data_dir, artifact_file_name)

    def test_fit_transform(self):
        data: pd.DataFrame = pd.read_pickle(self.data_file_path)
        e: Extractor = Extractor()
        X: np.ndarray = e.fit_transform(data)
        y: np.ndarray = data["prompt"].values

        w: Windower = Windower()
        X = w.fit_transform(X, y)
        y = w._y_hat

        f: Filterer = Filterer(filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None})
        X = f.fit_transform(X, y)

        assert X.shape[1] == 1312

        data = pd.read_pickle(self.artifact_file_path)
        X = e.fit_transform(data)
        y = data["prompt"].values

        w = Windower(trial_size=2200)
        X = w.fit_transform(X, y)
        y = w._y_hat
        X = f.fit_transform(X, y)
        y = f._y_hat

        assert X.shape[0] == 2184
        assert f._artifacts.shape[0] == 9

        # No artifact filtering should still set _y_hat
        data = pd.read_pickle(self.artifact_file_path)
        X = e.fit_transform(data)
        y = data["prompt"].values

        w = Windower(trial_size=2200)
        X = w.fit_transform(X, y)
        y = w._y_hat

        f = Filterer(
            filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None},
            artifact_threshold=None)
        X = f.fit_transform(X, y)
        assert f._y_hat is not None

    def test__detect_channel_artifacts(self):
        data: pd.DataFrame = pd.read_pickle(self.artifact_file_path)
        X: np.ndarray = data
        y: np.ndarray = data["prompt"].values

        e: Extractor = Extractor()
        X = e.fit_transform(X, y)

        w: Windower = Windower(trial_size=2200)
        X = w.fit_transform(X, y)
        assert X.shape[0] == 2193

        window_channel_sizes = w._window_channel_size
        channel_bounds: List = [
            [
                0 if i == 0 else sum(window_channel_sizes[:i]),
                sum(window_channel_sizes[:i+1])
            ]
            for i, x in enumerate(window_channel_sizes)
        ]
        channel_idxs: np.ndarray = np.arange(*channel_bounds[0])
        ch1: np.ndarray = X[:, channel_idxs]
        f: Filterer = Filterer(filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None})
        artifact_mask: np.ndarray = f._detect_channel_artifacts(ch1)
        aX: np.ndarray = X[~artifact_mask, :]
        assert aX.shape[0] == 2184
