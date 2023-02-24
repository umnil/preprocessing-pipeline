import os
import numpy as np
import pandas as pd  # type: ignore

from typing import Dict
from scipy.fft import fft, fftfreq  # type: ignore
from pipeline.inline.extractor import Extractor
from pipeline.inline.windower import Windower
from pipeline.inline.filter import Filterer
from pipeline.inline.featurizer import Featurizer


def fftfilt(data, *args, **kwargs):
    N = data.shape[1]
    T = 1.0 / 200.0

    yf = fft(data)
    xf = fftfreq(N, T)[: N // 2]

    yf = 2.0 / N * np.abs(yf[:, 0 : N // 2])

    return yf, xf


class TestFeaturizer:

    pipeline_dir: str = os.path.dirname(__file__)
    test_dir: str = os.path.join(pipeline_dir, "..")
    data_dir: str = os.path.join(test_dir, "data")
    data_file_name: str = "sample.p"
    data_file_path: str = os.path.join(data_dir, data_file_name)

    def test_fit_transform(self):
        filter_args: Dict = {"sfreq": 200, "l_freq": 1, "h_freq": None}
        data: pd.DataFrame = pd.read_pickle(self.data_file_path)
        e: Extractor = Extractor()
        X: np.ndarray = e.fit_transform(data)
        y: np.ndarray = data["prompt"].values

        w: Windower = Windower()
        X = w.fit_transform(X, y)
        y = w._y_hat

        f: Filterer = Filterer(filter_args=filter_args)
        X = f.fit_transform(X, y)

        ft: Featurizer = Featurizer()
        X = ft.fit_transform(X)

        assert X.shape[1] == 644

        # Different window size
        sample_data: pd.DataFrame = data.query("state == 2")
        X = e.fit_transform(sample_data)
        y = sample_data["prompt"].values
        w = Windower(window_size=3, trial_size=150)
        X = w.fit_transform(X, y)
        y = w._y_hat
        f = Filterer(
            window_channel_sizes=w._window_channel_size, filter_args=filter_args
        )
        X = f.fit_transform(X, y)
        ft = Featurizer(window_channel_sizes=w._window_channel_size)
        X = ft.fit_transform(X)

        assert X.shape[0] == 296

        # Different window size and function
        sample_data = data.query("state == 2")
        X = e.fit_transform(sample_data)
        y = sample_data["prompt"].values
        w = Windower(window_size=3, trial_size=150)
        X = w.fit_transform(X, y)
        y = w._y_hat
        f = Filterer(
            window_channel_sizes=w._window_channel_size, filter_args=filter_args
        )
        X = f.fit_transform(X, y)
        ft = Featurizer(
            window_channel_sizes=w._window_channel_size, spectral_func=fftfilt
        )
        X = ft.fit_transform(X)

        assert X.shape[0] == 296
