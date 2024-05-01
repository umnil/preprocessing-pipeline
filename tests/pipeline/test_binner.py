import os
import numpy as np
import pandas as pd  # type: ignore

from preprocessingpipeline.inline.extractor import Extractor
from preprocessingpipeline.inline.featurizer import Featurizer
from preprocessingpipeline.inline.filter import Filterer
from preprocessingpipeline.inline.psdbinner import PSDBinner
from preprocessingpipeline.inline.windower import Windower
from typing import List


class TestBinner:
    def test_binner(self) -> None:
        pipeline_dir: str = os.path.dirname(__file__)
        test_dir: str = os.path.join(pipeline_dir, "..")
        data_dir: str = os.path.join(test_dir, "data")
        data_filename: str = "sample.p"
        data_filepath: str = os.path.join(data_dir, data_filename)
        data: pd.DataFrame = pd.read_pickle(data_filepath)
        e: Extractor = Extractor()
        X: np.ndarray = e.fit_transform(data)
        y: np.ndarray = data["prompt"].values

        w: Windower = Windower()
        X = w.fit_transform(X, y)
        y = w._y_hat

        fl: Filterer = Filterer(filter_args=dict(sfreq=200, l_freq=1, h_freq=None))
        X = fl.fit_transform(X, y)
        y = fl._y_hat

        ft: Featurizer = Featurizer()
        X = ft.fit_transform(X, y)

        bins: List = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 100)]
        pb: PSDBinner = PSDBinner(
            bins=bins, window_channel_sizes=ft._window_channel_sizes, freqs=ft._freqs
        )
        X = pb.fit_transform(X, y)
        assert X.shape[-1] == 12
