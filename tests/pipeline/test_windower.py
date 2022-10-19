import os
import pickle
import numpy as np
import pandas as pd  # type: ignore

from typing import List, Dict, Tuple
from pipeline.extractor import Extractor
from pipeline.windower import Windower


class TestWindower:

    pipeline_dir: str = os.path.dirname(__file__)
    test_dir: str = os.path.join(pipeline_dir, "..")
    data_dir: str = os.path.join(test_dir, "data")
    data_file_name: str = "sample.p"
    data_file_path: str = os.path.join(data_dir, data_file_name)

    example_data_202201_filename: str = "example_data_20220101162641.p"
    example_data_202201_filepath: str = os.path.join(
        data_dir, example_data_202201_filename
    )

    def test__make_labels(self):
        data: pd.DataFrame = pd.read_pickle(self.data_file_path)
        data = data.query("state == 2")
        assert data.shape[0] == 300

        e: Extractor = Extractor()
        X: np.ndarray = e.fit_transform(data)
        y: np.ndarray = data["prompt"].values
        assert X.shape[0] == y.shape[0]

        w: Windower = Windower(trial_size=300)
        labels: np.ndarray = w._make_labels(y)
        assert labels.shape[0] == 300 - 7
        assert (labels == y[:-7]).all()

        w = Windower(window_size=5, label_scheme=1, trial_size=300)
        labels = w._make_labels(y)
        assert labels.shape[0] == 300 - 4
        assert (labels == y[4:]).all()

        w = Windower(window_size=15, window_step=15, trial_size=300)
        labels = w._make_labels(y)
        assert labels.shape[0] == 300 // 15

        w = Windower(window_size=15, window_step=15, trial_size=150)
        labels = w._make_labels(y)
        assert labels.shape[0] == 300 // 15

        # Multi Trial
        trials: List = [data, data]
        X = e.fit_transform(trials)
        y = pd.concat(trials)["prompt"].values
        assert X.shape[0] == y.shape[0]

        w = Windower(window_size=8, trial_size=300)
        labels = w._make_labels(y)
        assert labels.shape[0] == 600 - (7*2)

    def test__get_channel_packets(self):
        ch1: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
        ch2: List[int] = [0, 1, 2, 3]
        p1: List[int] = ch1[:4] + ch2[:2]
        p2: List[int] = ch1[4:] + ch2[2:]
        pch1: List[List] = [ch1[:4], ch1[4:]]
        pch2: List[List] = [ch2[:2], ch2[2:]]
        X: np.ndarray = np.array([p1, p2])

        w: Windower = Windower(
            window_size=2,
            packet_channel_sizes=[4, 2]
        )
        _ch1: np.ndarray = w._get_channel_packets(X, 0)
        assert _ch1.tolist() == pch1
        _ch2: np.ndarray = w._get_channel_packets(X, 1)
        assert _ch2.tolist() == pch2

    def test__transform_channel(self):
        ch1: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
        ch2: List[int] = [0, 1, 2, 3]
        p1: List[int] = ch1[:4] + ch2[:2]
        p2: List[int] = ch1[4:] + ch2[2:]
        X: np.ndarray = np.array([p1, p2])
        y: np.ndarray = np.array([0, 1])
        w: Windower = Windower(
            window_size=2,
            packet_channel_sizes=[4, 2],
            trial_size=2
        )
        w.fit(X, y)
        _ch1: np.ndarray = w._get_channel_packets(X, 0)
        ch1t: np.ndarray = w._transform_channel(_ch1)
        assert ch1t.tolist() == [ch1]
        _ch2: np.ndarray = w._get_channel_packets(X, 1)
        ch2t: np.ndarray = w._transform_channel(_ch2)
        assert ch2t.tolist() == [ch2]

        # Example Data
        wn: Windower = Windower(trial_size=150)
        example_data_202201: Dict
        with open(self.example_data_202201_filepath, "rb") as fh:
            example_data_202201 = pickle.load(fh)
        X, y = example_data_202201.values()
        wn.fit(X, y)
        X_ch1: np.ndarray = wn._get_channel_packets(X, 0)
        X_ch2: np.ndarray = wn._get_channel_packets(X, 1)

        X_ch1_hat: np.ndarray = wn._transform_channel(X_ch1)
        expected_shape: Tuple = (1716, 640)
        assert expected_shape == X_ch1_hat.shape
        X_ch2_hat: np.ndarray = wn._transform_channel(X_ch2)
        expected_shape = (1716, 16)
        assert expected_shape == X_ch2_hat.shape

    def test__fit_transform(self):
        ch1: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
        ch2: List[int] = [0, 1, 2, 3]
        p1: List[int] = ch1[:4] + ch2[:2]
        p2: List[int] = ch1[4:] + ch2[2:]
        X: np.ndarray = np.array([p1, p2])
        y: np.ndarray = np.array([0, 1])
        assert X.shape == (2, 6)

        w: Windower = Windower(
            window_size=2,
            packet_channel_sizes=[4, 2],
            trial_size=2
        )
        X_trans: np.ndarray = w.fit_transform(X, y)
        assert X_trans.tolist() == [
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
        ]

        # Example data
        w = Windower(trial_size=150)
        example_data_202201: Dict
        with open(self.example_data_202201_filepath, "rb") as fh:
            example_data_202201 = pickle.load(fh)
        X, y = example_data_202201.values()
        assert X.shape == (1800, 164)
        X_hat: np.ndarray = w.fit_transform(X, y)
        assert X_hat.shape == (1716, 1312)

    def test_window_packets(self):
        """This test is required to ensure that if the window_size ever
        changes, that the window_packets array will as well"""

        ch1: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
        ch2: List[int] = [0, 1, 2, 3]
        p1: List[int] = ch1[:4] + ch2[:2]
        p2: List[int] = ch1[4:] + ch2[2:]
        X: np.ndarray = np.array([p1, p2])
        y: np.ndarray = np.array([0, 1])
        assert X.shape == (2, 6)

        w: Windower = Windower(
            window_size=2,
            packet_channel_sizes=[4, 2],
            trial_size=2
        )
        X_hat: np.ndarray = w.fit(X, y)
        assert w.window_packets.shape == (1, 2)
        assert w._trial_window_lengths == [1]

        # Example data
        example_data_202201: Dict
        with open(self.example_data_202201_filepath, "rb") as fh:
            example_data_202201 = pickle.load(fh)
        X, y = example_data_202201.values()
        assert X.shape == (1800, 164)

        w.window_size = 8
        w.packet_channel_sizes = [80, 2, 80, 2]
        w.trial_size = 150
        X_hat = w.fit_transform(X, y)
        assert X_hat.shape == (1716, 1312)
        assert w._trial_window_lengths == ([150 - 7] * 12)
