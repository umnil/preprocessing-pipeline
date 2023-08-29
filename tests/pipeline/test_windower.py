import numpy as np
from pipeline.windower import Windower
from typing import List, Tuple, cast


# Create mock data for testing
def create_mock_data(
    uniform=True, different_shapes=False
) -> Tuple[np.ndarray, np.ndarray]:
    # Create mock data with the desired shape for windowing
    # shape (3, 10, 5000)
    sfreq: int = 1000
    n_time: int = 5
    n_timepoints: int = sfreq * n_time
    n_channels: int = 10
    n: int = 3
    data: np.ndarray = np.random.randn(n, n_channels, n_timepoints)
    label_order: List = [0, 1, 2, 3, 1]
    label_times: List = [1000] * len(label_order)
    labels = make_labels(label_order, label_times)
    labels = np.vstack([labels] * n)
    if not uniform:
        label_times = [1000, 500, 750, 1500, 1250]
        labels = np.hstack([[label] * t for label, t in zip(label_order, label_times)])
        labels = np.vstack([labels] * n)

    if different_shapes:
        label_order = [[0, 1, 2, 3, 1], [1, 2, 4], [3, 2, 1, 0]]
        label_times = [
            [1000, 500, 750, 1500, 1250],
            [1750, 1625, 1625],
            [500, 2000, 750, 1750],
        ]
        labels = np.vstack(
            [
                make_labels(orders, times)
                for orders, times in zip(label_order, label_times)
            ]
        )

    return data, labels


# Create mod labels
def make_labels(order: List, times: List) -> np.ndarray:
    return np.hstack([[label] * time for label, time in zip(order, times)])


# Test the Windower class
class TestWindower:
    # Test the _window_transform method
    def test__window_transform(self) -> None:
        # Unconcatenated
        data, labels = create_mock_data(False)
        assert data.shape == (3, 10, 5000)
        assert labels.shape == (3, 5000)
        windower = Windower(samples_per_window=500, window_step=250)
        x = cast(np.ndarray, windower._window_transform(data))
        assert x.shape == (3, 10, 19, 500)

    # Test _window_by_label
    def test__window_by_label(self) -> None:
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data()
        windower: Windower = Windower(
            samples_per_window=500, window_step=250, label_scheme=4
        )
        y: np.ndarray = cast(np.ndarray, windower._window_by_label(labels[0]))
        assert y.shape == (5,)

    # Test _window_by_label
    def test__window_by_label_non_uniform(self) -> None:
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data(uniform=False)
        windower: Windower = Windower(
            samples_per_window=500, window_step=250, label_scheme=4
        )
        y: np.ndarray = cast(np.ndarray, windower._window_by_label(labels[0]))
        assert y.shape == (5,)

    def test__make_labels(self) -> None:
        # Unconcatenated 0
        data, labels = create_mock_data()
        windower = Windower(samples_per_window=500, window_step=250)
        y = windower._make_labels(labels)
        assert y.shape == (3, 19)

        # Unconcatenated 3
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        y = windower._make_labels(labels)
        assert y.shape == (3, 15)

        # Unconcatenated 4
        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        windower._y = labels
        y = windower._make_labels(labels)
        assert y.shape == (3, 5)

    def test_transform(self) -> None:
        data, labels = create_mock_data()
        windower = Windower(samples_per_window=500, window_step=250)
        x = windower.fit_transform(data[0:1], labels[0:1])
        assert x.shape == (1, 10, 19, 500)

        # Unconcatenated 0
        data, labels = create_mock_data()
        windower = Windower(samples_per_window=500, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 19, 500)

        # Unconcatenated 0 Windowed precise
        data, labels = create_mock_data()
        windower = Windower(samples_per_window=1000, window_step=1000)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 5, 1000)

        # Unconcatenated 3
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 15, 500)

        # Unconcatenated 4
        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 5, 1000)

    def test_transform_non_uniform(self) -> None:
        # Unconcatenated 0
        data, labels = create_mock_data(uniform=False)
        windower = Windower(samples_per_window=500, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 19, 500)

        # Unconcatenated 3
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 15, 500)

        # Unconcatenated 4
        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 5, 1500)

        data, labels = create_mock_data(uniform=False, different_shapes=True)
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 16, 500)

        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 10, 5, 2000)
        assert windower._y_lengths == [5, 3, 4]

    def test_novel_transform(self) -> None:
        data, labels = create_mock_data(uniform=False)
        windower = Windower(samples_per_window=500, window_step=250)
        x = windower.transform(data, labels)
        assert x.shape == (3, 10, 19, 500)
