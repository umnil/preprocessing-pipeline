import numpy as np
from pipeline.windower import Windower
from typing import cast, Tuple


# Create mock data for testing
def create_mock_data(uniform=True) -> Tuple[np.ndarray, np.ndarray]:
    # Create mock data with the desired shape for windowing
    # shape (3, 10, 5000)
    sfreq: int = 1000
    n_time: int = 5
    n_timepoints: int = sfreq * n_time
    n_channels: int = 10
    n: int = 3
    data: np.ndarray = np.random.randn(n, n_channels, n_timepoints)
    labels: np.ndarray = np.array([[0, 1, 2, 3, 1]] * 1000).T.flatten() * np.array(
        [[1]] * 3
    )
    if not uniform:
        labels = np.array(
            [0] * 1000 + [1] * 500 + [2] * 750 + [3] * 1500 + [1] * 1250
        ) * np.array([[1]] * 3)
    return data, labels


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
