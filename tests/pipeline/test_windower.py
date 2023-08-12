import numpy as np
from pipeline.windower import Windower
from typing import cast, Tuple


# Create mock data for testing
def create_mock_data(concatenated=True, uniform=True) -> Tuple[np.ndarray, np.ndarray]:
    # Create mock data with the desired shape for windowing
    # concatenated shape (15000, 10, 1)
    # unconcatenated shape (3, 5000, 10, 1)
    sfreq: int = 1000
    n_time: int = 5
    n_timepoints: int = sfreq * n_time
    n_channels: int = 10
    n: int = 3
    data: np.ndarray = np.random.randn(n, n_timepoints, n_channels, 1)
    labels: np.ndarray = np.array([[0, 1, 2, 3, 1]] * 1000).T.flatten() * np.array(
        [[1]] * 3
    )
    if not uniform:
        labels = np.array(
            [0] * 1000 + [1] * 500 + [2] * 750 + [3] * 1500 + [1] * 1250
        ) * np.array([[1]] * 3)
    if concatenated:
        data = data.reshape(-1, n_channels, 1)
        labels = labels.flatten()
    return data, labels


# Test the Windower class
class TestWindower:
    # Test the _window_transform method
    def test__window_transform(self) -> None:
        # Concatenated
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data()
        assert data.shape == (15000, 10, 1)
        assert labels.shape == (15000,)
        windower: Windower = Windower(samples_per_window=500, window_step=250)
        x: np.ndarray = cast(np.ndarray, windower._window_transform(data, axis=0))
        assert x.shape == (59, 10, 1, 500)

        # Unconcatenated
        data, labels = create_mock_data(False)
        assert data.shape == (3, 5000, 10, 1)
        assert labels.shape == (3, 5000)
        windower = Windower(samples_per_window=500, window_step=250, axis=[1, -1])
        x = cast(np.ndarray, windower._window_transform(data, axis=1))
        assert x.shape == (3, 19, 10, 1, 500)

    # Test _window_by_label
    def test__window_by_label(self) -> None:
        # Concatenated
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data()
        windower: Windower = Windower(
            samples_per_window=500, window_step=250, label_scheme=4
        )
        y: np.ndarray = cast(np.ndarray, windower._window_by_label(labels))
        assert y.shape == (15,)

    # Test _window_by_label
    def test__window_by_label_non_uniform(self) -> None:
        # Concatenated
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data(uniform=False)
        windower: Windower = Windower(
            samples_per_window=500, window_step=250, label_scheme=4
        )
        y: np.ndarray = cast(np.ndarray, windower._window_by_label(labels))
        assert y.shape == (15,)

    def test__make_labels(self) -> None:
        # Concatenated 0
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data()
        windower: Windower = Windower(samples_per_window=500, window_step=250)
        y: np.ndarray = windower._make_labels(labels)
        assert y.shape == (59,)

        # Concatenated 3
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        y = windower._make_labels(labels)
        assert y.shape == (45,)

        # Concatenated 4
        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        windower._y = labels
        y = windower._make_labels(labels)
        assert y.shape == (15,)

        # Unconcatenated 0
        data, labels = create_mock_data(False)
        windower = Windower(samples_per_window=500, window_step=250, axis=[1, -1])
        y = windower._make_labels(labels)
        assert y.shape == (3, 19)

        # Unconcatenated 3
        windower = Windower(
            samples_per_window=500, label_scheme=3, window_step=250, axis=[1, -1]
        )
        y = windower._make_labels(labels)
        assert y.shape == (3, 15)

        # Unconcatenated 4
        windower = Windower(
            samples_per_window=500, label_scheme=4, window_step=250, axis=[1, -1]
        )
        windower._y = labels
        y = windower._make_labels(labels)
        assert y.shape == (3, 5)

    def test_transform(self) -> None:
        # Concatenated 0
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data()
        windower: Windower = Windower(samples_per_window=500, window_step=250)
        x: np.ndarray = windower.fit_transform(data, labels)
        assert x.shape == (59, 10, 500)

        # Concatenated 3
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (45, 10, 500)

        # Concatenated 4
        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (15, 10, 1000)

        # Unconcatenated 0
        data, labels = create_mock_data(False)
        windower = Windower(samples_per_window=500, window_step=250, axis=[1, -1])
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 19, 10, 500)

        # Unconcatenated 3
        windower = Windower(
            samples_per_window=500, label_scheme=3, window_step=250, axis=[1, -1]
        )
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 15, 10, 500)

        # Unconcatenated 4
        windower = Windower(
            samples_per_window=500, label_scheme=4, window_step=250, axis=[1, -1]
        )
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 5, 10, 1000)

    def test_transform_non_uniform(self) -> None:
        # Concatenated 0
        data: np.ndarray
        labels: np.ndarray
        data, labels = create_mock_data(uniform=False)
        windower: Windower = Windower(samples_per_window=500, window_step=250)
        x: np.ndarray = windower.fit_transform(data, labels)
        assert x.shape == (59, 10, 500)

        # Concatenated 3
        windower = Windower(samples_per_window=500, label_scheme=3, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (45, 10, 500)

        # Concatenated 4
        windower = Windower(samples_per_window=500, label_scheme=4, window_step=250)
        x = windower.fit_transform(data, labels)
        assert x.shape == (15, 10, 1500)

        # Unconcatenated 0
        data, labels = create_mock_data(concatenated=False, uniform=False)
        windower = Windower(samples_per_window=500, window_step=250, axis=[1, -1])
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 19, 10, 500)

        # Unconcatenated 3
        windower = Windower(
            samples_per_window=500, label_scheme=3, window_step=250, axis=[1, -1]
        )
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 15, 10, 500)

        # Unconcatenated 4
        windower = Windower(
            samples_per_window=500, label_scheme=4, window_step=250, axis=[1, -1]
        )
        x = windower.fit_transform(data, labels)
        assert x.shape == (3, 5, 10, 1500)
