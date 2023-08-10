import mne
import numpy as np
from pipeline.mne.labeler import Labeler
from typing import List


# Create a mock mne.Raw object for testing
def create_mock_raw() -> mne.io.RawArray:
    # 5 seconds of data with 6 channels
    sfreq: int = 1000
    t: int = 5
    n: int = t * sfreq
    c: int = 6
    raw = mne.io.RawArray(
        np.random.randn(c, n), mne.create_info(ch_names=c, sfreq=sfreq)
    )
    raw.annotations.append(1.0, 1.0, "label_1")
    raw.annotations.append(2.0, 1.0, "label_2")
    raw.annotations.append(3.0, 1.0, "label_3")
    raw.annotations.append(4.0, 1.0, "label_1")
    return raw


# Test the Labeler class
def test_labeler_load_labels():
    # Test load_labels
    raw: mne.io.RawArray = create_mock_raw()
    labeler: Labeler = Labeler(labels=["label_1", "label_2"])
    result: List[str] = labeler.load_labels(raw)
    expected: List[str] = (
        np.array(["None", "label_1", "label_2", "label_3", "label_1"] * 1000)
        .reshape(1000, -1)
        .T.flatten()
    ).tolist()
    assert result == expected


# Test filter_labels
def test_labeler_filter_labels():
    # Test with no labels
    raw: mne.io.RawArray = create_mock_raw()
    labeler: Labeler = Labeler()
    y_labels: List[str] = labeler.load_labels(raw)
    observed: np.ndarray = labeler.filter_labels(y_labels)
    expected: np.ndarray = np.array([[0, 1, 2, 3, 1]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels no None. All other labels should be removed
    labeler = Labeler(labels=["label_1", "label_2"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)
    expected = np.array([[0, 1, 0]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels in different order no None
    labeler = Labeler(labels=["label_2", "label_1"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)
    expected = np.array([[1, 0, 1]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels in different order with None
    labeler = Labeler(labels=["None", "label_1", "label_2"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)
    expected = np.array([[0, 1, 2, 1]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()


def test_labeler_fit():
    # Test fit with a single mne.Raw object
    raw: mne.io.RawArray = create_mock_raw()
    labeler: Labeler = Labeler(labels=["label_1", "label_2"])
    labeler.fit(raw)
    assert labeler._y_hat.size == 3000
    assert labeler._y_lengths == [3000]

    # Test fit with a list of mne.Raw objects
    raw_list: List[mne.io.RawArray] = [create_mock_raw(), create_mock_raw()]
    labeler = Labeler(labels=["label_1", "label_2"])
    labeler.fit(raw_list)
    assert labeler._y_hat.size == 6000
    assert labeler._y_hat.ndim == 1
    assert labeler._y_lengths == [3000, 3000]

    # Test fit with list and don't concatenate
    raw_list: List[mne.io.RawArray] = [create_mock_raw(), create_mock_raw()]
    labeler = Labeler(labels=["label_1", "label_2"], concatenate=False)
    labeler.fit(raw_list)
    assert labeler._y_hat.size == 6000
    assert labeler._y_hat.ndim == 2
    assert labeler._y_hat.shape[0] == 2
    assert labeler._y_lengths == [3000, 3000]


def test_labeler_transform():
    # Test transform with a single mne.Raw object
    raw: mne.io.RawArray = create_mock_raw()
    labeler: Labeler = Labeler()
    x: np.ndarray = labeler.fit_transform(raw)
    assert x.shape == (5000, 6, 1)

    # Test transform with a single mne.Raw object
    raw: mne.io.RawArray = create_mock_raw()
    labeler: Labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw)
    assert x.shape == (3000, 6, 1)

    # Test transform with a list of mne.Raw objects
    raw_list: List[mne.io.RawArray] = [create_mock_raw(), create_mock_raw()]
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw_list)
    assert labeler._y_hat.shape == (6000,)
    assert labeler._mask.shape == (2, 5000)
    assert x.shape == (6000, 6, 1)

    # Test transform with no concatenation
    raw_list: List[mne.io.RawArray] = [create_mock_raw(), create_mock_raw()]
    labeler = Labeler(labels=["label_1", "label_2"], concatenate=False)
    x = labeler.fit_transform(raw_list)
    assert x.shape == (2, 3000, 6, 1)
