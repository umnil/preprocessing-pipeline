import mne  # type: ignore
import numpy as np
from mne.io import RawArray  # type: ignore
from mne import create_info, set_log_level  # type: ignore
from preprocessingpipeline.mne.labeler import Labeler
from typing import List


# Create a mock mne.Raw object for testing
def create_mock_raw(uniform=True, version=0) -> RawArray:
    set_log_level("critical")  # supress logs

    # 5 seconds of data with 6 channels
    sfreq: int = 1000
    t: int = 5
    n: int = t * sfreq
    c: int = 6
    raw = mne.io.RawArray(np.random.randn(c, n), create_info(ch_names=c, sfreq=sfreq))
    if uniform:
        raw.annotations.append(1.0, 1.0, "label_1")
        raw.annotations.append(2.0, 1.0, "label_2")
        raw.annotations.append(3.0, 1.0, "label_3")
        raw.annotations.append(4.0, 1.0, "label_1")
    else:
        if version == 0:
            raw.annotations.append(1.0, 0.5, "label_1")
            raw.annotations.append(1.5, 0.75, "label_2")
            raw.annotations.append(2.25, 1.5, "label_3")
            raw.annotations.append(3.75, 1.25, "label_1")
        elif version == 1:
            # 5 Seconds
            raw.annotations.append(1.0, 0.75, "label_1")
            raw.annotations.append(1.75, 1.5, "label_2")
            raw.annotations.append(3.25, 0.5, "label_3")
            raw.annotations.append(3.75, 1.25, "label_1")
        elif version == 2:
            # 6 seconds
            t = 6
            n = t * sfreq
            raw = mne.io.RawArray(
                np.random.randn(c, n), create_info(ch_names=c, sfreq=sfreq)
            )
            raw.annotations.append(1.0, 0.50, "label_1")
            raw.annotations.append(1.5, 1.75, "label_2")
            raw.annotations.append(3.25, 1.5, "label_3")
            raw.annotations.append(4.75, 1.25, "label_1")
    return raw


# Test the Labeler class
def test_labeler_load_labels() -> None:
    # Test load_labels
    raw: RawArray = create_mock_raw()
    labeler: Labeler = Labeler(labels=["label_1", "label_2"])
    result: List[str] = labeler.load_labels(raw)
    expected: List[str] = (
        np.array(["None", "label_1", "label_2", "label_3", "label_1"] * 1000)
        .reshape(1000, -1)
        .T.flatten()
    ).tolist()
    assert result == expected

    # Test load_labels, non uniform
    raw = create_mock_raw(False)
    labeler = Labeler(labels=["label_1", "label_2"])
    result = labeler.load_labels(raw)
    expected = (
        np.array(
            ["None"] * 1000
            + ["label_1"] * 500
            + ["label_2"] * 750
            + ["label_3"] * 1500
            + ["label_1"] * 1250
        )
        .flatten()
        .tolist()
    )
    assert result == expected


# Test filter_labels
def test_labeler_filter_labels() -> None:
    # Test with no labels
    raw: RawArray = create_mock_raw()
    labeler: Labeler = Labeler()
    y_labels: List[str] = labeler.load_labels(raw)
    observed: np.ndarray = labeler.filter_labels(y_labels)[0]
    expected: np.ndarray = np.array([[0, 1, 2, 3, 1]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels no None. All other labels should be removed
    labeler = Labeler(labels=["label_1", "label_2"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)[0]
    expected = np.array([[0, 1, 0]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels in different order no None
    labeler = Labeler(labels=["label_2", "label_1"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)[0]
    expected = np.array([[1, 0, 1]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels in different order with None
    labeler = Labeler(labels=["None", "label_1", "label_2"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)[0]
    expected = np.array([[0, 1, 2, 1]] * 1000).T.flatten()
    assert observed.shape == expected.shape
    assert (observed == expected).all()


def test_labeler_filter_labels_nonuniform() -> None:
    # Test with no labels
    raw: RawArray = create_mock_raw(False)
    labeler: Labeler = Labeler()
    y_labels: List[str] = labeler.load_labels(raw)
    observed: np.ndarray = labeler.filter_labels(y_labels)[0]
    expected: np.ndarray = np.array(
        [0] * 1000 + [1] * 500 + [2] * 750 + [3] * 1500 + [1] * 1250
    )
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels no None. All other labels should be removed
    labeler = Labeler(labels=["label_1", "label_2"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)[0]
    expected = np.array([0] * 500 + [1] * 750 + [0] * 1250)
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels in different order no None
    labeler = Labeler(labels=["label_2", "label_1"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)[0]
    expected = np.array([1] * 500 + [0] * 750 + [1] * 1250)
    assert observed.shape == expected.shape
    assert (observed == expected).all()

    # Test with specified labels in different order with None
    labeler = Labeler(labels=["None", "label_1", "label_2"])
    y_labels = labeler.load_labels(raw)
    observed = labeler.filter_labels(y_labels)[0]
    expected = np.array([0] * 1000 + [1] * 500 + [2] * 750 + [1] * 1250)
    assert observed.shape == expected.shape
    assert (observed == expected).all()


def test_labeler_transform() -> None:
    # Test transform with a single mne.Raw object
    raw: RawArray = create_mock_raw()
    labeler: Labeler = Labeler()
    x: np.ndarray = labeler.fit_transform(raw)
    assert labeler._y_hat.shape == (1, 5000)
    assert labeler._y_lengths == [5000]
    assert x.shape == (1, 6, 5000)

    # Test transform with a single mne.Raw object
    raw = create_mock_raw()
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw)
    assert x.shape == (1, 6, 3000)

    # Test transform with a list of mne.Raw objects
    raw_list = [create_mock_raw(), create_mock_raw()]
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw_list)
    assert labeler._y_hat.shape == (2, 3000)
    assert labeler._y_lengths == [3000, 3000]
    assert x.shape == (2, 6, 3000)

    # Non-Uniform
    # Test transform with a single mne.Raw object
    raw = create_mock_raw(False)
    labeler = Labeler()
    x = labeler.fit_transform(raw)
    assert x.shape == (1, 6, 5000)

    # Test transform with a single mne.Raw object
    raw = create_mock_raw(False)
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw)
    assert x.shape == (1, 6, 2500)

    # Test transform with a list of mne.Raw objects
    raw_list = [create_mock_raw(False), create_mock_raw(False)]
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw_list)
    assert labeler._y_hat.shape == (2, 2500)
    assert x.shape == (2, 6, 2500)

    # Test transform with a list of mne.Raw objects with different event timing
    raw_list = [create_mock_raw(False), create_mock_raw(False, 1)]
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.fit_transform(raw_list)
    assert labeler._y_hat.shape == (2, 3500)
    assert x.shape == (2, 6, 3500)

    # Test transform with different event timing and no concatenation
    # Test transform with a list of mne.Raw objects with different event timing
    # and lengths
    raw_list = [create_mock_raw(False), create_mock_raw(False, 2)]
    labeler = Labeler(labels=["label_1", "label_2"])
    x = labeler.transform(raw_list)
    y = labeler._y_hat
    assert y.shape == (2, 3500)
    assert x.shape == (2, 6, 3500)
