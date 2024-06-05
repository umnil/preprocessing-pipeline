import numpy as np
from preprocessingpipeline.utils import (
    equalize_list_to_array,
    equalize_shape,
    generate_padding_param,
    mask_list,
    unmask_array,
)
from typing import List


# Test the generate_padding_param function
def test_generate_padding_param() -> None:
    a: List = [np.array([1, 2, 3]), np.array([4, 5])]
    max_lens: List[int] = [4]
    padding_params: List = generate_padding_param(a[0], max_lens)
    assert padding_params == [[0, 1]]

    a = [
        np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),  # Shape (2, 4)
        np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),  # shape (3, 3)
    ]
    max_lens = [3, 4]
    padding_params = generate_padding_param(a[0], max_lens, axes=[0, 1])
    assert padding_params == [[0, 1], [0, 0]]


# Test the equalize_list_to_array function
def test_equalize_list_to_array() -> None:
    # Test case with a list of 1D arrays
    input_list: List = [np.array([1, 2, 3]), np.array([4, 5])]
    result: np.ndarray = equalize_list_to_array(input_list)
    expected: np.ndarray = np.array([[1, 2, 3], [4, 5, np.nan]])
    np.testing.assert_allclose(result, expected)


# Test the equalize_shape function
def test_equalize_shape() -> None:
    # Test case with a 2D array
    input_array: np.ndarray = np.array([[1, 2, np.nan, np.nan], [3, 4, 5, np.nan]])
    result: np.ndarray = equalize_shape(input_array)
    expected: np.ndarray = np.array([[1, 2, np.nan], [3, 4, 5]])
    assert expected.shape == result.shape

    # Test case with a 3D array  (2, 4, 2)
    input_array_3d: np.ndarray = np.array(
        [
            [[1, 2], [3, 4], [np.nan, np.nan], [np.nan, np.nan]],
            [[5, 6], [7, 8], [9, 10], [np.nan, np.nan]],
        ]
    )
    result_3d: np.ndarray = equalize_shape(input_array_3d, axis=1)
    expected_3d: np.ndarray = np.array(
        [[[1, 2], [3, 4], [np.nan, np.nan]], [[5, 6], [7, 8], [9, 10]]]
    )
    assert expected_3d.shape == result_3d.shape


def test_unmask_array() -> None:
    input_array: np.ndarray = np.array(  # Shape (4, 10, 6)
        [
            [
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
                [9, 1, 8, np.nan, np.nan, np.nan],
            ],
            [
                [9, 1, 8, 2, 7, 4],
                [9, 1, 8, 2, 7, 4],
                [9, 1, 8, 2, 7, 4],
                [9, 1, 8, 2, 7, 4],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [9, 1, 8, 2, np.nan, np.nan],
                [9, 1, 8, 2, np.nan, np.nan],
                [9, 1, 8, 2, np.nan, np.nan],
                [9, 1, 8, 2, np.nan, np.nan],
                [9, 1, 8, 2, np.nan, np.nan],
                [9, 1, 8, 2, np.nan, np.nan],
                [9, 1, 8, 2, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ]
    )
    expected_list: List = [
        [
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
            [9, 1, 8],
        ],
        [
            [9, 1, 8, 2, 7, 4],
            [9, 1, 8, 2, 7, 4],
            [9, 1, 8, 2, 7, 4],
            [9, 1, 8, 2, 7, 4],
        ],
        [
            [9, 1, 8, 2],
            [9, 1, 8, 2],
            [9, 1, 8, 2],
            [9, 1, 8, 2],
            [9, 1, 8, 2],
            [9, 1, 8, 2],
            [9, 1, 8, 2],
        ],
    ]
    observed_list: List = unmask_array(input_array)
    assert observed_list == expected_list

    observed_array: np.ndarray = mask_list(expected_list)
    assert np.allclose(
        np.ma.masked_invalid(observed_array), np.ma.masked_invalid(input_array)
    )
