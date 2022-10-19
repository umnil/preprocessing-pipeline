import os
import numpy as np
import pandas as pd  # type: ignore

from typing import Tuple
from pipeline.extractor import Extractor


class TestExtractor:

    pipeline_dir: str = os.path.dirname(__file__)
    test_dir: str = os.path.join(pipeline_dir, "..")
    data_file_name: str = "sample.p"
    data_file_path: str = os.path.join(test_dir, "data", data_file_name)
    data: pd.DataFrame = pd.read_pickle(data_file_path)

    def test__resolve_data_column_name(self):
        extractor: Extractor = Extractor()
        observed: str = extractor._resolve_data_column_name(self.data)
        expected: str = "data"
        assert observed == expected

    def test__resolve_channel_info(self):
        extractor: Extractor = Extractor()
        observed: np.ndarray = extractor._resolve_channel_info(self.data)
        expected: np.ndarray = np.array([80, 2, 80, 2])
        assert (observed == expected).all()

    def test_transform(self):
        extractor: Extractor = Extractor()
        transformed_data: np.ndarray = extractor.fit_transform(self.data)
        observed: Tuple = transformed_data.shape
        expected: Tuple = (1045, 164)
        assert observed == expected

        transformed_data = extractor.fit_transform([self.data, self.data])
        observed = transformed_data.shape
        expected = (2090, 164)
        assert observed == expected
