import os
import pandas as pd  # type: ignore

from pipeline import preprocessor


class TestPipeline:

    pipeline_dir: str = os.path.dirname(__file__)
    test_dir: str = os.path.join(pipeline_dir, "..")
    data_dir: str = os.path.join(test_dir, "data")
    data_file_name: str = "sample.p"
    data_file_path: str = os.path.join(data_dir, data_file_name)

    def test_fit_transform(self):
        data = pd.read_pickle(self.data_file_path)
        assert data.shape[0] == data["prompt"].values.shape[0]
        preprocessor.fit_transform(data, data["prompt"].values)
