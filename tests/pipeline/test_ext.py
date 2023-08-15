import os
import pandas as pd  # type: ignore

from pipeline.ext import calc_wcs


class TestExt:
    def test_calc_wcs(self) -> None:
        pipeline_dir: str = os.path.dirname(__file__)
        test_dir: str = os.path.join(pipeline_dir, "..")
        data_dir: str = os.path.join(test_dir, "data")
        data_filename: str = "sample.p"
        data_filepath: str = os.path.join(data_dir, data_filename)

        df: pd.DataFrame = pd.read_pickle(data_filepath)
        result = calc_wcs(df)
        assert result == [24880, 622] * 2
