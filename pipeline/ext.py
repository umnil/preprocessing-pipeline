import pandas as pd  # type: ignore

from typing import List
from .inline import Extractor, Windower
from .transform_pipeline import TransformPipeline


def calc_wcs(df: pd.DataFrame) -> List:
    """
    Given a dataframe compute the anticipated
    window channel sizes as determined by the
    windower

    Parameters
    ----------
    df : pd.Dataframe
        The input dataframe

    Returns
    -------
    List
        Return a set of sizes for each channel in
        the set
    """
    temp_pipeline: TransformPipeline = TransformPipeline(
        [("ex", Extractor()), ("wn", Windower(label_scheme=4))]
    )
    temp_pipeline.fit_transform(df, df["prompt"].values)
    return temp_pipeline.steps[-1][-1]._window_channel_size
