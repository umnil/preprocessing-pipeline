from .extractor import Extractor
from .featurizer import Featurizer
from .filter import Filterer
from .psdbinner import PSDBinner
from .transform_pipeline import TransformPipeline
from .windower import Windower

Extractor
Featurizer
Filterer
PSDBinner
Windower

preprocessor = TransformPipeline(
    [
        ("ex", Extractor()),
        ("wn", Windower()),
        ("fl", Filterer(filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None})),
        ("ft", Featurizer()),
    ]
)
