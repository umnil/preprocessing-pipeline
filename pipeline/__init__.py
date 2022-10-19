from .extractor import Extractor
from .windower import Windower
from .filter import Filterer
from .featurizer import Featurizer
from .transform_pipeline import TransformPipeline

Extractor
Featurizer
Filterer
Windower

preprocessor = TransformPipeline([
    ("ex", Extractor()),
    ("wn", Windower()),
    ("fl", Filterer(filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None})),
    ("ft", Featurizer())
])
