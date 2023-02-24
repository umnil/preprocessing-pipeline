from . import inline
from . import masked
from .extractor import Extractor
from .inline.featurizer import Featurizer
from .inline.filter import Filterer
from .masked import Masker
from .psdbinner import PSDBinner
from .transform_pipeline import TransformPipeline
from .windower import Windower

Extractor
Featurizer
Filterer
Masker
PSDBinner
Windower

inline
masked

preprocessor = TransformPipeline(
    [
        ("ex", Extractor()),
        ("wn", Windower()),
        ("fl", Filterer(filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None})),
        ("ft", Featurizer()),
    ]
)
