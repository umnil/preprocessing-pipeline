from . import inline
from . import masked
from .extractor import Extractor
from .masked import Masker
from .pairer import Pairer
from .psdbinner import PSDBinner
from .transform_pipeline import TransformPipeline
from .windower import Windower

Extractor
Masker
Pairer
PSDBinner
Windower

inline
masked

preprocessor = TransformPipeline(
    [
        ("ex", inline.Extractor()),
        ("wn", inline.Windower()),
        (
            "fl",
            inline.Filterer(filter_args={"sfreq": 200, "l_freq": 1, "h_freq": None}),
        ),
        ("ft", inline.Featurizer()),
    ]
)
