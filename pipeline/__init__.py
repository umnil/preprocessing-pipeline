from . import inline
from . import masked
from . import mne
from .artifact_remover import ArtifactRemover
from .extractor import Extractor
from .function_transformer import FunctionTransformer as TFunctionTransformer
from .masked import Masker
from .pairer import Pairer
from .polydetrend import PolyDetrend
from .psdbinner import PSDBinner
from .transform_pipeline import TransformPipeline
from .windower import Windower

ArtifactRemover
Extractor
Masker
Pairer
PolyDetrend
PSDBinner
TFunctionTransformer
Windower

inline
masked
mne

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
