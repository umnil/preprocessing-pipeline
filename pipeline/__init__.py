from . import funcs
from . import inline
from . import masked
from . import mne
from . import utils
from .artifact_remover import ArtifactRemover
from .extractor import Extractor
from .decorrelate import Decorrelate
from .feature_union import TransformFeatureUnion
from .function_transformer import FunctionTransformer as TFunctionTransformer
from .interpolator import Interpolator
from .masked import Masker
from .optional_transformer import OptionalTransformer
from .pairer import Pairer
from .pca import NDPCA
from .polydetrend import PolyDetrend
from .psdbinner import PSDBinner
from .transform_pipeline import TransformPipeline
from .windower import Windower

ArtifactRemover
Decorrelate
Extractor
Interpolator
Masker
NDPCA
OptionalTransformer
Pairer
PolyDetrend
PSDBinner
TFunctionTransformer
TransformFeatureUnion
Windower

funcs
inline
masked
mne
utils

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
