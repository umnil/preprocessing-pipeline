import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from typing import Tuple


def masked_accuracy_score(
    y_true: np.ma.core.MaskedArray, y_pred: np.ma.core.MaskedArray, *args, **kwargs
) -> float:
    if not isinstance(y_true, np.ma.core.MaskedArray):
        return accuracy_score(y_true, y_pred, *args, **kwargs)

    y_pred = y_pred[~y_true.flatten().mask]
    y_true = y_true.data[~y_true.mask]
    return accuracy_score(y_true, y_pred, *args, **kwargs)


def masked_f1_score(
    y_true: np.ma.core.MaskedArray, y_pred: np.ma.core.MaskedArray, *args, **kwargs
) -> float:
    if not isinstance(y_true, np.ma.core.MaskedArray):
        return f1_score(y_true, y_pred, *args, **kwargs)

    y_pred = y_pred[~y_true.flatten().mask]
    y_true = y_true.data[~y_true.mask]
    return f1_score(y_true, y_pred, *args, **kwargs)


def masked_precision_recall_curve(
    y_true: np.ma.core.MaskedArray, y_pred: np.ma.core.MaskedArray, *args, **kwargs
) -> Tuple[float, float, float]:
    if not isinstance(y_true, np.ma.core.MaskedArray):
        return precision_recall_curve(y_true, y_pred, *args, **kwargs)

    y_pred = y_pred[~y_true.flatten().mask]
    y_true = y_true.data[~y_true.mask]
    return precision_recall_curve(y_true, y_pred, *args, **kwargs)


def masked_roc_auc_score(
    y_true: np.ma.core.MaskedArray, y_pred: np.ma.core.MaskedArray, *args, **kwargs
) -> float:
    if not isinstance(y_true, np.ma.core.MaskedArray):
        return roc_auc_score(y_true, y_pred)

    y_pred = y_pred[~y_true.flatten().mask]
    y_true = y_true.data[~y_true.mask]
    return roc_auc_score(y_true, y_pred, *args, **kwargs)
