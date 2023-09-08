import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore


def masked_accuracy_score(
    y_true: np.ma.core.MaskedArray, y_pred: np.ma.core.MaskedArray, *args, **kwargs
) -> float:
    if not isinstance(y_true, np.ma.core.MaskedArray):
        return accuracy_score(y_true, y_pred, *args, **kwargs)

    y_pred = y_pred[~y_true.flatten().mask]
    y_true = y_true.data[~y_true.mask]
    return accuracy_score(y_true, y_pred, *args, **kwargs)


def masked_roc_auc_score(
    y_true: np.ma.core.MaskedArray, y_pred: np.ma.core.MaskedArray, *args, **kwargs
) -> float:
    if not isinstance(y_true, np.ma.core.MaskedArray):
        return roc_auc_score(y_true, y_pred)

    y_pred = y_pred[~y_true.flatten().mask]
    y_true = y_true.data[~y_true.mask]
    return roc_auc_score(y_true, y_pred, *args, **kwargs)
