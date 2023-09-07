import scipy  # type: ignore
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import Dict, List, Optional


class Decorrelate(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 1.0):
        """A transformer that removes features from the input data that are
        correlated beyond the `threshold`

        Parameters
        ----------
        threshold : float
            The limit of correlation distance to allow
        """
        self.threshold: float = threshold

    def fit(self, x: np.ndarray, y: np.ndarray, **fitparams) -> "Decorrelate":
        # Ensure 2D
        x = x.reshape(-1, x.shape[-1])

        # Compute Spearman Correlation between features
        corr: np.ndarray = scipy.stats.spearmanr(x)[0]

        # Make symmetric
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)

        # Compute correlative distance
        self.dist: np.ndarray = 1 - np.abs(corr)

        # convert to condenced distance vector
        dist_vec: np.ndarray = scipy.spatial.distance.squareform(self.dist)

        # Compute Ward Linkage
        dist_link: np.ndarray = scipy.cluster.hierarchy.ward(dist_vec)

        # Compute clustered features
        self.clusters: np.ndarray = scipy.cluster.hierarchy.fcluster(
            dist_link, self.threshold, criterion="distance"
        )

        cluster_to_feat_idx: Dict = {id: i for i, id in enumerate(self.clusters)}
        self.feat_idxs: List = [
            feat_idx for id, feat_idx in cluster_to_feat_idx.items()
        ]
        return self

    def transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return x[..., self.feat_idxs]
