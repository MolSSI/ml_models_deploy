import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, features: list = None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def clean_method(self, X):
        """
        Attempt to get the base method from the method string.
        This mostly involves removing the performance-unaffecting
        empirical dispersion string.
        """

        X["method"] = (
            X["method"]
            .str.replace(r"\-d3\$", "")
            .str.replace(r"\-d3bj$", "")
            .str.replace(r"\-d3\(bj\)$", "")
            .str.replace(r"\-d3m$", "")
            .str.replace(r"\-d3m\(bj\)$", "")
        )

        return X

    def transform(self, X):
        X = X.copy()

        # restricted can present in X or can be calculated from nbeta and nalpha
        if 'restricted' not in X and 'restricted' in self.features \
                and 'nalpha' in X and 'nbeta' in X:
            X["restricted"] = X['nalpha'] == X['nbeta']

        # calc nelec if missing from input
        if 'nelec' not in X and 'nelec' in self.features:
            X['nelec'] = X['nalpha'] + X['nbeta']

        if 'o3' in self.features:
            X["o3"] = X['nelec']/2 ** 3

        if 'nmo3' in self.features:
            X["nmo3"] = X["nmo"] ** 3

        if 'o4' in self.features:
            X["o4"] = X['nelec']/2 ** 4

        if 'nmo4' in self.features:
            X["nmo4"] = X["nmo"] ** 4

        if 'kscale' in self.features:
            X["kscale"] = (
                X["nmo"] * X["nmo"] * (X["nmo"] * 3) * X['nelec']/2
            )

        if 'jscale' in self.features:
            X["jscale"] = X["nmo"] * X["nmo"] * (X["nmo"] * 3)

        if 'diag' in self.features:
            X["diag"] = X["nmo"] * X["nmo"] * X["nmo"]

        if 'nvirt' in self.features:
            X["nvirt"] = X["nmo"] - X['nelec']/2

        X = self.clean_method(X)

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


class DebugStep(BaseEstimator, TransformerMixin):
    """A step to be added to a pipeline to log the current X values
    after transformation"""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(X.shape)
        logger.info(X.columns)
        logger.info(X.head(3))

        return X

# ---------------------------------------------
