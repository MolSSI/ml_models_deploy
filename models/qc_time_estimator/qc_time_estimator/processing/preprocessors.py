import numpy as np
import pandas as pd
from enum import Enum
from collections import defaultdict
from qcelemental.info import cpu_info
from qcelemental.info import dft_info
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin



class MethodEnum(int, Enum):
        """ method classes, ranked by cost"""

        forcefield = 0
        ml = 1
        semiempirical = 2
        dft = 3
        mp2 = 4
        ccsd = 5
        ccsdprt = 6


class DriverEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, params=None):
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DriverEncoder':
        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()

        driver_map = {"energy": 0, "gradient": 1, "hessian": 2}
        X['driver'] = X['driver'].map(driver_map)
        return X


class MethodFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, params=None):
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Enumerates methods and adds extra DFT features for DFT methods"""

        X = X.copy()

        dft_features = defaultdict(list)
        dft_feature_names = ["ansatz", "c_hybrid", "x_hybrid", "c_lrc", "x_lrc", "nlc"]

        forcefield_names = frozenset(["uff", "mm94"])
        ml_names = frozenset(["ani1", "ani1x", "ani1cxx", "ani-1", "ani-1x", "ani-1cxx"])
        semiempirical_names = frozenset(
            ["am1", "pm3", "pm6", "pm7", "xtb", "gfn0-xtb", "gfn-xtb", "gfn2-xtb"]
        )
        mp2_names = frozenset(["mp2", "scs-mp2", "sapt0"])
        ccsd_names = frozenset(["ccsd", "sapt2", "sapt2+", "sapt2/3"])
        ccsdt_names = frozenset(["ccsd(t)"])


        method_enum = []
        for method in X.method:
            try:
                info = dft_info.get(method)
                for name in dft_feature_names:
                    dft_features[name].append(getattr(info, name))
                method_enum.append(MethodEnum.dft)
            except (KeyError):
                for name in dft_feature_names[1:]:
                    dft_features[name].append(False)
                dft_features["ansatz"].append(0)

                if method.lower() in forcefield_names:
                    method_enum.append(MethodEnum.forcefield)
                elif method.lower() in ml_names:
                    method_enum.append(MethodEnum.ml)
                elif method.lower() in semiempirical_names:
                    method_enum.append(MethodEnum.semiempirical)
                elif method.lower() in mp2_names:
                    method_enum.append(MethodEnum.mp2)
                elif method.lower() in ccsd_names:
                    method_enum.append(MethodEnum.ccsd)
                elif method.lower() in ccsdt_names:
                    method_enum.append(MethodEnum.ccsdprt)
                else:
                    raise ValueError(f"Unknown method: {method}")

        for name in dft_feature_names:
            X[name] = dft_features[name]
        X["method"] = [int(i) for i in method_enum]

        return X


class CPUFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, params=None):
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # not features to extract
        if 'cpu' not in X:
            return X

        clock = []
        launch = []
        vendor = []
        instructions = []
        for cpu in X["cpu"]:
            info = cpu_info.get(cpu)
            clock.append(info.base_clock / 1_000_000)
            launch.append(info.launch_date)
            vendor.append(info.vendor)
            instructions.append(info.instructions)

        ret_df = X.copy()
        ret_df["cpu_clock_speed"] = clock
        ret_df["cpu_launch_year"] = launch
        ret_df["cpu_vendor"] = vendor
        ret_df["cpu_instructions"] = instructions

        return ret_df


# ----------------------- Not used yet----------------

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables: List=None) -> None:
       self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategoricalImputer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables: List=None):
        self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Temporal variable calculator."""

    def __init__(self, variables: List=None, reference_variable=None):
        self.variables = variables
        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables: List=None):
        self.tol = tol
        self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder based on the target."""

    def __init__(self, variables: List=None):
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items()
                     if value is True}
            raise AttributeError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}')

        return X

