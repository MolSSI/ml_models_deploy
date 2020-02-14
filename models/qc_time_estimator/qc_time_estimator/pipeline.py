from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from qc_time_estimator.processing import preprocessors
from qc_time_estimator.processing import features_extraction
from qc_time_estimator.config import config

import logging


_logger = logging.getLogger(__name__)


features_steps = {
    'feature_extractor': features_extraction.FeatureExtractor(config.FEATURES),
    # 'debug1': features_extraction.DebugStep(),
    'driver_encoder': preprocessors.DriverEncoder(),
    # 'debug2': features_extraction.DebugStep(),
    'cpu_features': preprocessors.CPUFeatures(),
    # 'debug3': features_extraction.DebugStep(),
    'method_features': preprocessors.MethodFeatures(),
    # 'debug4': features_extraction.DebugStep(),
    'feature_selector': features_extraction.FeatureSelector(features=config.FEATURES),
    # 'debug5': features_extraction.DebugStep(),
}

model_mean = GradientBoostingRegressor(
    loss="ls",
    max_depth=config.MAX_DEPTH,
    random_state=config.SEED
)

model_low_98 = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.01,
    max_depth=config.MAX_DEPTH,
    random_state=config.SEED
)

model_high_98 = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.99,
    max_depth=config.MAX_DEPTH,
    random_state=config.SEED
)

input_features_pipeline = Pipeline([
    ('feature_extractor', features_extraction.FeatureExtractor(config.FEATURES)),
    ('cpu_features', preprocessors.CPUFeatures()),
    ('feature_selector', features_extraction.FeatureSelector(features=config.INPUT_FEATURES)),
])

qc_time = Pipeline([
    *[(k,v) for k, v in features_steps.items()],
    # ('scaler', MinMaxScaler()),
    ('mean_gradient_boosting_regressor', model_mean)
])

