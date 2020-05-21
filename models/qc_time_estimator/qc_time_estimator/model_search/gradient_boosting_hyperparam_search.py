from qc_time_estimator.config import config
import numpy as np
np.random.seed(config.SEED)

import logging
from pprint import pprint
from time import time
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from qc_time_estimator.pipeline import qc_time
from qc_time_estimator.metrics import mape, percentile_rel_90
from qc_time_estimator.processing.data_management import load_dataset, get_train_test_split



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


parameters = {
    'mean_gradient_boosting_regressor__max_depth': [50],
}


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block

    # change max_rows, None means all data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE, nrows=None)

    data = data.dropna(axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(
        data,
        data[config.TARGET] / 3600.0,
        test_size=0.2)


    grid_search = GridSearchCV(qc_time,
                               parameters,
                               scoring=make_scorer(percentile_rel_90),
                               n_jobs=-1,
                               cv=3,
                               verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in qc_time.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    best_parameters = grid_search.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    y_pred = grid_search.best_estimator_.predict(X_test)
    y_train_pred = grid_search.best_estimator_.predict(X_train)

    print('Y Train score 90th percentile: ', percentile_rel_90(y_train, y_train_pred))
    print('Train mean: ', mape(y_train, y_train_pred))

    print('Y Test score 90th percentile: ', percentile_rel_90(y_test, y_pred))
    print('Test mean: ', mape(y_test, y_pred))