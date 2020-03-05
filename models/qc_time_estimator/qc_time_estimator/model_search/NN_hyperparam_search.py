from qc_time_estimator.config import config
import numpy as np
np.random.seed(config.SEED)

import logging
from pprint import pprint
from time import time
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from qc_time_estimator.pipeline import qc_time_nn
from qc_time_estimator.metrics import mape, percentile_rel_99
from qc_time_estimator.processing.data_management import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from qc_time_estimator.processing.data_management import save_pipeline



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


parameters = {
    'nn_model__input_dim': [22],
    # 'nn_model__nodes_per_layer': [[10]],
    'nn_model__batch_size': [100],  # better than 25
    'nn_model__epochs': [50], #, 500, 25],  # 30 was good, 100 less
    'nn_model__optimizer': ['adam'], #, 'rmsprop'],
}


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block

    # change max_rows, None means all data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE, nrows=None)

    data = data.dropna(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.TARGET]/ 3600.0,
        test_size=0.2,
        # test_size=config.TEST_SIZE,
        # train_size=config.TRAIN_SIZE,
        random_state=config.SEED)

    grid_search = GridSearchCV(qc_time_nn,
                               parameters,
                               scoring=make_scorer(percentile_rel_99),
                               n_jobs=-1,
                               cv=KFold(n_splits=2, random_state=0),
                               verbose=0)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in qc_time_nn.steps])
    print("parameters:")
    pprint(parameters)

    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


    y_pred = grid_search.best_estimator_.predict(X_test)
    y_train_pred = grid_search.best_estimator_.predict(X_train)

    print('Y Train 99th percentile: ', percentile_rel_99(y_train, y_train_pred))
    print('Train mape: ', mape(y_train, y_train_pred))

    print('Y Test 99th percentile: ', percentile_rel_99(y_test, y_pred))
    print('Test mape: ', mape(y_test, y_pred))
