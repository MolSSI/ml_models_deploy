from pprint import pprint
from time import time
import logging
import numpy as np
from sklearn.metrics import make_scorer
from qc_time_estimator import config
from sklearn.model_selection import GridSearchCV
from qc_time_estimator.pipeline import qc_time
from qc_time_estimator.processing.data_management import load_dataset
from sklearn.model_selection import train_test_split



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)



parameters = {
    'mean_gradient_boosting_regressor__max_depth': (10, 100, 10),
}

def mape(y_true, y_pred):
    return - np.abs(100 * (y_true - y_pred) / y_true).mean()


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block

    # change max_rows, None means all data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE, max_rows=None)

    data = data.dropna(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.TARGET]/ 3600.0,
        test_size=config.TEST_SIZE,
        train_size=config.TRAIN_SIZE,
        random_state=config.SEED)

    grid_search = GridSearchCV(qc_time,
                               parameters,
                               scoring=make_scorer(mape),
                               n_jobs=-1,
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
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))