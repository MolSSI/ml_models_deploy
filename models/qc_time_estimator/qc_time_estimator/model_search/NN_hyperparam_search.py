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
import pandas as pd
from datetime import datetime
from sklearn.utils.fixes import loguniform
from qc_time_estimator.processing.data_management import save_pipeline


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


parameters = {
    'nn_model__input_dim': [22,],
    'nn_model__nodes_per_layer': [(10, 5), (10, 10, 5)],
    'nn_model__dropout': [0.1, 0.2], #[0, 0.1, 0.2, 0.3],  # 0.1 or 0.2
    'nn_model__batch_size': [75, 100], #[50, 75, 100, 150, 200, 300],  # 75
    'nn_model__epochs': [200, 300], # [50, 100, 150, 200, 250, 300],  # 200 better
    'nn_model__optimizer': ['adam'], #, 'rmsprop'],  # adam is better
}


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block

    # change max_rows, None means all data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE, nrows=None)

    data = data.dropna(axis=0)

    bins = np.linspace(0, data.shape[0], 100)  # 100 bins
    y_binned = np.digitize(data[config.TARGET]/ 3600.0, bins)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.TARGET]/ 3600.0,
        test_size=0.2,
        stratify=y_binned,
        # test_size=config.TEST_SIZE,
        # train_size=config.TRAIN_SIZE,
        random_state=config.SEED)

    grid_search = GridSearchCV(qc_time_nn,
                               parameters,
                               scoring={
                                   'percentile99': make_scorer(percentile_rel_99, greater_is_better=False),
                                   'MAPE': make_scorer(mape, greater_is_better=False),
                               },
                               refit='percentile99',
                               n_jobs=-1,
                               cv=KFold(n_splits=2, random_state=0),
                               verbose=1)

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

    df = pd.DataFrame(grid_search.cv_results_)
    df.to_csv(f'./search_results/grid_search_nn_statified{datetime.now().strftime("%Y-%m-%d %H:%M")}.csv', index=None)

    y_pred = grid_search.best_estimator_.predict(X_test)
    y_train_pred = grid_search.best_estimator_.predict(X_train)

    print('Y Train 99th percentile: ', percentile_rel_99(y_train, y_train_pred))
    print('Train mape: ', mape(y_train, y_train_pred))

    print('Y Test 99th percentile: ', percentile_rel_99(y_test, y_pred))
    print('Test mape: ', mape(y_test, y_pred))

    # save_pipeline(pipeline_to_persist=grid_search.best_estimator_)