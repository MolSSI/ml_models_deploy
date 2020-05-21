from qc_time_estimator.config import config
import numpy as np
np.random.seed(config.SEED)

import logging
from pprint import pprint
from time import time
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from qc_time_estimator.pipeline import qc_time_nn
from qc_time_estimator.metrics import mape, percentile_rel_90
from qc_time_estimator.processing.data_management import load_dataset, get_train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime
# from sklearn.utils.fixes import loguniform
# from qc_time_estimator.processing.data_management import save_pipeline


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


parameters = {
    'nn_model__input_dim': [22,],
    'nn_model__nodes_per_layer': [(10, 5), (10, 10, 5)],
    'nn_model__dropout': [0.1, 0.2], #[0, 0.1, 0.2, 0.3],  # 0.1 or 0.2
    'nn_model__batch_size': [64, 128, 256, 512],   # 75
    'nn_model__epochs': [100, 200, 300], # [50, 100, 150, 200, 250, 300],  # 200 better
    'nn_model__optimizer': ['adam'], #, 'rmsprop'],  # adam is better
    'nn_model__learning_rate': [0.001, 0.005, 0.01],
}


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block

    # change max_rows, None means all data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE, nrows=None)

    X_train, X_test, y_train, y_test = get_train_test_split(data, test_size=0.2)

    grid_search = GridSearchCV(qc_time_nn,
                               parameters,
                               scoring={
                                   'percentile99': make_scorer(percentile_rel_90, greater_is_better=False),
                                   'MAPE': make_scorer(mape, greater_is_better=False),
                               },
                               refit='percentile99',
                               n_jobs=-1,  # -2 to use all CPUs except one
                               return_train_score=True,
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
    df.to_csv(f'./search_results/grid_search_nn_no_duplicates{datetime.now().strftime("%Y-%m-%d %H:%M")}.csv', index=None)

    y_pred = grid_search.best_estimator_.predict(X_test)
    y_train_pred = grid_search.best_estimator_.predict(X_train)

    print('Y Train 90th percentile: ', percentile_rel_90(y_train, y_train_pred))
    print('Train mape: ', mape(y_train, y_train_pred))

    print('Y Test 90th percentile: ', percentile_rel_90(y_test, y_pred))
    print('Test mape: ', mape(y_test, y_pred))

    # save_pipeline(pipeline_to_persist=grid_search.best_estimator_)