from qc_time_estimator.config import config
import numpy as np
np.random.seed(config.SEED)

# from sklearn.metrics import mean_absolute_error
from qc_time_estimator.pipeline import qc_time_nn as model
from qc_time_estimator.pipeline import input_features_pipeline
from qc_time_estimator.processing.data_management import (
     load_dataset, save_pipeline, save_data, current_model_exists, get_train_test_split,
     load_pipeline)
from qc_time_estimator.predict import get_accuracy, make_prediction
from qc_time_estimator import __version__ as _version
from typing import Union, Tuple
import logging


logger = logging.getLogger(__name__)


def run_training(with_accuracy=True, overwrite=True,
                 use_all_data=False) -> Union[Tuple[float, float], None]:
    """
    Run trainging using the data and prams in the config file
    Saves the model (using the name and location in the config)
    Optionally: calculate the train and test accuracy (Mean Absolute Percent Error)

    Parameters
    ----------

    with_accuracy: bool, default True
        If true, calculate and return the training and test accuracy

    overwrite: bool
        overwrite the model file if it exists

    use_all_data: bool
        use all available data for training (used ONLY for out of sample prediction
        in production)

    """

    if not overwrite and current_model_exists():
        logger.info("Model is already saved. Skipping training")
        return

    logger.info('Reading training data.')
    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    logger.debug(f'Training data columns: \n{data.columns}')

    test_size, train_size = config.TEST_SIZE, config.TRAIN_SIZE
    if use_all_data:
        test_size, train_size = None, 0.99

    X_train, X_test, y_train, y_test = get_train_test_split(data, test_size=test_size, train_size=train_size)

    logger.info('Start fitting model...')

    # Save some formatted test data
    X_test_ = input_features_pipeline.fit_transform(X_test, y_test)
    save_data(X=X_test_, y=y_test, file_name=config.TESTING_DATA_FILE)  #, max_rows=5000)

    # train and save the model
    model.set_params(**config.BEST_MODEL_PARAMS)
    model.fit(X_train, y_train)

    logger.info(f'Saving model version: {_version}')
    save_pipeline(pipeline_to_persist=model)

    if with_accuracy:
        train_mape, train_99per = get_accuracy(model, X_train, y_train)
        test_mape, test_99per = get_accuracy(model, X_test, y_test)

        logger.info(f'Training Mean absolute % error: {train_mape}')
        logger.info(f'Testing Mean absolute % error: {test_mape}')

        logger.info(f'Training 99th Percentile % error: {train_99per}')
        logger.info(f'Testing 99th Percentile % error: {test_99per}')

        return train_mape, test_mape


def run_testing(file_name=config.TESTING_DATA_FILE) -> Tuple[float, float]:
    """
    Run testing using held out data
    """

    test_data = load_dataset(file_name=file_name)

    pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    curr_model = load_pipeline(file_name=pipeline_file_name)
    test_mape, test_99per = get_accuracy(curr_model, test_data, test_data['wall_time'])

    logger.info(f'Testing Mean absolute % error: {test_mape}')
    logger.info(f'Testing 99th Percentile % error: {test_99per}')

    return test_mape, test_99per


if __name__ == '__main__':
    run_training()
