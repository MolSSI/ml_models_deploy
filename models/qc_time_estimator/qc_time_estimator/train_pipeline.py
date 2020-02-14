from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
from qc_time_estimator import pipeline
from qc_time_estimator.processing.data_management import (
     load_dataset, save_pipeline, save_data)
from qc_time_estimator.config import config
from qc_time_estimator.predict import get_accuracy
from qc_time_estimator import __version__ as _version
from typing import Union, Tuple
import logging


logger = logging.getLogger(__name__)


def run_training(with_accuracy=True) -> Union[Tuple[float, float], None]:
    """
    Run trainging using the data and prams in the config file
    Saves the model (using the name and location in the config)
    Optionally: calculate the train and test accuracy (Mean Absolute Percent Error)


    """

    logger.info('Reading training data.')
    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    logger.debug(data.columns)

    # Drop rows with any NAN values - No imputation
    data = data.dropna(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.TARGET]/ 3600.0,
        test_size=config.TEST_SIZE,
        train_size=config.TRAIN_SIZE,
        random_state=config.SEED)

    logger.info('Start fitting model...')

    # Save some formated test data
    X_test_ = pipeline.input_features_pipeline.fit_transform(X_test, y_test)
    save_data(X=X_test_, y=y_test, file_name=config.TESTING_DATA_FILE, max_rows=5000)

    # train and save the model
    pipeline.qc_time.fit(X_train, y_train)

    logger.info(f'Saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.qc_time)

    if with_accuracy:
        train_mape = get_accuracy(pipeline.qc_time, X_train, y_train)
        test_mape = get_accuracy(pipeline.qc_time, X_test, y_test)

        logger.info(f'Training Mean absolute % error: {train_mape}')
        logger.info(f'Testing Mean absolute % error: {test_mape}')

        return train_mape, test_mape


if __name__ == '__main__':
    run_training(with_accuracy=True)
