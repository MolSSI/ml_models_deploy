from qc_time_estimator.train_pipeline import run_training
from qc_time_estimator import config
from qc_time_estimator import __version__ as _version
import pathlib
import pytest
import logging


@pytest.fixture(scope='session')
def trained_model():

    file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    curr_pipeline_path = config.TRAINED_MODEL_DIR / file_name

    if not pathlib.Path(curr_pipeline_path).exists():
        logging.info("Running training to generate the model for tests")

        run_training(with_accuracy=False)

    return True