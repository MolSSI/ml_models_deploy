import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import requests
import shutil
from qc_time_estimator.config import config
from qc_time_estimator import __version__ as _version
import logging
from typing import List
from zipfile import ZipFile
import pathlib


logger = logging.getLogger(__name__)


def _download_unzip_dataset(file_name: str) -> None:
    """Try download and unziping the datset zip file from zenodo"""

    zip_file = f"{file_name.rstrip('.csv')}.zip"
    if not pathlib.Path(zip_file).exists():
        # Get from Zenodo
        logger.info('Downloading training data from Zenodo...')
        url = config.ZENODO_TRAINING_DATA_URL
        r = requests.get(url, verify=False, stream=True)
        r.raw.decode_content = True
        with open(zip_file, 'wb') as f:
            logger.info('Saving downloaded training data')
            shutil.copyfileobj(r.raw, f)

    logger.info('Unzipping training data....')
    with ZipFile(zip_file, 'r') as zip_obj:
        zip_obj.extractall(config.DATASET_DIR)

    logger.info('Unzipping done.')


def load_dataset(*, file_name: str, nrows=None) -> pd.DataFrame:

    # check if file exists, otherwise, try unziping
    pathlib_file = config.DATASET_DIR / file_name

    if not pathlib_file.exists():  # try downloading and unzipping it
        _download_unzip_dataset(pathlib_file)

    data = pd.read_csv(pathlib_file, nrows=nrows)

    return data


def save_pipeline(*, pipeline_to_persist):
    """Persist the pipeline.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    logger.info(f'saved pipeline: {save_file_name}')


def curr_model_exists():
    file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    path = config.TRAINED_MODEL_DIR / file_name

    return path.exists()

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]):
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.
    """

    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_data(*, X, y, file_name : str, max_rows=-1):

    save_path = config.DATASET_DIR / file_name
    tmp = pd.concat([X, y], axis=1)

    if max_rows:
        tmp.iloc[:max_rows].to_csv(save_path, index=False)
    else:
        tmp.to_csv(save_path, index=False)
