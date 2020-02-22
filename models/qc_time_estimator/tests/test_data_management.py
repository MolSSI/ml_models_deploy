import pytest
from qc_time_estimator import config
from qc_time_estimator.processing.data_management import load_dataset, _download_unzip_dataset
import pathlib
import os


@pytest.mark.slow
def test_zenodo_download():

    csv_file = f'{config.DATASET_DIR / config.TRAINING_DATA_FILE}'
    zip_file = f'{config.DATASET_DIR / config.TRAINING_DATA_FILE.rstrip("csv")}zip'

    try:
        os.remove(csv_file)
        os.remove(zip_file)
    except FileNotFoundError:
        pass

    assert not pathlib.Path(csv_file).exists()
    assert not pathlib.Path(zip_file).exists()

    # download and unzip
    _download_unzip_dataset(csv_file)

    assert pathlib.Path(csv_file).exists()
    assert pathlib.Path(zip_file).exists()

    os.remove(csv_file)

    # unzip only
    _download_unzip_dataset(csv_file)

    assert pathlib.Path(csv_file).exists()


@pytest.mark.slow
def test_load_dataset():

    data = load_dataset(file_name=config.TRAINING_DATA_FILE, nrows=5)

    assert data.shape[0] == 5
