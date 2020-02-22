import pathlib
import qc_time_estimator
import pandas as pd


PACKAGE_ROOT = pathlib.Path(qc_time_estimator.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test_data.csv'  # for pytest
TRAINING_DATA_FILE = 'train_test_with_header.csv'
ZENODO_TRAINING_DATA_URL = 'https://zenodo.org/record/3669414/files/train_test_with_header.zip?download=1'

PIPELINE_NAME = 'qc_time_estimator'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'


# Model config

TEST_SIZE = 100_000
TRAIN_SIZE = 100_000

MAX_DEPTH = 18

SEED = 1234

TARGET = 'wall_time'

FEATURES = [
    "nthreads",
    "nelec",
    "nvirt",
    "restricted",
    "nmo",
    "o3",
    "o4",
    "nmo3",
    "nmo4",
    "jscale",
    "kscale",
    "diag",
    "driver",
    "method",
    "ansatz",
    "c_hybrid",
    "x_hybrid",
    "c_lrc",
    "x_lrc",
    "nlc",
    "cpu_clock_speed",
    "cpu_launch_year",
]

# Internal Input features to the model (not the user input)
INPUT_FEATURES = [
    "nthreads",
    "driver",
    "method",
    "restricted",
    "cpu_clock_speed",
    "cpu_launch_year",
    "nelec",
    "nmo",
]

# For differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 1e-2

pd.options.display.max_rows = 10
pd.options.display.max_columns = 20



