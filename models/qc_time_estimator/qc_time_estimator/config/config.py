import pathlib
import qc_time_estimator
import pandas as pd


PACKAGE_ROOT = pathlib.Path(qc_time_estimator.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test_data.csv'  # for pytest
TRAINING_DATA_FILE = 'all_data_duplicates_removed.csv'
ZENODO_TRAINING_DATA_URL = f'https://zenodo.org/record/3827947/files/{TRAINING_DATA_FILE.rstrip(".csv")}.zip'
PIPELINE_NAME = 'qc_time_estimator'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'


# ver 1.0.0
# BEST_MODEL_PARAMS = {
#     'mean_gradient_boosting_regressor__max_depth': 18,
# }

# ver 2.0.0
# BEST_MODEL_PARAMS = {
#     'nn_model__input_dim': 22,
#     'nn_model__nodes_per_layer': [10],
#     'nn_model__batch_size': 75,
#     'nn_model__epochs': 200,
#     'nn_model__optimizer': 'adam',
# }


# new clean data, ver 3.0.0
# mean_test_percentile99
# mean_test_MAPE
BEST_MODEL_PARAMS = {
    'nn_model__batch_size': 128,
    'nn_model__dropout': 0.0714014,
    'nn_model__epochs': 298,
    'nn_model__learning_rate': 0.0011857,
    'nn_model__nodes_per_layer': (10, 5),
    'nn_model__optimizer': 'adam'}

# Model config

TEST_SIZE = 0.2
TRAIN_SIZE = None

SEED = 1234

MAX_DEPTH = 18

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



