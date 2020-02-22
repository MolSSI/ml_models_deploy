import pytest

from qc_time_estimator.predict import make_prediction, get_accuracy
from qc_time_estimator.processing.data_management import load_dataset, load_pipeline
from qc_time_estimator import config
from qc_time_estimator import __version__ as _version


def test_single_prediction(trained_model):

    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    # get first row of the Dataframe
    single_test_input = test_data.iloc[0:1]

    pred = make_prediction(input_data=single_test_input)

    assert pred is not None
    assert isinstance(pred.get('predictions')[0], float)
    assert pytest.approx(pred.get('predictions')[0], 0.0285, abs=1e-3)


def test_multiple_predictions(trained_model):

    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)

    pred = make_prediction(input_data=test_data)

    assert pred is not None
    assert len(pred.get('predictions')) == test_data.shape[0]

    pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    curr_model = load_pipeline(file_name=pipeline_file_name)
    test_mape = get_accuracy(curr_model, test_data, test_data['wall_time'])
    print(f'Test MAPE score: {test_mape}')

    # Current Model expected MAPE accuracy is ~12.0
    assert pytest.approx(test_mape, 12.0, abs=1e-3)



