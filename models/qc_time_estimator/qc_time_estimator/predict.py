import pandas as pd
import numpy as np
from qc_time_estimator.config import config
from qc_time_estimator.processing.data_management import load_pipeline
from qc_time_estimator.processing.validation import validate_inputs
from qc_time_estimator import __version__ as _version
import logging
from typing import Union, List


logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_qc_time = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, List[dict]]) -> dict:
    """Make a prediction using a saved model pipeline.
    Throws exception for invalid input.

    Parameters
    ----------
    input_data : DataFram or list of dict
        Array of model prediction inputs.

        1- Required input:
            cpu_clock_speed (in MHz, between 500 and 10,000)
            cpu_launch_year: (between 1990 and current year)

            driver: DriverEnum
            method: str

        2- Required one of those two groups:

            molecule
            basis_set

            # OR
            nelec
            nmo

        3- Optional:
            restricted: bool (default=False)
            nthreads: (default=1)

        Other extra fields are ignored and don't cause error.


    Returns
    -------
        Dict with Lit of Predictions for each input row,
        as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _qc_time.predict(validated_data)

    results = {'predictions': prediction, 'version': _version}

    logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results


def get_accuracy(model, X, y):
    """Calculate the prediction acuracy (MAPE) for the given data using the
    given model"""

    pred =  model.predict(X)
    prec_error = (y - pred) / y * 100
    mape = np.abs(prec_error).mean()  # Mean absolute % error

    return mape