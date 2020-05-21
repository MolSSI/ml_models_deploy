import numpy as np
from keras import backend


def percentile_rel_90(y_true, y_pred):
    err = backend.abs(y_true - y_pred) / y_true * 100
    return np.percentile(list(err), 90)

def mape(y_true, y_pred):
    return float(backend.mean(backend.abs(y_true - y_pred) * 100 / y_true))