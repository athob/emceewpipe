#! /usr/bin/env python
"""
Can be modified
"""
import numpy as np
import pandas as pd
import wpipe as wp

CHARA_LENGTHS = [0.1, 0.1]
DATA = None
MODEL_X = None


def convert_utheta(utheta):
    return np.array([utheta[0] * (0.5 - (-5.0)) + (-5.0),
                     utheta[1] * (10.0 - 0.0) + 0.0,
                     utheta[2] * (1.0 - (-10.0)) + (-10.0)])


def convert_theta(theta):
    return np.array([(theta[0] - (-5.0)) / (0.5 - (-5.0)),
                     (theta[1] - 0.0) / (10.0 - 0.0),
                     (theta[2] - (-10.0)) / (1.0 - (-10.0))])


def prepare_data_model():
    global DATA, MODEL_X
    raw_dp = wp.ThisJob.config.rawdataproducts[0]
    DATA = pd.read_csv(raw_dp.path)
    MODEL_X = DATA['x']


def domain(theta):
    m, b, log_f = theta
    output = -5.0 < m < 0.5 and \
             0.0 < b < 10.0 and \
             -10.0 < log_f < 1.0
    return output


def model(m, b):
    return m * DATA['x'] + b


def log_likelihood_of_model(tested_model, log_f):
    sigma2 = DATA['yerr'] ** 2 + tested_model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((DATA['y'] - tested_model) ** 2 / sigma2 + np.log(sigma2))


if __name__ == '__main__':
    pass
elif hasattr(wp, 'ThisJob'):
    prepare_data_model()
