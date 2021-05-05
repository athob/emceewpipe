#! /usr/bin/env python
import numpy as np
import pandas as pd
import wpipe as wp

CHARA_LENGTHS = [1, 1]
DATA_X = DATA_Y = DATA_YERR = None


def get_data():
    raw_dp = wp.ThisJob.config.rawdataproducts[0]
    data = pd.read_csv(raw_dp.path)
    return data


def domain(theta):
    m, b, log_f = theta
    output = -5.0 < m < 0.5 and \
             0.0 < b < 10.0 and \
             -10.0 < log_f < 1.0
    return output


def model(m, b):
    return m * DATA_X + b


def log_likelihood_of_model(tested_model, log_f):
    sigma2 = DATA_YERR ** 2 + tested_model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((DATA_Y - tested_model) ** 2 / sigma2 + np.log(sigma2))


if __name__ == '__main__':
    pass
else:
    DATA = get_data()
    DATA_X, DATA_Y, DATA_YERR = DATA['x'], DATA['y'], DATA['yerr']
