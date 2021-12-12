#! /usr/bin/env python
"""
Can be modified
"""
import sys

import numpy as np
import wpipe as wp
from astropy import units, table
from dustpy import SynDustPy

LABELS = ["$T_{star}$", "$\\tau$", "$f_{T}$"]
CHARA_LENGTHS = [100, 0.5, 0.01]
DATA = None
MODEL_X = None
NB_DIM = len(LABELS)

DISTANCE = 778 * units.kpc
REDSHIFT = -0.001001

LAMBDA_UNIT = units.micron
IRRADI_UNIT = units.Unit('erg/(cm2 s)')

FILTER_IDS = ['Spitzer/IRAC.I1', 'Spitzer/IRAC.I2', 'Spitzer/IRAC.I3', 'Spitzer/IRAC.I4', 'Spitzer/MIPS.24mu']
FILTER_IDS += ['HST/ACS_WFC.F435W', 'HST/ACS_WFC.F606W', 'HST/ACS_WFC.F814W', 'HST/WFC3_IR.F110W', 'HST/WFC3_IR.F160W']

T_STAR_MIN = 2500.
T_STAR_MAX = 25000.
DEPTH_MIN = 0.
DEPTH_MAX = 30  # np.inf
# T_INNER_MIN = 0.
# T_INNER_MAX = 1500.
T_INN_F_MIN = 0.01
T_INN_F_MAX = 1.

LOG_BOUND = 0.9
SYNDUSTPY = None


def convert_utheta(utheta):
    return np.array([np.exp(utheta[0] * (np.log(T_STAR_MAX) - np.log(T_STAR_MIN)) + np.log(T_STAR_MIN)),
                     DEPTH_MAX*np.log(1 - utheta[1])/np.log(1-LOG_BOUND),
                     # utheta[2] * (T_INNER_MAX - T_INNER_MIN) + T_INNER_MIN])
                     utheta[2] * (T_INN_F_MAX - T_INN_F_MIN) + T_INN_F_MIN])


def convert_theta(theta):
    return np.array([(np.log(theta[0]) - np.log(T_STAR_MIN))/(np.log(T_STAR_MAX) - np.log(T_STAR_MIN)),
                     1 - (1 - LOG_BOUND) ** (theta[1] / DEPTH_MAX),
                     # (theta[2] - T_INNER_MIN) / (T_INNER_MAX - T_INNER_MIN)])
                     (theta[2] - T_INN_F_MIN) / (T_INN_F_MAX - T_INN_F_MIN)])


def prepare_syndustpy(filter_ids=None):
    # TODO: SHOULD GRAB HERE SDP PARAMETERS FROM CONFIG
    if filter_ids is None:
        filter_ids = FILTER_IDS
    return SynDustPy(filterids=filter_ids, distance=DISTANCE, src_redshift=REDSHIFT)


def prepare_data_model():
    global SYNDUSTPY, DATA, MODEL_X
    raw_dp = wp.ThisJob.config.rawdataproducts[0]
    mag_data = table.QTable.read(raw_dp.path)
    MODEL_X = mag_data['band'].tolist()
    SYNDUSTPY = prepare_syndustpy(MODEL_X)
    zeropoints = SYNDUSTPY.band_data.zeropoint.to(IRRADI_UNIT / LAMBDA_UNIT,
                                                  equivalencies=units.spectral_density(
                                                      SYNDUSTPY.band_data.wavelengtheff.to(LAMBDA_UNIT)))
    magnitudes = units.Magnitude(mag_data['mag'])
    mag_errors = units.Magnitude(mag_data['sig'])
    DATA = table.QTable([magnitudes.physical * zeropoints,
                         ((magnitudes - mag_errors).physical - magnitudes.physical) * zeropoints,
                         (magnitudes.physical - (magnitudes + mag_errors).physical) * zeropoints],
                        names=['flux_obs', 'upper_err', 'lower_err'])


def domain(theta):
    # temp_star, depth, temp_inner = theta
    temp_star, depth, temp_inn_f = theta
    output = T_STAR_MIN < temp_star < T_STAR_MAX and \
             DEPTH_MIN < depth < DEPTH_MAX and \
             T_INN_F_MIN < temp_inn_f < T_INN_F_MAX  # T_INNER_MIN < temp_inner < T_INNER_MAX
    return output


def model(temp_star, depth, temp_inn_f):  # (temp_star, temp_inner, depth):
    temp_inner = temp_inn_f * temp_star
    dusty_params = {'star_temperature': temp_star,
                    'v_band_optical_depth': depth,
                    'inner_edge_temperature': temp_inner}
    output_sdp = SYNDUSTPY.run(**dusty_params)  # TODO: catch ValueError or understand what happened?
    sys.stdout.flush()
    flux = output_sdp.flux_by_band.to(IRRADI_UNIT / LAMBDA_UNIT)
    return flux.value


def log_likelihood_of_model(tested_model):
    sigma2_up = DATA['upper_err'] ** 2
    sigma2_low = DATA['lower_err'] ** 2
    sigma2_mean = ((DATA['upper_err'] + DATA['lower_err']) / 2) ** 2
    difference = DATA['flux_obs'] - tested_model*(IRRADI_UNIT / LAMBDA_UNIT)  # diff > 0 <=> DATA['flux_obs'] > tested_model <=> tested_model € sigma2_low <=> sigma2 = sigma2_low
    sigma2 = np.vstack([sigma2_low, sigma2_up])[(difference <= 0).astype('int'), np.arange(len(difference))]
    return (-0.5 * np.sum(difference ** 2 / sigma2 + np.log(sigma2_mean.value))).value


if __name__ == '__main__':
    pass
elif hasattr(wp, 'ThisJob'):
    prepare_data_model()
