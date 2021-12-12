#!/usr/bin/env python
import numpy as np
from scipy import special
import wpipe as wp
import matplotlib.pyplot as plt
import corner

if __name__ == '__main__':
    import data_model_utils as dmu


CORNER_RANGE = 0.98
PLUSSIGMA = (1+special.erf(2**-.5))/2
MINUSSIGMA = (1+special.erf(-2**-.5))/2


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='make_corner', value='*')


def get_flat_samples():
    dp_filename = wp.ThisJob.target.name + '_mcmc_samples.csv'
    samples_dp = wp.ThisJob.config.dataproduct(dp_filename, group='proc')
    flat_samples = np.loadtxt(samples_dp.path, delimiter=',')
    return flat_samples


def make_corner_plot(flat_samples):
    fig = corner.corner(
        flat_samples, range=dmu.NB_DIM*[CORNER_RANGE], labels=dmu.LABELS, quantiles=[MINUSSIGMA, 0.5, PLUSSIGMA], show_titles=True
    )
    return fig


def save_corner_plot(fig):
    new_dp_filename = wp.ThisJob.target.name + '_mcmc_corner.pdf'
    new_dp = wp.ThisJob.config.dataproduct(new_dp_filename, relativepath=wp.ThisJob.config.procpath, group='proc')
    fig.savefig(new_dp.path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    FLAT_SAMPLES = get_flat_samples()
    FIG = make_corner_plot(FLAT_SAMPLES)
    save_corner_plot(FIG)
