#!/usr/bin/env python
import inspect
import copy
import time
import tenacity as tn
import wpipe as wp
import numpy as np
import pandas as pd
from scipy import linalg, spatial


if __name__ == '__main__':
    import data_model_utils as dmu


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='start_walker', value='*')


EXISTING_MODELS = pd.DataFrame()
EXISTING_VORONOI = None

HYPERBOLIC_PARAM = 0.1


def wait_model_dp_ready(model_dp):
    for retry in tn.Retrying(retry=tn.retry_if_exception_type(KeyError), wait=tn.wait_random()):
        with retry:
            while not model_dp.options['ready']:
                time.sleep(1)


def load_models(model_dps):
    if len(model_dps):
        return pd.concat([pd.read_csv(model_dp.path).set_index(['a0', 'a1'])
                          for model_dp in model_dps if wait_model_dp_ready(model_dp) is None])
    else:
        return pd.DataFrame()


def update_models():
    global EXISTING_MODELS
    proc_dps = wp.ThisJob.config.procdataproducts
    if len(EXISTING_MODELS):
        missing = ~np.in1d([proc_dp.filename for proc_dp in proc_dps], list(EXISTING_MODELS['name']))
    else:
        missing = slice(None)
    EXISTING_MODELS = pd.concat([EXISTING_MODELS, load_models(np.array(proc_dps)[missing])])
    return EXISTING_MODELS


def kernel(u):
    return np.exp(-3 * u ** 2)


def make_new_model(*args):
    models = update_models()
    wp.ThisJob.logprint('\n\nMODELS UPDATED\n\n')
    if len(models):
        deviations = (np.array(models.index.to_list()) - args) / dmu.CHARA_LENGTHS
        distances = linalg.norm(deviations, axis=1)
        weights = kernel(distances)
        structure = np.sum(weights[:, np.newaxis, np.newaxis] * (
            np.matmul(deviations[:, :, np.newaxis], deviations[:, np.newaxis, :])), axis=0) / np.sum(
            weights)
        eigval, eigvec = linalg.eigh(structure)
        changed = np.matmul(linalg.inv(eigvec), deviations[..., np.newaxis])[..., 0]
        squ_conic_ratios = changed ** 2 / (HYPERBOLIC_PARAM ** 2 + distances[:, np.newaxis] ** 2 - changed ** 2)
        axes_weights = np.sqrt(weights[:, np.newaxis] * np.clip((1 - squ_conic_ratios) / (1 + squ_conic_ratios), 0, np.inf))
        dim, ind = np.where(changed.T > 0)
        dim = np.unique(dim, return_index=True)
        all_directions = np.all([np.sum(axes_weights[i, d]) >= 1 for d, i in zip(dim[0], np.split(ind, dim[1][1:]))])
        dim, ind = np.where(changed.T < 0)
        dim = np.unique(dim, return_index=True)
        all_directions &= np.all([np.sum(axes_weights[i, d]) >= 1 for d, i in zip(dim[0], np.split(ind, dim[1][1:]))])
    else:
        all_directions = False
    return ~all_directions


def comput_model(*args):
    wp.ThisJob.logprint('\n\nMAKE NEW MODEL\n\n')
    model_dp = wp.ThisJob.config.dataproduct(filename=('M' + 2 * '_%.10e' + '.csv') % args,
                                             relativepath=wp.ThisJob.config.procpath,
                                             group='proc',
                                             options={'ready': False})
    new_model = dmu.model(*args)
    args_tags = ['a%d' % i for i in range(len(args))]
    pd.DataFrame([[model_dp.filename]+list(args)], columns=['name']+args_tags).join(
        pd.DataFrame([np.array(new_model)], columns=dmu.DATA_X)).set_index(  # TODO is that DATA_X needed?
        args_tags).to_csv(model_dp.path)
    model_dp.options['ready'] = True
    return new_model


def interp_model(*args):
    wp.ThisJob.logprint('\n\nINTERPOLATE MODEL\n\n')
    global EXISTING_MODELS, EXISTING_VORONOI
    models = EXISTING_MODELS.drop('name', axis=1)
    # points = (np.array(models.index.to_list()) - args) / dmu.CHARA_LENGTHS
    if EXISTING_VORONOI is None:
        EXISTING_VORONOI = spatial.Voronoi(np.array(models.index.to_list()) / dmu.CHARA_LENGTHS, incremental=True)
    elif EXISTING_VORONOI.npoints < len(models):
        EXISTING_VORONOI.add_points(np.array(models.index[EXISTING_VORONOI.npoints:].to_list()) / dmu.CHARA_LENGTHS)
    vor = copy.copy(EXISTING_VORONOI)
    volumes = np.zeros(vor.npoints)
    vertices = np.vstack([vor.vertices, np.nan * np.ones(vor.ndim)])
    regions = np.array(vor.regions)[vor.point_region]
    # vor.add_points([np.zeros(vor.ndim)])
    vor.add_points([args / np.array(dmu.CHARA_LENGTHS)])
    central_region = np.array(vor.regions[vor.point_region[-1]])
    if -1 in central_region:
        raise
    # calculate unit vectors where voronoi centroids are
    unit_vectors = vor.points / np.linalg.norm(vor.points, axis=1)[:, np.newaxis]
    # dict of ridges' vertices for old voronoi cells intersecting with new central centroid
    ridges = {np.min(key): vor.vertices[item] for key, item in vor.ridge_dict.items() if vor.npoints - 1 in key}
    # dict of old regions' vertices for old voronoi cells intersecting with new central centroid
    region_vertices = {key: vertices[regions[key]] for key in ridges.keys()}
    # dict of sub-region' vertices resulting from these intersections
    subregions = {key: np.vstack([item, region_vertices[key][
        np.matmul(unit_vectors[key], region_vertices[key].T) < np.max(np.matmul(unit_vectors[key], item.T))]])
                  for key, item in ridges.items()}
    # compute volumes of these sub-regions and save the results
    volumes[list(subregions.keys())] = [spatial.qhull.ConvexHull(subregion).volume
                                        for subregion in subregions.values()]
    # compute Sibson weights to perform natural neighbor interpolation of the known models
    weights = volumes / np.sum(volumes)
    # while np.sum(weights) != 1.:
    #     weights /= np.sum(weights)
    # perform interpolation
    return np.sum(models.to_numpy().T * weights, axis=1)


def model_fun(*args):
    if make_new_model(*args):
        return comput_model(*args)
    else:
        try:
            return interp_model(*args)
        except Exception:
            return comput_model(*args)


def log_likelihood(theta):
    try:
        if not dmu.domain(theta):
            raise ValueError
        n_extra_args = len(inspect.signature(dmu.log_likelihood_of_model).parameters)-1
        extra_args = tuple(theta[len(theta)-n_extra_args:])
        args = tuple(theta[:len(theta)-n_extra_args])
        return dmu.log_likelihood_of_model(*((model_fun(*args),) + extra_args))
    except ValueError:
        return -np.inf


def log_prior(theta):
    if dmu.domain(theta):
        return 0.0
    else:
        return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def to_mcmc(theta):
    if dmu.domain(theta):
        return log_probability(theta)
    else:
        return -np.inf


def send_to_mcmc(theta):
    log_prob = to_mcmc(theta)
    wp.ThisJob.logprint('\nSENDING\t' + repr(log_prob))
    if log_prob == -np.inf:
        log_prob = '-Inf'
    wp.ThisEvent.options['log_prob'] = log_prob
    wp.ThisEvent.options['new_log_prob'] = True


def listen_for_theta():
    while not wp.ThisEvent.options['new_theta']:
        time.sleep(0.01)
    wp.ThisEvent.options['new_theta'] = False
    theta = [wp.ThisEvent.options['theta_%d' % i] for i in range(wp.ThisEvent.options['len_theta'])]
    return theta


def check_for_stop():
    try:
        return wp.ThisEvent.options['stop_walker']
    except KeyError:
        return False


def listen_send_and_stop():
    while not check_for_stop():
        theta = listen_for_theta()
        send_to_mcmc(theta)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    listen_send_and_stop()
