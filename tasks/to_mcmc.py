#!/usr/bin/env python
import os
import gc
import inspect
import copy
import time
import traceback
import tenacity as tn
import wpipe as wp
import numpy as np
import pandas as pd
from scipy import linalg, spatial

if __name__ == '__main__':
    import data_model_utils as dmu
    from ModelCaching import return_models, update_models, append_to_dp, create_model_dp, clear_model_dp_from_wpipe_cache


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='start_walker', value='*')


EXISTING_VORONOI = None

HYPERBOLIC_PARAM = 0.1


def kernel(u):
    return np.exp(-3 * u ** 2)


def make_new_model(*args):
    wp.ThisJob.logprint("ENTERING MAKE_NEW_MODEL")
    models = update_models()
    if len(models):
        deviations = (np.array(models.index.to_list()) - args) / dmu.CHARA_LENGTHS
        distances = linalg.norm(deviations, axis=1)
        weights = kernel(distances)
        structure = np.sum(weights[:, np.newaxis, np.newaxis] * (
            np.matmul(deviations[:, :, np.newaxis], deviations[:, np.newaxis, :])), axis=0) / np.sum(
            weights)
        if np.all(np.isfinite(structure)):
            eigval, eigvec = linalg.eigh(structure)
            changed = np.matmul(linalg.inv(eigvec), deviations[..., np.newaxis])[..., 0]
            squ_conic_ratios = changed ** 2 / (HYPERBOLIC_PARAM ** 2 + distances[:, np.newaxis] ** 2 - changed ** 2)
            axes_weights = np.sqrt(
                weights[:, np.newaxis] * np.clip((1 - squ_conic_ratios) / (1 + squ_conic_ratios), 0, np.inf))
            dim, ind = np.where(changed.T > 0)
            dim = np.unique(dim, return_index=True)
            all_directions = np.all([np.sum(axes_weights[i, d]) >= 1 for d, i in zip(dim[0], np.split(ind, dim[1][1:]))])
            dim, ind = np.where(changed.T < 0)
            dim = np.unique(dim, return_index=True)
            all_directions &= np.all([np.sum(axes_weights[i, d]) >= 1 for d, i in zip(dim[0], np.split(ind, dim[1][1:]))])
        else:
            all_directions = False
    else:
        all_directions = False
    return ~all_directions


def comput_model(*args):
    wp.ThisJob.logprint('MAKE NEW MODEL')
    model_dp = create_model_dp(args)
    wp.ThisJob.firing_event.options['current_dpid'] = model_dp.dp_id
    new_model = dmu.model(*args)
    args_tags = ['a%d' % i for i in range(len(args))]
    append_to_dp(pd.DataFrame([[model_dp.dp_id]+list(args)], columns=['dp_id']+args_tags).join(
        pd.DataFrame([np.array(new_model)], columns=dmu.MODEL_X)).set_index(  # TODO is that MODEL_X needed?
        args_tags), model_dp)
    wp.ThisJob.firing_event.options['current_dpid'] = None
    return new_model


class NegInCentralReg(Exception):
    pass


class InModelLibrary(Exception):
    pass


def interp_model(*args):
    wp.ThisJob.logprint('INTERPOLATE MODEL')
    global EXISTING_VORONOI
    # models = EXISTING_MODELS.drop('name', axis=1)
    models = return_models().drop('dp_id', axis=1)
    # points = (np.array(models.index.to_list()) - args) / dmu.CHARA_LENGTHS
    wp.ThisJob.logprint('STEP 1')
    EXISTING_VORONOI = spatial.Voronoi(np.array(models.index.to_list()) / dmu.CHARA_LENGTHS, incremental=True)  # TODO
    # if EXISTING_VORONOI is None:
    #     EXISTING_VORONOI = spatial.Voronoi(np.array(models.index.to_list()) / dmu.CHARA_LENGTHS, incremental=True)
    # elif EXISTING_VORONOI.npoints < len(models):
    #     EXISTING_VORONOI.add_points(np.array(models.index[EXISTING_VORONOI.npoints:].to_list()) / dmu.CHARA_LENGTHS)
    wp.ThisJob.logprint('STEP 2')
    vor = EXISTING_VORONOI  # vor = copy.copy(EXISTING_VORONOI)  # TODO: THIS DIDN'T WORK :'(
    # wp.ThisJob.logprint('STEP 3')
    volumes = np.zeros(vor.npoints)
    vertices = np.vstack([vor.vertices, np.nan * np.ones(vor.ndim)])
    regions = np.array(vor.regions, dtype=object)[vor.point_region]
    # vor.add_points([np.zeros(vor.ndim)])
    wp.ThisJob.logprint('STEP 4')
    center = args / np.array(dmu.CHARA_LENGTHS)
    distances = np.linalg.norm(vor.points - center, axis=1)
    nearest_model = np.argmin(distances)
    if distances[nearest_model] <= 2*np.finfo('float').resolution:
        return models.to_numpy()[nearest_model]
    # vor.add_points([center])
    try:
        vor.add_points([center])
    except spatial.qhull.QhullError as Err:
        wp.ThisJob.logprint('Encountered a QhullError:\n' + Err.args[0])
        vor.add_points([center])
    central_region = np.array(vor.regions[vor.point_region[-1]])
    # wp.ThisJob.logprint('STEP 5')
    if -1 in central_region:
        raise NegInCentralReg()
    # calculate unit vectors where voronoi centroids are
    # wp.ThisJob.logprint('STEP 6')
    unit_vectors = vor.points / np.linalg.norm(vor.points, axis=1)[:, np.newaxis]
    # dict of ridges' vertices for old voronoi cells intersecting with new central centroid
    # wp.ThisJob.logprint('STEP 7')
    ridges = {np.min(key): vor.vertices[item] for key, item in vor.ridge_dict.items() if vor.npoints - 1 in key}
    # dict of old regions' vertices for old voronoi cells intersecting with new central centroid
    # wp.ThisJob.logprint('STEP 8')
    region_vertices = {key: vertices[regions[key]] for key in ridges.keys()}
    # dict of sub-region' vertices resulting from these intersections
    # wp.ThisJob.logprint('STEP 9')
    subregions = {key: np.vstack([item, region_vertices[key][
        np.matmul(unit_vectors[key], region_vertices[key].T) < np.max(np.matmul(unit_vectors[key], item.T))]])
                  for key, item in ridges.items()}
    # wp.ThisJob.logprint('STEP 10')
    # compute volumes of these sub-regions and save the results
    volumes[list(subregions.keys())] = [spatial.qhull.ConvexHull(subregion).volume
                                        for subregion in subregions.values()]
    # wp.ThisJob.logprint('STEP 11')
    # compute Sibson weights to perform natural neighbor interpolation of the known models
    weights = volumes / np.sum(volumes)
    # while np.sum(weights) != 1.:
    #     weights /= np.sum(weights)
    # perform interpolation
    wp.ThisJob.logprint(
        'RETURN INTERPOLATE, models.to_numpy().T.shape = %s, weights.shape = %s' % (models.to_numpy().T.shape,
                                                                                    weights.shape))
    return np.sum(models.to_numpy().T * weights, axis=1)


def model_fun(*args):
    wp.ThisJob.logprint("ENTERING MODEL_FUN")
    if make_new_model(*args):
        return comput_model(*args)
    else:
        try:
            return interp_model(*args)
        except NegInCentralReg:
            return comput_model(*args)


class NotInDomain(Exception):
    pass


def __after(retry_state):
    try:
        retry_state.outcome.result()
    except Exception as Err:
        with wp.ThisJob.logprint().open('a') as _f:
            _f.write("ENCOUNTERED EXCEPTION, TRACEBACK BELOW:\n")
            traceback.print_tb(Err.__traceback__, file=_f)
            _f.write(repr(Err)+'\n')
            _f.write('RETRYING\n')


def log_likelihood(theta):
    wp.ThisJob.logprint("ENTERING LOG_LIKELIHOOD")
    try:
        if not dmu.domain(theta):
            raise NotInDomain()
        n_extra_args = len(inspect.signature(dmu.log_likelihood_of_model).parameters)-1
        extra_args = tuple(theta[len(theta)-n_extra_args:])
        args = tuple(theta[:len(theta)-n_extra_args])
        for retry in tn.Retrying(
                retry=tn.retry_if_exception_type(ValueError),
                wait=tn.wait_random(),
                after=__after):
            with retry:
                return dmu.log_likelihood_of_model(*((model_fun(*args),) + extra_args))
    except NotInDomain:
        return -np.inf


def log_prior(theta):
    if dmu.domain(theta):
        return 0.0
    else:
        return -np.inf


def log_probability(theta):
    wp.ThisJob.logprint("ENTERING LOG_PROBABILITY")
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def to_mcmc(theta):
    wp.ThisJob.logprint("ENTERING TO_MCMC")
    if dmu.domain(theta):
        return log_probability(theta)
    else:
        return -np.inf


def send_to_mcmc(theta):
    wp.ThisJob.logprint("ENTERING SEND_TO_MCMC")
    log_prob = to_mcmc(theta)
    if log_prob == -np.inf:
        log_prob = '-Inf'
    wp.ThisEvent.options['log_prob'] = log_prob
    wp.ThisEvent.options['new_log_prob'] = True
    wp.ThisJob.logprint("SENT " + str(log_prob) + " TO MCMC")


def listen_for_theta():
    wp.ThisJob.logprint("LISTENING FOR NEW THETA")
    while not wp.ThisEvent.options['new_theta']:
        time.sleep(0.01)
    wp.ThisJob.logprint("DETECTED NEW THETA")
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
        clear_model_dp_from_wpipe_cache()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    listen_send_and_stop()
