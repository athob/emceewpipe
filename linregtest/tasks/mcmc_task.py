#!/usr/bin/env python
import wpipe as wp
import numpy as np
# import pandas as pd
# from scipy.optimize import minimize
import emcee


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='run_mcmc', value='*')


class EventPool:
    def __init__(self, job, pool_length, *args, **kwargs):
        self._job = job
        d = repr(len(repr(pool_length)))
        self._events = list(map(lambda n: job.child_event(*args, tag=('PoolEvent#%0' + d + 'd') % n,
                                                          **kwargs), range(pool_length)))
        self._initialize_events()

    def _initialize_events(self):
        for event in self._events:
            event.options['new_log_prob'] = True
            event.options['new_theta'] = False

    def fire(self):
        for event in self._events:
            event.fire()

    def map(self, func, struct):  # TODO: struct is actually a generator, think about a more suitable implementation
        struct = list(struct)
        struct_length = len(list(struct))
        pool_length = len(self._events)
        output = np.nan * np.empty(struct_length)
        pool_indexes = dict(zip(range(pool_length), pool_length * [-1]))
        n = 0
        m = 0
        while m < struct_length or pool_indexes:
            wp.ThisJob.logprint('\nSTRUCT\t'+repr(m)+'\t'+repr(pool_indexes))
            # looping around the pool to find available event
            # TODO: add management to that loop to restart forced closed events
            while not(self._events[n].options['new_log_prob']) if (n in pool_indexes.keys()) else True:
                n += 1
                n %= pool_length
            self._events[n].options['new_log_prob'] = False
            # check if available event should have a result and get it
            if pool_indexes[n] >= 0:
                log_prob = self._get_log_prob(self._events[n])
                if log_prob == '-Inf':
                    log_prob = -np.inf
                output[pool_indexes[n]] = log_prob
                wp.ThisJob.logprint('\nGRABBING\t'+repr(n)+'\t'+repr(output[pool_indexes[n]]))
            # send element to event if more left, remove from pool_indexes otherwise
            if m < struct_length:
                self._send_theta(self._events[n], func(struct[m]))
                wp.ThisJob.logprint('\nSENDING\t'+repr(m))
                pool_indexes[n] = m
                m += 1
                n += 1
                n %= pool_length
            else:
                wp.ThisJob.logprint('\nDELETING\t'+repr(n))
                del pool_indexes[n]
                self._events[n].options['new_log_prob'] = True
        wp.ThisJob.logprint('\nRETURN\n'+repr(output))
        return output

    def kill(self):
        for event in self._events:
            self._stop_walker(event)

    @staticmethod
    def _send_theta(event, theta):
        event.options['len_theta'] = len(theta)
        for i, theta_i in enumerate(theta):
            event.options['theta_%d' % i] = theta_i
        event.options['new_theta'] = True

    @staticmethod
    def _get_log_prob(event):
        log_prob = event.options['log_prob']
        return log_prob

    @staticmethod
    def _stop_walker(event):
        event.options['stop_walker'] = True
        event.options['new_theta'] = True

# # Choose the "true" parameters.
# M_TRUE = -0.9594
# B_TRUE = 4.294
# F_TRUE = 0.534
# Generate some synthetic data from the model.
# def get_data_synth():
#     n = 50
#     data = {}
#     x_data = data['x'] = np.sort(10 * np.random.rand(n))
#     yerr_data = data['yerr'] = 0.1 + 0.5 * np.random.rand(n)
#     y_data = data['y'] = M_TRUE * x_data + B_TRUE
#     y_data += np.abs(F_TRUE * y_data) * np.random.randn(n)
#     y_data += yerr_data * np.random.randn(n)
#     return pd.DataFrame(data)


# def get_data_target():
#     raw_dp = wp.ThisJob.config.rawdataproducts[0]
#     data = pd.read_csv(raw_dp.path)
#     return data
#
#
# get_data = get_data_target


def convert_utheta(utheta):
    return np.array([utheta[0] * (0.5 - (-5.0)) + (-5.0),
                     utheta[1] * (10.0 - 0.0) + 0.0,
                     utheta[2] * (1.0 - (-10.0)) + (-10.0)])


def convert_theta(theta):
    return np.array([(theta[0] - (-5.0)) / (0.5 - (-5.0)),
                     (theta[1] - 0.0) / (10.0 - 0.0),
                     (theta[2] - (-10.0)) / (1.0 - (-10.0))])


# def domain(theta):
#     m, b, log_f = theta
#     output = -5.0 < m < 0.5 and \
#              0.0 < b < 10.0 and \
#              -10.0 < log_f < 1.0
#     return output
#
#
# def model_fun(m, b, x):
#     return m * x + b
#
#
# def log_likelihood(theta, x, y, yerr):
#     try:
#         if not domain(theta):
#             raise ValueError
#         m, b, log_f = theta
#         model = model_fun(m, b, x)
#         sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
#         return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
#     except ValueError:
#         return -np.inf
#
#
# def log_prior(theta):
#     if domain(theta):
#         return 0.0
#     else:
#         return -np.inf
#
#
# def log_probability(theta, x, y, yerr):
#     lp = log_prior(theta)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + log_likelihood(theta, x, y, yerr)
#
#
# def to_mcmc(utheta, x, y, yerr):
#     theta = convert_utheta(utheta)
#     if domain(theta):
#         return log_probability(theta, x, y, yerr)
#     else:
#         return -np.inf


def to_mcmc(utheta):
    theta = convert_utheta(utheta)
    return theta


# def nll(*args):
#     return -log_likelihood(*args)
#
# def get_maxlikelihood_estimates(initial, data):
#     soln = minimize(nll, initial, args=(data['x'], data['y'], data['yerr']))
#     m_ml, b_ml, log_f_ml = soln.x
#     print("Maximum likelihood estimates:")
#     print("m = {0:.3f}".format(m_ml))
#     print("b = {0:.3f}".format(b_ml))
#     print("f = {0:.3f}".format(np.exp(log_f_ml)))
#     return soln


# def run_mcmc(upos, data):
#     x_data = data['x']
#     y_data = data['y']
#     yerr_data = data['yerr']
#     nwalkers, ndim = upos.shape
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, to_mcmc, args=(x_data, y_data, yerr_data))
#     sampler.run_mcmc(upos, 5000, progress=True)
#     return sampler


def run_mcmc(upos, pool):
    nwalkers, ndim = upos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, to_mcmc, pool=pool)
    sampler.run_mcmc(upos, 5000, progress=True)
    return sampler


def get_flat_samples(sampler):
    tau = sampler.get_autocorr_time()
    print(tau)
    flat_samples = sampler.get_chain(discard=int(np.ceil(np.max(2.5 * tau))),
                                     thin=int(np.ceil(np.max(tau / 2.5))),
                                     flat=True)
    print(flat_samples.shape)
    return convert_utheta(flat_samples.T).T


def save_flat_samples(flat_samples):
    new_dp_filename = wp.ThisJob.target.name + '_mcmc_samples.csv'
    new_dp = wp.ThisJob.config.dataproduct(new_dp_filename, relativepath=wp.ThisJob.config.procpath, group='proc')
    np.savetxt(new_dp.path, flat_samples, delimiter=',')
    wp.ThisJob.child_event('make_corner').fire()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    EVENTPOOL = EventPool(wp.ThisJob, 3, name='start_walker')
    EVENTPOOL.fire()
    # DATA = get_data()
    # SOLN = get_maxlikelihood_estimates(np.array([M_TRUE, B_TRUE, np.log(F_TRUE)]) + 0.1 * np.random.randn(3), DATA)
    # INITIAL = SOLN.x + 1e-4 * np.random.randn(32, 3)
    INITIAL = np.random.rand(32, 3)
    # SAMPLER = run_mcmc(INITIAL, DATA)
    SAMPLER = run_mcmc(INITIAL, EVENTPOOL)
    EVENTPOOL.kill()
    FLAT_SAMPLES = get_flat_samples(SAMPLER)
    save_flat_samples(FLAT_SAMPLES)
