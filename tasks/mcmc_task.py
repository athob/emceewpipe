#!/usr/bin/env python
import wpipe as wp
import numpy as np
import emcee

if __name__ == '__main__':
    import data_model_utils as dmu
    from EventPool import EventPool


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='run_mcmc', value='*')


SUBMISSION_TYPE = 'pbs'
NB_WALKERS = 32
# NB_DIM = 3
# NB_ITERATIONS = 5000


def to_mcmc(utheta):
    theta = dmu.convert_utheta(utheta)
    return theta


def run_mcmc(pool):
    upos = np.random.rand(NB_WALKERS, dmu.NB_DIM)
    this_config = wp.ThisJob.config
    NB_ITERATIONS = this_config.parameters['nb_iterations']
    backend_dp = this_config.dataproduct(filename='emcee_backend.h5',
                                         relativepath=this_config.procpath,
                                         group='proc')
    filename = backend_dp.path
    backend = emcee.backends.HDFBackend(filename)
    sampler = emcee.EnsembleSampler(NB_WALKERS, dmu.NB_DIM, to_mcmc, backend=backend, pool=pool)
    sampler.run_mcmc(None if sampler.iteration else upos, NB_ITERATIONS - sampler.iteration, progress=True)
    return sampler


def get_flat_samples(sampler):
    tau = sampler.get_autocorr_time(quiet=True)  # TODO: quiet should be avoided
    print(tau)
    flat_samples = sampler.get_chain(discard=int(np.ceil(np.max(2.5 * tau))),
                                     thin=int(np.ceil(np.max(tau / 2.5))),
                                     flat=True)
    print(flat_samples.shape)
    return dmu.convert_utheta(flat_samples.T).T


def save_flat_samples(flat_samples):
    new_dp_filename = wp.ThisJob.target.name + '_mcmc_samples.csv'
    new_dp = wp.ThisJob.config.dataproduct(new_dp_filename, relativepath=wp.ThisJob.config.procpath, group='proc')
    np.savetxt(new_dp.path, flat_samples, delimiter=',')
    wp.ThisJob.child_event('make_corner').fire()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conf_params = wp.ThisJob.config.parameters
    MASTER_CACHE = wp.ThisJob.child_event(name='start_cache')
    MASTER_CACHE.fire()
    EVENTPOOL = EventPool(wp.ThisJob, int(conf_params['len_eventpool']), name='start_walker',
                          options={} if SUBMISSION_TYPE is None
                          else {'submission_type': SUBMISSION_TYPE, 'job_time': 40,
                                'walltime': str(conf_params['walltime']),
                                'job_openmp': bool(conf_params['job_openmp'])})
    EVENTPOOL.fire()
    SAMPLER = run_mcmc(EVENTPOOL)
    EVENTPOOL.kill()
    MASTER_CACHE.options['stop_cache'] = True
    FLAT_SAMPLES = get_flat_samples(SAMPLER)
    save_flat_samples(FLAT_SAMPLES)
