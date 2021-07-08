#!/usr/bin/env python
import os
import sys
import time
import tenacity as tn
import wpipe as wp
import pandas as pd


PARENT_IS_MASTER_CACHE = os.path.basename(sys.argv[0]) == 'master_cache.py'
try:
    LEN_EVENTPOOL = int(wp.ThisJob.parameters['len_eventpool'])
except AttributeError:
    LEN_EVENTPOOL = None


def wait_model_dp_ready(model_dp):
    for retry in tn.Retrying(retry=tn.retry_if_exception_type(KeyError), wait=tn.wait_random()):
        with retry:
            while not model_dp.options['ready']:
                time.sleep(1)


def read_model_dp(model_dp):
    temp = pd.read_csv(model_dp.path)
    model = temp.set_index([tag for tag in temp.columns if tag[:1] == 'a'])
    if PARENT_IS_MASTER_CACHE:
        model_dp.options['cached'] = True
    else:
        model_dp.options['readings'] += 1
    return model


def create_model_dp(args):
    model_dp = wp.ThisJob.config.dataproduct(filename=('M' + len(args) * '_%.10e' + '.csv') % args,
                                             relativepath=wp.ThisJob.config.procpath,
                                             group='proc',
                                             data_type='Model',
                                             options={'ready': False, 'readings': 0, 'cached': False})
    return model_dp


def create_cache_dp():
    cache_dp = wp.ThisJob.config.dataproduct(filename="Cache.csv",
                                             relativepath=wp.ThisJob.config.procpath,
                                             group='proc',
                                             data_type='Cache',
                                             options={'ready': False})
    return cache_dp


def get_cache_dp():
    cache_dp = wp.DataProduct.select(dpowner_id=wp.ThisJob.config_id, group='proc', data_type='Cache')[0]
    return cache_dp


if PARENT_IS_MASTER_CACHE:
    EXISTING_MODELS = pd.DataFrame()
else:
    try:
        EXISTING_MODELS = read_model_dp(get_cache_dp())
    except (IndexError, AttributeError):
        EXISTING_MODELS = pd.DataFrame()


def load_models(model_dps):
    # wp.ThisJob.logprint('\nLOAD_MODELS')
    if len(model_dps):
        return pd.concat([read_model_dp(model_dp) for model_dp in model_dps if wait_model_dp_ready(model_dp) is None])
    else:
        return pd.DataFrame()


def update_models():
    global EXISTING_MODELS
    wp.ThisJob.logprint("UPDATING MODELS: current len(EXISTING_MODELS) = %d" % len(EXISTING_MODELS))
    dp_ids = list(EXISTING_MODELS['dp_id']) if len(EXISTING_MODELS) else []
    proc_dps = wp.DataProduct.select(wp.si.DataProduct.id.not_in(dp_ids),
                                     dpowner_id=wp.ThisJob.config_id, group='proc', data_type='Model')
    # filenames = [proc_dp.filename for proc_dp in proc_dps]
    # missing = np.array([os.path.splitext(name)[1] == '.csv' for name in filenames])
    # if len(EXISTING_MODELS):
    #     missing &= ~np.in1d(filenames, list(EXISTING_MODELS['name']))
    # EXISTING_MODELS = pd.concat([EXISTING_MODELS, load_models(np.array(proc_dps)[missing])])
    EXISTING_MODELS = pd.concat([EXISTING_MODELS, load_models(proc_dps)])
    return EXISTING_MODELS


def return_models():
    global EXISTING_MODELS
    return EXISTING_MODELS


def delete_cached_models():
    model_dps = wp.DataProduct.select(dpowner_id=wp.ThisJob.config_id, group='proc', data_type='Model')
    for model_dp in model_dps:
        if model_dp.options['readings'] == LEN_EVENTPOOL and model_dp.options['cached']:
            model_dp.delete()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
