#!/usr/bin/env python
import os
import sys
import time
import tenacity as tn
import wpipe as wp
import pandas as pd

PARENT_IS_MASTER_CACHE = os.path.basename(sys.argv[0]) == 'master_cache.py'
try:
    LEN_EVENTPOOL = int(wp.ThisJob.config.parameters['len_eventpool'])
except AttributeError:
    LEN_EVENTPOOL = None


def wait_model_dp_ready(model_dp, or_skip_after=5):
    start = time.time()
    for retry in tn.Retrying(
            retry=(tn.retry_if_exception_type(KeyError) | tn.retry_if_exception_type(wp.si.exc.NoResultFound)),
            wait=tn.wait_random()):
        with retry:
            while not model_dp.options['ready']:
                if time.time() - start < or_skip_after:
                    time.sleep(1)
                else:
                    return 1


def read_model_dp(model_dp):
    for retry in tn.Retrying(
            retry=tn.retry_if_exception_type(FileNotFoundError),
            after=lambda retry_state:
            wp.ThisJob.logprint("Failed first reading attempt of %s; entering retrying loop" % model_dp.path)
            if retry_state.attempt_number == 1 else None,
            wait=tn.wait_random()):
        with retry:
            temp = pd.read_csv(model_dp.path)
            wp.ThisJob.logprint("Succesfully read %s at attempt #%d" % (model_dp.path,
                                                                        retry.retry_state.attempt_number))
    if not temp.empty:
        model = temp.set_index([tag for tag in temp.columns if tag[:1] == 'a'])
    else:
        model = pd.DataFrame()
    if model_dp.data_type == 'Model':
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


if False:  # PARENT_IS_MASTER_CACHE: TODO different implementation?
    EXISTING_MODELS = pd.DataFrame()
else:
    try:
        wp.ThisJob.logprint('ATTEMPTING LOADING CACHE DATAPRODUCT')
        EXISTING_MODELS = read_model_dp(get_cache_dp())
        wp.ThisJob.logprint('CACHE DATAPRODUCT LOADED')
    except (IndexError, AttributeError):
        EXISTING_MODELS = pd.DataFrame()


def load_models(model_dps):
    # wp.ThisJob.logprint('\nLOAD_MODELS')
    temp = [read_model_dp(model_dp) for model_dp in model_dps if wait_model_dp_ready(model_dp) is None]
    if len(temp):
        return pd.concat(temp)
    else:
        return pd.DataFrame()


def update_models():
    global EXISTING_MODELS
    wp.ThisJob.logprint("UPDATING MODELS: current EXISTING_MODELS.shape = %s" % repr(EXISTING_MODELS.shape))
    wp.ThisJob.logprint("                         EXISTING_MODELS.columns = %s" % repr(EXISTING_MODELS.columns.to_list()))
    dp_ids = list(EXISTING_MODELS['dp_id']) if len(EXISTING_MODELS) else []
    proc_dps = wp.DataProduct.select(wp.si.DataProduct.id.not_in(dp_ids),
                                     dpowner_id=wp.ThisJob.config_id, group='proc', data_type='Model')
    # filenames = [proc_dp.filename for proc_dp in proc_dps]
    # missing = np.array([os.path.splitext(name)[1] == '.csv' for name in filenames])
    # if len(EXISTING_MODELS):
    #     missing &= ~np.in1d(filenames, list(EXISTING_MODELS['name']))
    # EXISTING_MODELS = pd.concat([EXISTING_MODELS, load_models(np.array(proc_dps)[missing])])
    _temp = load_models(proc_dps)
    wp.ThisJob.logprint("CONCATENATING DATAFRAME:\n %s" % repr(_temp))
    EXISTING_MODELS = pd.concat([EXISTING_MODELS, _temp])
    return EXISTING_MODELS


def return_models():
    global EXISTING_MODELS
    return EXISTING_MODELS


def delete_cached_models():
    model_dps = wp.DataProduct.select(dpowner_id=wp.ThisJob.config_id, group='proc', data_type='Model')
    for model_dp in model_dps:
        wp.ThisJob.logprint("PROCESSING DELETION OF %s DP_ID %d" % (model_dp.filename, model_dp.dp_id))
        try:
            if wait_model_dp_ready(model_dp) is None:
                if model_dp.options['readings'] == LEN_EVENTPOOL and model_dp.options['cached']:
                    model_dp.delete()
        except KeyError:
            model_dp.delete()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
