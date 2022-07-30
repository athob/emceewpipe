#!/usr/bin/env python
import os
import sys
import gc
import time
import tenacity as tn
import wpipe as wp
import numpy as np
import pandas as pd
import tables

PARENT_IS_MASTER_CACHE = os.path.basename(sys.argv[0]) == 'master_cache.py'
try:
    LEN_EVENTPOOL = int(wp.ThisJob.config.parameters['len_eventpool'])
except AttributeError:
    LEN_EVENTPOOL = None

DATA_EXT = '.h5'


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
            retry=(tn.retry_if_exception_type(FileNotFoundError) | tn.retry_if_exception_type(tables.exceptions.HDF5ExtError)),
            after=lambda retry_state:
            wp.ThisJob.logprint("Failed first reading attempt of %s; entering retrying loop" % model_dp.path)
            if retry_state.attempt_number == 1 else None,
            wait=tn.wait_random()):
        with retry:
            if DATA_EXT == '.csv':
                temp = pd.read_csv(model_dp.path)
            elif DATA_EXT == '.h5':
                with pd.HDFStore(model_dp.path, 'r') as HDF5:
                    temp = HDF5['data']
            else:
                raise ValueError("Wrong DATA_EXT")
            wp.ThisJob.logprint("Succesfully read %s at attempt #%d" % (model_dp.path,
                                                                        retry.retry_state.attempt_number))
    if not temp.empty:
        if DATA_EXT == '.csv':
            model = temp.set_index([tag for tag in temp.columns if isinstance(tag, str) if tag[:1] == 'a'])
        else:
            model = temp
    else:
        model = pd.DataFrame()
    if model_dp.data_type == 'Model':
        if PARENT_IS_MASTER_CACHE:
            model_dp.options['cached'] = True
        else:
            model_dp.options['readings'] += 1
    return model


def compare_model_dps(model_dp1, model_dp2):
    model1 = read_model_dp(model_dp1)
    model2 = read_model_dp(model_dp2)
    return model1.equals(model2)


def create_model_dp(args):
    model_dp = wp.ThisJob.config.dataproduct(filename=('M' + len(args) * '_%.10e' + DATA_EXT) % args,
                                             relativepath=wp.ThisJob.config.procpath,
                                             group='proc',
                                             data_type='Model',
                                             options={'ready': False, 'readings': 0, 'cached': False})
    return model_dp


def create_cache_dp():
    cache_dp = wp.ThisJob.config.dataproduct(filename="Cache" + DATA_EXT,
                                             relativepath=wp.ThisJob.config.procpath,
                                             group='proc',
                                             data_type='Cache',
                                             options={'ready': False})
    cache_dp.options['ready'] = os.path.exists(cache_dp.path)
    return cache_dp


def create_backup_dp():
    backup_dp = wp.ThisJob.config.dataproduct(filename="Cache_backup" + DATA_EXT,
                                              relativepath=wp.ThisJob.config.procpath,
                                              group='proc',
                                              data_type='Backup')
    return backup_dp


def create_repacked_dp():
    repacked_dp = wp.ThisJob.config.dataproduct(filename="Cache_repacked" + DATA_EXT,
                                                relativepath=wp.ThisJob.config.procpath,
                                                group='proc',
                                                data_type='Repack')
    return repacked_dp


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
    except (OSError, KeyError, IndexError, AttributeError):
        EXISTING_MODELS = pd.DataFrame()


def load_models(model_dps):
    # wp.ThisJob.logprint('\nLOAD_MODELS')
    temp = [read_model_dp(model_dp) for model_dp in model_dps if wait_model_dp_ready(model_dp) is None]
    if len(temp):
        return pd.concat(temp)
    else:
        return pd.DataFrame()


def append_to_dp(models, dp):
    path = dp.path
    while os.path.exists(path) and not dp.options['ready']:
        time.sleep(1)
    dp.options['ready'] = False
    for retry in tn.Retrying(
            retry=(tn.retry_if_exception_type(FileNotFoundError) | tn.retry_if_exception_type(tables.exceptions.HDF5ExtError)),
            after=lambda retry_state:
            wp.ThisJob.logprint("Failed first reading attempt of %s; entering retrying loop" % dp.path)
            if retry_state.attempt_number == 1 else None,
            wait=tn.wait_random()):
        with retry:
            if DATA_EXT == '.csv':
                models.to_csv(path)
            elif DATA_EXT == '.h5':
                with pd.HDFStore(dp.path) as HDF5:
                    HDF5.append('data', models)
            else:
                raise ValueError("Wrong DATA_EXT")
    dp.options['ready'] = True


def update_models(cache_dp_to_update=None):
    global EXISTING_MODELS
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    wp.ThisJob.logprint(f"len(INSTANCES) = {len(wp.si.INSTANCES)}")
    wp.ThisJob.logprint(' '.join([f"{n}x{t[:5]}" for t,n in zip(*np.unique(list(map(lambda I: type(I).__name__,wp.si.INSTANCES)), return_counts=True))]))
    wp.ThisJob.logprint(f"DataProduct.__cache__.shape = {wp.DataProduct.__cache__.shape}")
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    wp.ThisJob.logprint("UPDATING MODELS: current EXISTING_MODELS.shape = %s" % repr(EXISTING_MODELS.shape))
    # wp.ThisJob.logprint("                         EXISTING_MODELS.columns = %s" % repr(EXISTING_MODELS.columns.to_list()))
    dp_ids = list(EXISTING_MODELS['dp_id']) if len(EXISTING_MODELS) else []
    proc_dps = wp.DataProduct.select(wp.si.DataProduct.id.not_in(dp_ids),
                                     dpowner_id=wp.ThisJob.config_id, group='proc', data_type='Model')
    _temp = load_models(proc_dps)
    wp.ThisJob.logprint("CONCATENATING DATAFRAME.shape = %s" % repr(_temp.shape))
    EXISTING_MODELS = pd.concat([EXISTING_MODELS, _temp])
    if cache_dp_to_update is not None and not _temp.empty:
        append_to_dp([_temp, EXISTING_MODELS][DATA_EXT == '.csv'], cache_dp_to_update)
    # ----------------------------------------------------------------------------------
    # CLEAR CACHE, bit of a clumsy fix there...-----------------------------------------
    # ----------------------------------------------------------------------------------
    if not _temp.empty:
        del proc_dps
        wp.ThisJob.logprint(f"Before clear len(INSTANCES) = {len(wp.si.INSTANCES)}")
        wp.ThisJob.logprint(' '.join(["Before clear"]+[f"{n}x{t[:5]}" for t,n in zip(*np.unique(list(map(lambda I: type(I).__name__,wp.si.INSTANCES)), return_counts=True))]))
        wp.ThisJob.logprint(f"Before clear DataProduct.__cache__.shape = {wp.DataProduct.__cache__.shape}")
        _tmp_opt = wp.Option.select(wp.si.Option.optowner_id.in_(_temp.dp_id))
        _tmp_opt = wp.Option.__cache__.query("optowner_id in @_temp.dp_id")
        _temp = wp.DataProduct.__cache__.query('dp_id in @_temp.dp_id')
        # _temp = wp.DataProduct.__cache__.groupby('group').get_group('proc').query("filename != 'Cache.h5'")
        # _options = pd.DataFrame(
        #     [tuple(opt._sa_instance_state.dict[k] for k in ['optowner_id', 'id'])+(opt,)
        #      for opt in wp.si.INSTANCES if 'id' in opt._sa_instance_state.dict.keys() and isinstance(opt, wp.si.Option)],
        #      columns=['own_id', 'id', 'opt']).query("own_id in @_temp.dp_id").drop(columns='own_id')
        # wp.si.INSTANCES = list(set(wp.si.INSTANCES)-set(_options.opt)-{opt._option for opt in _tmp_opt.option})
        # wp.si.INSTANCES = list(set(wp.si.INSTANCES)-{dp._dataproduct for dp in _temp.dataproduct})
        wp.si.INSTANCES = list(set(wp.si.INSTANCES)-{dp._dataproduct for dp in _temp.dataproduct}-{opt._option for opt in _tmp_opt.option})
        wp.Option.__cache__.drop(_tmp_opt.index, inplace=True)
        # wp.Option.__cache__.drop(wp.Option.__cache__.query("option_id in @_options.id").index, inplace=True)
        wp.DataProduct.__cache__.drop(_temp.index, inplace=True)
        # del _tmp_opt, _options, _temp
        del _tmp_opt, _temp
        gc.collect()
        wp.ThisJob.logprint(f"After clear len(INSTANCES) = {len(wp.si.INSTANCES)}")
        wp.ThisJob.logprint(' '.join(["After clear"]+[f"{n}x{t[:5]}" for t,n in zip(*np.unique(list(map(lambda I: type(I).__name__,wp.si.INSTANCES)), return_counts=True))]))
        wp.ThisJob.logprint(f"After clear DataProduct.__cache__.shape = {wp.DataProduct.__cache__.shape}")
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
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
