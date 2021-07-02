#!/usr/bin/env python
import time
import tenacity as tn
import wpipe as wp
import pandas as pd


EXISTING_MODELS = pd.DataFrame()


def wait_model_dp_ready(model_dp):
    for retry in tn.Retrying(retry=tn.retry_if_exception_type(KeyError), wait=tn.wait_random()):
        with retry:
            while not model_dp.options['ready']:
                time.sleep(1)


def read_model_dp(model_dp):
    model = pd.read_csv(model_dp.path).set_index(['a0', 'a1'])
    model_dp.options['readings'] += 1
    return model


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
