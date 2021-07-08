#!/usr/bin/env python
import time
import wpipe as wp
import numpy as np
import pandas as pd
from scipy import linalg, spatial

if __name__ == '__main__':
    from ModelCaching import update_models, create_cache_dp, delete_cached_models


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='start_cache', value='*')


CACHE_SLEEP = 60


def read_cat_and_cache(cache_dp):
    cache_dp.options['ready'] = False
    models = update_models()
    models.to_csv(cache_dp.path)
    cache_dp.options['ready'] = True
    return


def check_for_stop():
    try:
        return wp.ThisEvent.options['stop_cache']
    except KeyError:
        return False


def main_loop():
    cache_dp = create_cache_dp()
    while not check_for_stop():
        time.sleep(CACHE_SLEEP)
        read_cat_and_cache(cache_dp)
        delete_cached_models()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_loop()
