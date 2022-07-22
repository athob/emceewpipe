#!/usr/bin/env python
import time
import wpipe as wp

if __name__ == '__main__':
    from ModelCaching import update_models, create_cache_dp, delete_cached_models


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='start_cache', value='*')


CACHE_SLEEP = 60


def read_cat_and_cache(cache_dp):
    wp.ThisEvent.options['currently_caching'] = True
    models = update_models(cache_dp_to_update=cache_dp)
    wp.ThisEvent.options['currently_caching'] = False
    return


def check_for_stop():
    try:
        return wp.ThisEvent.options['stop_cache']
    except KeyError:
        return False


def main_loop():
    cache_dp = create_cache_dp()
    temp_time = time.time()-CACHE_SLEEP
    while not check_for_stop():
        time.sleep((lambda x: (abs(x)+x)/2)(temp_time+CACHE_SLEEP-time.time()))
        temp_time = time.time()
        read_cat_and_cache(cache_dp)
        delete_cached_models()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_loop()
