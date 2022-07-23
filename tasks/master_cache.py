#!/usr/bin/env python
import shutil
import time
import subprocess
import wpipe as wp

if __name__ == '__main__':
    from ModelCaching import update_models, compare_model_dps, create_cache_dp, create_backup_dp, create_repacked_dp, delete_cached_models


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='start_cache', value='*')


CACHE_SLEEP = 60
REBASE_CYCLE = 3600


def check_for_stop():
    try:
        return wp.ThisEvent.options['stop_cache']
    except KeyError:
        return False


def backup_and_repack(cache_dp, backup_dp):
    shutil.copy2(cache_dp.path, backup_dp.path)
    repacked_dp = create_repacked_dp()
    subprocess.call(['ptrepack' , cache_dp.path, repacked_dp.path])
    shutil.copy2(repacked_dp.path ,cache_dp.path)
    repacked_dp.delete()
    if not compare_model_dps(backup_dp, cache_dp):
        raise RuntimeError("Repacking cache ran into an issue")


def main_loop():
    cache_dp = create_cache_dp()
    backup_dp = create_backup_dp()
    last_repacking = time.time()
    last_caching = last_repacking - CACHE_SLEEP
    while not check_for_stop():
        time.sleep((lambda x: (abs(x)+x)/2)(last_caching+CACHE_SLEEP-time.time()))
        wp.ThisEvent.options['currently_caching'] = True
        if time.time() > last_repacking + REBASE_CYCLE:
            backup_and_repack(cache_dp, backup_dp)
            last_repacking = time.time()
        last_caching = time.time()
        models = update_models(cache_dp_to_update=cache_dp)
        wp.ThisEvent.options['currently_caching'] = False
        delete_cached_models()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_loop()
