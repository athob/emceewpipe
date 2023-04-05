#! /usr/bin/env python
import wpipe as wp


def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='__init__', value='*')


if __name__ == '__main__':
    my_pipe = wp.Pipeline()
    my_job = wp.Job()
    my_job.logprint("Preparing targets")
    for my_input in my_pipe.inputs:
        for my_dp in my_input.rawdataproducts:
            my_target = my_input.target(name=my_dp.filesplitext[0])
            for my_conf in my_target.configurations:
                my_job.logprint("Starting target "+my_target.name+" config "+my_conf.name)
                my_job.child_event('run_target', options={'config_id': my_conf.config_id}).fire()
