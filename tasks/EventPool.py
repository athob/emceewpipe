#!/usr/bin/env python
import datetime
import numpy as np
import wpipe as wp

try:
    from ModelCaching import EXISTING_MODELS, update_models
except ImportError:
    pass


def convert_walltime(walltime):  # TODO: includes days?
    if walltime is not None:
        return datetime.timedelta(**dict(zip(['hours', 'minutes', 'seconds'], map(float, walltime.split(':')))))


DEFAULT_WALLTIME_DICT = {'': None, 'pbs': wp.scheduler.PbsScheduler.DEFAULT_WALLTIME}


class EventPool:
    def __init__(self, job, pool_length, *args, **kwargs):
        self._job = job
        d = repr(len(repr(pool_length)))
        self._events = list(map(lambda n: job.child_event(*args, tag=('PoolEvent#%0' + d + 'd') % n,
                                                          **kwargs), range(pool_length)))
        self._empty_event_jobs = list(map(lambda event: len(event.fired_jobs) == 0, self._events))
        _options = kwargs.get('options', {'submission_type': ''})
        self._submission_type = kwargs['options'].get('submission_type', '')
        if 'walltime' not in _options.keys():
            _options['walltime'] = DEFAULT_WALLTIME_DICT[self._submission_type]
        self._walltime = convert_walltime(_options['walltime'])
        self._initialize_events()

    def _initialize_events(self):
        for event in self._events:
            event.options['new_log_prob'] = True
            event.options['new_theta'] = False
            event.options['current_dpid'] = None

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
            while not (self._events[n].options['new_log_prob']) if (n in pool_indexes.keys()) else True:
                # management to this loop to restart expired events
                if self._walltime is not None:
                    _event = self._events[n]
                    _jobs = _event.fired_jobs
                    if len(_jobs) if self._empty_event_jobs[n] else True:
                        self._empty_event_jobs[n] = False
                        _job = _jobs[-1]
                        if _job.endtime is None:
                            _starttime = _job.starttime
                            if _starttime is not None:
                                if datetime.datetime.utcnow() - _starttime > self._walltime:
                                    _current_dpid = _event.options['current_dpid']
                                    if _current_dpid is not None:
                                        wp.DataProduct(int(_current_dpid)).delete()
                                    _job.expire()
                                    _event.options['new_log_prob'] = False
                                    _event.options['new_theta'] = True
                                    _event.options['current_dpid'] = None
                                    _event.fire()
                n += 1
                n %= pool_length
            self._events[n].options['new_log_prob'] = False
            # check if available event should have a result and get it
            if pool_indexes[n] >= 0:
                log_prob = self._get_log_prob(self._events[n])
                if log_prob == 'MinusInfinity':
                    log_prob = -np.inf
                output[pool_indexes[n]] = log_prob
                wp.ThisJob.logprint('GRABBING\t'+repr(n)+'\t'+repr(output[pool_indexes[n]]))
            # send element to event if more left, remove from pool_indexes otherwise
            if m < struct_length:
                self._send_theta(self._events[n], func(struct[m]))
                wp.ThisJob.logprint('SENDING\t'+repr(m))
                pool_indexes[n] = m
                m += 1
                n += 1
                n %= pool_length
            else:
                wp.ThisJob.logprint('DELETING\t'+repr(n))
                del pool_indexes[n]
                self._events[n].options['new_log_prob'] = True
        # wp.ThisJob.logprint('\nRETURN\n'+repr(output))
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
