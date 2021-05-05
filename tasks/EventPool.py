#!/usr/bin/env python
import wpipe as wp
import numpy as np


class EventPool:
    def __init__(self, job, pool_length, *args, **kwargs):
        self._job = job
        d = repr(len(repr(pool_length)))
        self._events = list(map(lambda n: job.child_event(*args, tag=('PoolEvent#%0' + d + 'd') % n,
                                                          **kwargs), range(pool_length)))
        self._initialize_events()

    def _initialize_events(self):
        for event in self._events:
            event.options['new_log_prob'] = True
            event.options['new_theta'] = False

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
            # wp.ThisJob.logprint('\nSTRUCT\t'+repr(m)+'\t'+repr(pool_indexes))
            # looping around the pool to find available event
            # TODO: add management to this loop to restart forced closed events
            while not(self._events[n].options['new_log_prob']) if (n in pool_indexes.keys()) else True:
                n += 1
                n %= pool_length
            self._events[n].options['new_log_prob'] = False
            # check if available event should have a result and get it
            if pool_indexes[n] >= 0:
                log_prob = self._get_log_prob(self._events[n])
                if log_prob == 'MinusInfinity':
                    log_prob = -np.inf
                output[pool_indexes[n]] = log_prob
                # wp.ThisJob.logprint('\nGRABBING\t'+repr(n)+'\t'+repr(output[pool_indexes[n]]))
            # send element to event if more left, remove from pool_indexes otherwise
            if m < struct_length:
                self._send_theta(self._events[n], func(struct[m]))
                # wp.ThisJob.logprint('\nSENDING\t'+repr(m))
                pool_indexes[n] = m
                m += 1
                n += 1
                n %= pool_length
            else:
                # wp.ThisJob.logprint('\nDELETING\t'+repr(n))
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
