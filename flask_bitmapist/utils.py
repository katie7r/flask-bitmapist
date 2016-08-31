# -*- coding: utf-8 -*-
"""
    flask_bitmapist.utils
    ~~~~~~~~~~~~~~~~~~~~~
    Generic utility functions.

    :copyright: (c) 2016 by Cuttlesoft, LLC.
    :license: MIT, see LICENSE for more details.

"""

from datetime import datetime
from urlparse import urlparse

from dateutil.relativedelta import relativedelta

from bitmapist import (DayEvents, WeekEvents, MonthEvents, YearEvents,
                       BitOpAnd, BitOpOr, BitOpXor, delete_runtime_bitop_keys)


def _get_redis_connection(redis_url=None):
    url = urlparse(redis_url)
    return url.hostname, url.port


def get_event_data(event_name, time_group='days', now=None, system='default'):
    """
    Get the data for a single event at a single event in time.

    :param str event_name: Name of event for retrieval
    :param str time_group: Time scale by which to group results; can be `days`,
                           `weeks`, `months`, `years`
    :param datetime now: Time point at which to get event data (defaults to
                         current time if None)
    :param str system: Which bitmapist should be used
    :returns: Bitmapist events collection
    """
    now = now or datetime.utcnow()
    return _events_fn(time_group)(event_name, now, system)


class Cohort():
    """
    A class for handling and simplifying cohort creation and retrieval.

    :param str base_event_name: Name of main event to act as base event for cohort
    :param str next_event_name: Name of next event to build cohort on base event
    :param list chaining_events: List of events which will be chained to furhter
                                 filter cohort; each is a dict of 'name' and 'op'
                                 with the name of the event and the operator
                                 to use (i.e., 'and' or 'or'), repectively
    :param str time_group: Time scale by which to group results; valid values are
                           'days', 'weeks', 'months', and 'years' (TODO: 'hours')
    :param datetime now: Time point at which to get events (current time if `None`)
    :param str system: Which Bitmapist should be used
    """

    def __init__(self, base_event_name, next_event_name, chaining_events=[],
                 time_group='days', now=None, system='default'):
        # cohort events
        self.base_event_name = base_event_name
        self.next_event_name = next_event_name
        self.chaining_events = chaining_events

        # cohort settings
        self.time_group = time_group
        self.get_events = _events_fn(self.time_group)
        self._now = now or datetime.utcnow()
        self._sys = system

        # self.data  # list of lists of counts of ids (standard)
        # self._raw  # list of lists of full sets of ids, not counts  # ?

        # self.dates  # list of datetimes for defining cohort

        # self.base_total  # total users for base event in cohort
        # self.row_totals  # list of total users (per row of results in cohort)
        # self.col_totals  # list of total users (per col of results in cohort)

        # self.averages    # list of averages (# users or percent per column)

    def generate(self, m=10, n=10, as_percent=False, with_replacement=False):
        """
        Generate the cohort with the provided options.

        :param int m: Number of rows of results (or, how many steps back from current time)
        :param int n: Number of cols of results (or, how many steps forward for each row/time)
        :param bool as_percent: Whether to return cohort data as percents (vs standard user counts)
        :param bool with_replacement: Whether more than one occurence of an event for a single user
                                      should be counted; e.g., if a user logged in multiple times,
                                      whether to include subsequent logins in the cohort events

        :returns: List of lists of ints (user counts) making up the cohort results
        """
        pass

    def get_dates(self):
        """
        Gets the event time points (based on ``time_group`` and ``now``) for defining the cohort.

        :returns: List of datetimes
        """
        pass

    def get_totals(self, cohort):
        """
        Calculate the base, row, and column totals for the cohort.

        :returns: Tuple of (int) total users for base event in cohort,
                           (list, length m) total users per row of results,
                           (list, length n) total users per col of results
        """
        pass

    def get_averages(self, cohort):
        """
        Calculate the average for each column of the cohort, either based on the
        user count or on the percent, if ``as_percent``.

        :returns: List of ints (with user counts) or floats (with percents) for column averages
        """
        pass

    def get_as_percent(self, cohort):
        """
        Converts cohort data to use percent values rather than user counts.

        :returns: List of lists of floats (percents) making up the cohort results
        """
        pass

    def _chain(self, anchor_event_name, events_to_chain):
        """
        Chain a set of events with an anchoring set of events.

        Note: ``OR`` operators will apply only to their direct predecessors (i.e.,
        ``A && B && C || D`` will be handled as ``A && B && (C || D)``, and
        ``A && B || C && D`` will be handled as ``A && (B || C) && D``).

        :param str anchor_event_name: Name of event to chain additional events to/with
        :param list events_to_chain: List of events which will be chained to/with the
                                     anchor event; each is a dict of 'name' and 'op'
                                     with the name of the event and the operator
                                     to use (i.e., 'and' or 'or'), repectively

        :returns: Bitmapist events collection
        """
        pass


def get_cohort(primary_event_name, secondary_event_name,
               additional_events=[], time_group='days',
               num_rows=10, num_cols=10, system='default',
               with_replacement=False):
    """
    Get the cohort data for multiple chained events at multiple points in time.

    :param str primary_event_name: Name of primary event for defining cohort
    :param str secondary_event_name: Name of secondary event for defining cohort
    :param list additional_events: List of additional events by which to filter
                                   cohort (e.g., ``[{'name': 'user:logged_in',
                                   'op': 'and'}]``)
    :param str time_group: Time scale by which to group results; can be `days`,
                           `weeks`, `months`, `years`
    :param int num_rows: How many results rows to get; corresponds to how far
                         back to get results from current time
    :param int num_cols: How many results cols to get; corresponds to how far
                         forward to get results from each time point
    :param str system: Which bitmapist should be used
    :param bool with_replacement: Whether more than one occurence of an event
                                  should be counted for a given user; e.g., if
                                  a user logged in multiple times, whether to
                                  include subsequent logins for the cohort
    :returns: Tuple of (list of lists of cohort results, list of dates for
              cohort, primary event total for each date)
    """

    cohort = []
    dates = []
    primary_event_totals = []  # for percents

    fn_get_events = _events_fn(time_group)

    # TIMES

    def increment_delta(t):
        return relativedelta(**{time_group: t})

    now = datetime.utcnow()
    # - 1 for deltas between time points (?)
    event_time = now - relativedelta(**{time_group: num_rows - 1})

    if time_group == 'months':
        event_time -= relativedelta(days=event_time.day - 1)  # (?)

    # COHORT

    for i in range(num_rows):
        # get results for each date interval from current time point for the row
        row = []
        primary_event = fn_get_events(primary_event_name, event_time, system)

        primary_total = len(primary_event)
        primary_event_totals.append(primary_total)

        dates.append(event_time)

        if not primary_total:
            row = [None] * num_cols
        else:
            for j in range(num_cols):
                # get results for each event chain for current incremented time
                incremented = event_time + increment_delta(j)

                if incremented > now:
                    # date in future; no events and no need to go through chain
                    combined_total = None

                else:
                    chained_events = chain_events(secondary_event_name,
                                                  additional_events,
                                                  incremented, time_group, system)

                    if chained_events:
                        combined_events = BitOpAnd(chained_events, primary_event)
                        combined_total = len(combined_events)

                        if not with_replacement:
                            primary_event = BitOpXor(primary_event, combined_events)

                    else:
                        combined_total = 0

                row.append(combined_total)

        cohort.append(row)
        event_time += increment_delta(1)

    # Clean up results of BitOps
    delete_runtime_bitop_keys()

    return cohort, dates, primary_event_totals


def chain_events(base_event_name, events_to_chain, now, time_group,
                 system='default'):
    """
    Chain additional events with a base set of events.

    Note: ``OR`` operators will apply only to their direct predecessors (i.e.,
    ``A && B && C || D`` will be handled as ``A && B && (C || D)``, and
    ``A && B || C && D`` will be handled as ``A && (B || C) && D``).

    :param str base_event_name: Name of event to chain additional events to/with
    :param list events_to_chain: List of additional event names to chain
                                 (e.g., ``[{'name': 'user:logged_in',
                                 'op': 'and'}]``)
    :param datetime now: Time point at which to get event data
    :param str time_group: Time scale by which to group results; can be `days`,
                           `weeks`, `months`, `years`
    :param str system: Which bitmapist should be used
    :returns: Bitmapist events collection
    """

    fn_get_events = _events_fn(time_group)
    base_event = fn_get_events(base_event_name, now, system)

    if not base_event.has_events_marked():
        return ''

    if events_to_chain:
        chain_events = []

        # for idx, event_to_chain in enumerate(events_to_chain):
        for event_to_chain in events_to_chain:
            event_name = event_to_chain.get('name')
            chain_event = fn_get_events(event_name, now, system)
            chain_events.append(chain_event)

        # Each OR should operate only on its immediate predecessor, e.g.,
        #     `A && B && C || D` should be handled as ~ `A && B && (C || D)`,
        #     and
        #     `A && B || C && D` should be handled as ~ `A && (B || C) && D`.
        op_or_indices = [idx for idx, e in enumerate(events_to_chain) if e['op'] == 'or']

        # Work backwards; least impact on operator combos + list indexing
        for idx in reversed(op_or_indices):
            # If first of events to chain, OR will just operate on base event
            if idx > 0:
                prev_event = chain_events[idx - 1]
                or_event = chain_events.pop(idx)

                # OR events should not be re-chained below
                events_to_chain.pop(idx)

                chain_events[idx - 1] = BitOpOr(prev_event, or_event)

        for idx, name_and_op in enumerate(events_to_chain):
            if name_and_op.get('op') == 'or':
                base_event = BitOpOr(base_event, chain_events[idx])
            else:
                base_event = BitOpAnd(base_event, chain_events[idx])

    return base_event


def _events_fn(time_group='days'):
    if 'day' in time_group:
        return _day_events_fn
    elif 'week' in time_group:
        return _week_events_fn
    elif 'month' in time_group:
        return _month_events_fn
    elif 'year' in time_group:
        return _year_events_fn


# PRIVATE methods: copied directly from Bitmapist because you can't import
# from bitmapist.cohort without also having mako for the cohort __init__

def _dispatch(key, cls, cls_args):
    # ignoring CUSTOM_HANDLERS
    return cls(key, *cls_args)


def _day_events_fn(key, date, system):
    cls = DayEvents
    cls_args = (date.year, date.month, date.day, system)
    return _dispatch(key, cls, cls_args)


def _week_events_fn(key, date, system):
    cls = WeekEvents
    cls_args = (date.year, date.isocalendar()[1], system)
    return _dispatch(key, cls, cls_args)


def _month_events_fn(key, date, system):
    cls = MonthEvents
    cls_args = (date.year, date.month, system)
    return _dispatch(key, cls, cls_args)


def _year_events_fn(key, date, system):
    cls = YearEvents
    cls_args = (date.year, system)
    return _dispatch(key, cls, cls_args)
