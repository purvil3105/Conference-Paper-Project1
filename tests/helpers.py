"""Shared helpers for the test suite: hand-built traces and small runs."""
from __future__ import annotations

import numpy as np

from nbiot_rach.traffic import Trace, CLASS_PERIODIC, CLASS_EVENT
from nbiot_rach.timeutils import s_to_us


def custom_trace(meter_ids, arrivals_s, deadlines_s, classes=None,
                 n_meters=None, trace_seed=3001, name="custom") -> Trace:
    """Build an explicit trace from Python lists (sorted by arrival, meter)."""
    meter_id = np.asarray(meter_ids, dtype=np.int64)
    arrival_us = np.asarray([s_to_us(a) for a in arrivals_s], dtype=np.int64)
    deadline_us = np.asarray([s_to_us(d) for d in deadlines_s], dtype=np.int64)
    if classes is None:
        classes = [CLASS_PERIODIC] * len(meter_ids)
    traffic_class = np.asarray(classes, dtype=np.int8)
    n_meters = n_meters or (int(meter_id.max()) + 1)
    order = np.lexsort((meter_id, arrival_us))
    meter_id = meter_id[order]
    arrival_us = arrival_us[order]
    deadline_us = deadline_us[order]
    traffic_class = traffic_class[order]
    request_id = np.arange(meter_id.size, dtype=np.int64)
    return Trace(request_id, meter_id, arrival_us, deadline_us, traffic_class,
                 trace_seed, name, n_meters)
