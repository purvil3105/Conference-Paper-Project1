"""Resource feasibility: budget enforced, no free unlimited Msg3 service."""
import numpy as np
import pytest

from nbiot_rach.config import Config
from nbiot_rach.resource import ResourceCalendar
from nbiot_rach.actions import ActionRegistry


def test_over_budget_reservation_rejected():
    cfg = Config()
    cal = ResourceCalendar(cfg)
    with pytest.raises(ValueError):
        cal.reset_window(nprach_reserved_units=cal.total_units + 1.0)


def test_msg3_grants_bounded_by_data_budget():
    cfg = Config(msg3_resource_units=100.0)
    cal = ResourceCalendar(cfg)
    cal.reset_window(nprach_reserved_units=0.0)
    budget = cal.total_units
    max_grants = int(budget // 100.0)
    granted = sum(1 for _ in range(max_grants + 50) if cal.try_msg3_grant())
    assert granted == max_grants
    assert cal.data_used <= cal.total_units + 1e-9


def test_feasible_mask_respects_budget():
    cfg = Config()
    reg = ActionRegistry(cfg)
    mask = reg.feasible_mask()
    assert np.all(reg.resource_units[mask] <= reg.resource_budget_units() + 1e-9)
    # a pathological preamble duration makes the largest NPRACH config infeasible
    cfg2 = Config(preamble_duration_ms=100.0)
    reg2 = ActionRegistry(cfg2)
    assert not reg2.feasible_mask().all()


def test_resource_norm_in_unit_interval():
    cfg = Config()
    cal = ResourceCalendar(cfg)
    cal.reset_window(nprach_reserved_units=2000.0)
    cal.try_msg3_grant()
    assert 0.0 <= cal.r_norm() <= 1.0
