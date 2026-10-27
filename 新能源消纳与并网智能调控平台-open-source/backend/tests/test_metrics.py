from app.services.metrics import calculate_consumption


def test_consumption_metrics_use_available_power():
    metrics = calculate_consumption(80.0, 100.0, 90.0, 1)
    assert metrics.utilization_rate == 80 / 90
    assert metrics.curtailed_mw == 10.0
    assert metrics.curtailment_rate == 10 / 90
    assert metrics.quality == "complete"


def test_consumption_metrics_preserve_missing_values():
    metrics = calculate_consumption(2.0, None, None, None)
    assert metrics.utilization_rate is None
    assert metrics.curtailed_mw is None
    assert metrics.quality == "insufficient"


def test_consumption_metrics_do_not_require_theory_power():
    metrics = calculate_consumption(80.0, None, 100.0, 0)
    assert metrics.utilization_rate == 0.8
    assert metrics.curtailed_mw == 20.0
    assert metrics.quality == "partial"
