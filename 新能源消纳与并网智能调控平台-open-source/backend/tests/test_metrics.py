from app.services.metrics import calculate_consumption


def test_consumption_metrics_use_available_inputs():
    metrics = calculate_consumption(80.0, 100.0, 90.0, 1)
    assert metrics.utilization_rate == 80 / 90
    assert metrics.curtailed_mwh == 20.0
    assert metrics.quality == "complete"


def test_consumption_metrics_preserve_missing_values():
    metrics = calculate_consumption(2.0, None, None, None)
    assert metrics.utilization_rate is None
    assert metrics.curtailed_mwh is None
    assert metrics.quality == "insufficient"
