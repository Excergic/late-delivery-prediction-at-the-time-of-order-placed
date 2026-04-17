"""
Tests for drift detection logic.

Tests the alert threshold logic and drift summary parsing in isolation —
no Evidently, no ZenML, no CSV needed.
"""

import pytest


# ---------------------------------------------------------------------------
# Alert threshold logic (mirrors run_drift_detection step logic)
# ---------------------------------------------------------------------------

def _check_drift_alert(drifted_share: float, warning: float, critical: float) -> str:
    """Extracted alert logic from run_drift_detection for unit testing."""
    if drifted_share >= critical:
        return "critical"
    elif drifted_share >= warning:
        return "warning"
    return "ok"


def test_no_drift_returns_ok():
    assert _check_drift_alert(0.10, warning=0.30, critical=0.50) == "ok"


def test_drift_at_warning_threshold():
    assert _check_drift_alert(0.30, warning=0.30, critical=0.50) == "warning"


def test_drift_above_warning_below_critical():
    assert _check_drift_alert(0.45, warning=0.30, critical=0.50) == "warning"


def test_drift_at_critical_threshold():
    assert _check_drift_alert(0.50, warning=0.30, critical=0.50) == "critical"


def test_drift_above_critical():
    assert _check_drift_alert(0.80, warning=0.30, critical=0.50) == "critical"


def test_zero_drift():
    assert _check_drift_alert(0.0, warning=0.30, critical=0.50) == "ok"


def test_full_drift():
    assert _check_drift_alert(1.0, warning=0.30, critical=0.50) == "critical"


# ---------------------------------------------------------------------------
# Drift summary structure
# ---------------------------------------------------------------------------

def _build_drift_summary(drifted_features: list, total: int) -> dict:
    """Build the same summary dict structure as run_drift_detection."""
    return {
        "total_features": total,
        "drifted_count": len(drifted_features),
        "drifted_share": round(len(drifted_features) / total, 4) if total > 0 else 0.0,
        "drifted_features": drifted_features,
        "status": _check_drift_alert(
            len(drifted_features) / total if total > 0 else 0.0,
            warning=0.30,
            critical=0.50,
        ),
    }


def test_summary_drifted_share_calculation():
    summary = _build_drift_summary(["feature_a", "feature_b"], total=10)
    assert summary["drifted_share"] == 0.2
    assert summary["drifted_count"] == 2


def test_summary_status_propagates_correctly():
    ok_summary = _build_drift_summary(["f1"], total=10)       # 10% drifted
    warn_summary = _build_drift_summary(["f1", "f2", "f3"], total=10)   # 30%
    crit_summary = _build_drift_summary(["f%d" % i for i in range(6)], total=10)  # 60%

    assert ok_summary["status"] == "ok"
    assert warn_summary["status"] == "warning"
    assert crit_summary["status"] == "critical"


def test_summary_no_drift():
    summary = _build_drift_summary([], total=23)
    assert summary["drifted_count"] == 0
    assert summary["drifted_share"] == 0.0
    assert summary["status"] == "ok"
