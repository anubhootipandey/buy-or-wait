"""
Stage 2: deterministic financial-state reconstruction + 90-day forecast.

Architecture: data -> normalized financial state -> deterministic forecast.

AI understands messy evidence later (Stage 3). This package does the
financial math and enforces the reconciliation/recurrence/forecast rules
in pure, deterministic Python stdlib code - no Gemini, no image/message
interpretation, no affordability/planning decisions.
"""

from .state import CashEvent, CashDirection, EventSource, ReconciledState
from .reconciliation import reconcile_user_events
from .recurrence import RecurringSeries, SeriesKey, detect_recurring_series, project_series
from .forecast import ForecastResult, ForecastCheckpoint, build_forecast, FORECAST_DAYS

__all__ = [
    "CashEvent",
    "CashDirection",
    "EventSource",
    "ReconciledState",
    "reconcile_user_events",
    "RecurringSeries",
    "SeriesKey",
    "detect_recurring_series",
    "project_series",
    "ForecastResult",
    "ForecastCheckpoint",
    "build_forecast",
    "FORECAST_DAYS",
]
