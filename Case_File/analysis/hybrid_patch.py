"""Drop these two functions into Logic.py, then call them in App.py in place of
compute_hybrid_demand(consumer_demand, historical_forecast, alpha, scenario).

Implements the revised hybrid of the paper (Section 3.3):
    D_hybrid = D_h * (1 + alpha * g_c) * scenario
where g_c is the survey-implied relative change in demand.
"""
import pandas as pd

# Class midpoints for the survey item "expected purchasing volume change" (assumptions).
VOLUME_CHANGE_MIDPOINTS = {
    "Significant increase (>20% growth)": 0.25,
    "Moderate increase (1\u201320% growth)": 0.10,
    "Remain stable / unchanged": 0.0,
    "Decrease": -0.10,
}

def compute_consumer_growth(survey_df: pd.DataFrame, column: str = "volume_change",
                            weights: str = None, midpoints: dict = None) -> float:
    """Survey-implied relative change g_c (e.g. 0.107 = +10.7%)."""
    midpoints = midpoints or VOLUME_CHANGE_MIDPOINTS
    g = survey_df[column].map(midpoints)
    if weights and weights in survey_df:
        w = survey_df[weights].astype(float)
        return float((g * w).sum() / w.sum())
    return float(g.mean())

def compute_hybrid_demand_v2(historical_forecast: float, growth: float,
                             alpha: float, scenario: float) -> float:
    """D_hybrid = D_h * (1 + alpha * g_c) * scenario  (alpha clipped to [0, 1])."""
    alpha = float(min(1.0, max(0.0, alpha)))
    return max(0.0, float(historical_forecast) * (1.0 + alpha * float(growth)) * float(scenario))
