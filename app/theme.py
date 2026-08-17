"""Palette and business-facing labels for the dashboard.

Single source of truth for colour and wording: the figures, the KPI cards and
the customer table all read from here, so a strategy keeps the same colour and
the same name wherever it appears.
"""

BG = "#0E1117"
CARD_BG = "#1A1F2E"
ACCENT = "#2DD4BF"  # teal - ranking by value, "our method" under a flat cost
ACCENT_BRIGHT = "#7DF9E4"  # lighter teal - the knapsack optimum
BASELINE = "#5B8FF9"  # blue - the industry-standard risk baseline
ALERT = "#F87171"  # red - break-even, value destruction
NEUTRAL = "#8B92A5"
TEXT = "#E5E7EB"

SIMULATED = "#FBBF24"  # amber - anything driven by an assumption, not data

STRATEGY_LABELS = {
    "random": "At random",
    "by_churn": "By churn risk",
    "by_priority": "By risk × value",
    "optimal": "Optimal allocation",
    # Only drawn in simulation mode, as the "what it was before" ghost line.
    "optimal_reference": "Optimal (uniform effectiveness, reference)",
}

# `by_priority` and `optimal` sit in the same teal family because each is "our
# method" in its own cost regime: ranking by value when contact costs are flat,
# solving the knapsack when they are not. They only ever share a chart on the
# channel-cost tab, where the brightness gap plus a dashed line separates them.
STRATEGY_COLORS = {
    "random": NEUTRAL,
    "by_churn": BASELINE,
    "by_priority": ACCENT,
    "optimal": ACCENT_BRIGHT,
    "optimal_reference": NEUTRAL,
}

# Channel prices behind `channel_cost_from_contract`, restated here for display.
# The strategy module owns the policy; this is only how it is shown.
CHANNEL_COST_LABELS = {
    "Month-to-month": 40.0,
    "One year": 15.0,
    "Two year": 3.0,
}
