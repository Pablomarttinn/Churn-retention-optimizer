"""Palette and business-facing labels for the dashboard.

Single source of truth for colour and wording: the figures, the KPI cards and
the customer table all read from here, so a strategy keeps the same colour and
the same name wherever it appears.
"""

BG = "#0E1117"
CARD_BG = "#1A1F2E"
ACCENT = "#2DD4BF"  # teal - the method this project proposes
BASELINE = "#5B8FF9"  # blue - the industry-standard risk baseline
ALERT = "#F87171"  # red - break-even, value destruction
NEUTRAL = "#8B92A5"
TEXT = "#E5E7EB"

STRATEGY_LABELS = {
    "random": "At random",
    "by_churn": "By churn risk",
    "by_priority": "By risk × value",
    "optimal": "Optimal allocation",
}

# `by_priority` and `optimal` share the accent because each is "our method" in
# its own cost regime: ranking by value when contact costs are flat, solving the
# knapsack when they are not. They are never the recommended line at once.
STRATEGY_COLORS = {
    "random": NEUTRAL,
    "by_churn": BASELINE,
    "by_priority": ACCENT,
    "optimal": ACCENT,
}
