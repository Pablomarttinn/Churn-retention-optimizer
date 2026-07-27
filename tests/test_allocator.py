"""Tests for the retention budget allocator (0/1 knapsack)."""

import numpy as np
import pytest

from retention_optimizer.optimization.allocator import allocate, solve_knapsack


def test_infinite_budget_selects_positive_net():
    """With no binding budget, take every customer with positive net value."""
    net_value = np.array([10.0, -5.0, 30.0, -1.0, 8.0])
    cost = np.array([20.0, 20.0, 20.0, 20.0, 20.0])

    mask = solve_knapsack(net_value, cost, budget=1e9)

    assert np.array_equal(mask, net_value > 0)


def test_knapsack_beats_greedy_ratio():
    """The solver must beat a greedy net/cost ratio heuristic.

    Greedy by ratio picks customer 0 (ratio 1.0) and then has no room left,
    yielding 60. The optimum takes customers 1 and 2, spending the full 100
    for a value of 90. This is the test that proves we are not a greedy.
    """
    net_value = np.array([60.0, 45.0, 45.0])
    cost = np.array([60.0, 50.0, 50.0])

    mask = solve_knapsack(net_value, cost, budget=100.0)

    assert np.array_equal(mask, np.array([False, True, True]))
    assert net_value[mask].sum() == pytest.approx(90.0)


def test_constant_cost_collapses_to_top_k():
    """With a constant cost the knapsack degenerates to the top-k by net value."""
    net_value = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0])
    cost = np.full(10, 20.0)
    budget = 5 * 20.0  # room for exactly k = 5 customers

    mask = solve_knapsack(net_value, cost, budget)

    expected = np.zeros(10, dtype=bool)
    expected[[9, 2, 6, 4, 8]] = True  # net values 10, 9, 8, 7, 6
    assert np.array_equal(mask, expected)
    assert mask.sum() == 5


def test_allocate_returns_consistent_info():
    """The info dict must agree with the mask and respect the budget."""
    p = np.array([0.5, 0.3, 0.8, 0.1])
    m = np.array([70.0, 50.0, 90.0, 20.0])
    budget = 40.0  # at the default cost of 20, room for 2 customers

    mask, info = allocate(p, m, budget)

    assert info["spend"] == pytest.approx(20.0 * mask.sum())
    assert info["n_selected"] == int(mask.sum())
    assert info["spend"] <= budget
    assert info["expected_value"] >= 0
