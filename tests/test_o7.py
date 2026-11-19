"""Tests for the O7 synthetic model."""

import numpy as np
import pytest
from scipy.special import iv

from condgen_benchmark.data.synthetic import (
    DATA_CLASS_REGISTRY,
    O7Data,
    get_synthetic_data_class,
)


def test_o7_is_registered():
    assert DATA_CLASS_REGISTRY["O7"] is O7Data
    assert get_synthetic_data_class("O7") is O7Data


def test_o7_generate_shapes_without_x():
    Y, X = O7Data(seed=1).generate(50)
    assert Y.shape == (50, 1)
    assert X.shape == (50, 1)


def test_o7_generate_shapes_with_given_x():
    X_in = np.array([[0.1], [-0.2], [0.3]])
    Y, X = O7Data(seed=1).generate(7, X=X_in)
    assert Y.shape == (21, 1)
    np.testing.assert_array_equal(X, X_in)


def test_o7_is_deterministic_under_a_seed():
    Y1, X1 = O7Data(seed=7).generate(30)
    Y2, X2 = O7Data(seed=7).generate(30)
    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(X1, X2)
    Y3, _ = O7Data(seed=8).generate(30)
    assert not np.array_equal(Y1, Y3)


def test_o7_response_lies_in_the_exp_sin_range():
    Y, _ = O7Data(seed=2).generate(500)
    assert Y.min() >= np.exp(-1.0) - 1e-12
    assert Y.max() <= np.e + 1e-12


def test_o7_mean_std_matches_the_bessel_formula_at_fixed_inputs():
    model = O7Data(seed=0, c=20.0, K=20)
    X = np.array([0.0, 0.05, -0.1, 0.3])
    mean, std = model.mean_std(X)
    mu = 20.0 * X
    n = np.arange(1, 21, dtype=float)
    for j, m in enumerate(mu):
        c1 = iv(n, 1.0) * np.exp(-0.5 * n**2) * np.cos(n * (m - np.pi / 2))
        c2 = iv(n, 2.0) * np.exp(-0.5 * n**2) * np.cos(n * (m - np.pi / 2))
        m1 = iv(0, 1.0) + 2.0 * c1.sum()
        m2 = iv(0, 2.0) + 2.0 * c2.sum()
        assert np.isclose(mean[j], m1)
        assert np.isclose(std[j], np.sqrt(m2 - m1**2))


def test_o7_mean_std_agrees_with_monte_carlo():
    model = O7Data(seed=3)
    x = np.array([[0.12]])
    Y, _ = model.generate(40000, X=x)
    mean, std = model.mean_std(x)
    assert Y.mean() == pytest.approx(mean[0], abs=0.03)
    assert Y.std() == pytest.approx(std[0], abs=0.03)
