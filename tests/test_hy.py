"""CPU tests for the Hall--Yao loss, theta fit and conditional sampler."""

import numpy as np
import pytest
import torch

from condgen_benchmark.methods.hy.fit import fit_thetas
from condgen_benchmark.methods.hy.loss import HallYaoLoss
from condgen_benchmark.methods.hy.sampler import (
    PreparedConditionalSampler,
    energy_score_from_quantile,
    response_grid,
    sample_conditional,
)


@pytest.fixture(scope="module")
def problem():
    rng = np.random.RandomState(0)
    n, d = 40, 2
    X = rng.randn(n, d)
    Y = np.sin(X[:, 0]) + 0.3 * rng.randn(n)
    return X, Y


@pytest.fixture(scope="module")
def objective(problem):
    X, Y = problem
    return HallYaoLoss(X, Y, device="cpu", dtype=torch.float64)


def test_loss_objectives_are_finite_per_bandwidth(objective):
    theta = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]).double()
    values = objective.objectives(theta, [0.8, 1.0, 1.5])
    assert values.shape == (3,)
    assert torch.isfinite(values).all()
    assert (values >= 0).all()


def test_loss_rejects_wrong_theta_shape(objective):
    with pytest.raises(ValueError):
        objective.objectives(torch.ones(2, 3).double(), [1.0, 1.0])


def test_fit_thetas_shape_unit_norm_and_lower_objective(objective):
    bandwidths = [0.8, 1.2]
    seed = 3
    thetas, seconds, diagnostics = fit_thetas(
        objective, bandwidths, seed, max_epochs=40, patience=10
    )
    assert set(thetas) == set(bandwidths)
    assert seconds >= 0
    rng = np.random.RandomState(seed)
    start = rng.randn(objective.d)
    start /= np.linalg.norm(start)
    start_values = objective.objectives(
        torch.as_tensor(np.stack([start, start])), bandwidths
    )
    for k, h in enumerate(bandwidths):
        theta = thetas[h]
        assert theta.shape == (objective.d,)
        assert np.all(np.isfinite(theta))
        assert np.isclose(np.linalg.norm(theta), 1.0, atol=1e-3)
        assert diagnostics[k]["h"] == h
        assert diagnostics[k]["best_S_theta"] <= start_values[k].item()


def test_fit_thetas_is_deterministic_for_a_seed(objective):
    first, _, _ = fit_thetas(objective, [1.0], 5, max_epochs=10, patience=5)
    second, _, _ = fit_thetas(objective, [1.0], 5, max_epochs=10, patience=5)
    np.testing.assert_allclose(first[1.0], second[1.0])


@pytest.mark.parametrize(
    "kwargs",
    [{"bandwidths": []}, {"bandwidths": [-1.0]}, {"lr": 0.0}],
)
def test_fit_thetas_rejects_invalid_arguments(objective, kwargs):
    args = {"bandwidths": [1.0], "seed": 0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        fit_thetas(objective, **args)


def test_response_grid_spans_the_response(problem):
    _, Y = problem
    grid = response_grid(Y, 25)
    assert grid.shape == (25,)
    assert grid[0] == Y.min() and grid[-1] == Y.max()
    assert np.all(np.diff(grid) > 0)
    with pytest.raises(ValueError):
        response_grid(Y, 1)


def test_sampler_output_shape_and_range(problem):
    X, Y = problem
    grid = response_grid(Y, 50)
    theta = np.array([1.0, 0.0])
    samples = sample_conditional(
        X, Y, X[:6], theta, 1.0, 30, grid, rng=np.random.default_rng(0)
    )
    assert samples.shape == (6, 30)
    assert np.all(np.isfinite(samples))
    assert samples.min() >= grid[0] and samples.max() <= grid[-1]


def test_sampler_is_deterministic_given_uniforms(problem):
    X, Y = problem
    grid = response_grid(Y, 50)
    sampler = PreparedConditionalSampler(X, Y, X[:4], [1.0, 1.0], grid)
    u = np.random.default_rng(1).random((4, 20))
    first = sampler.sample(1.0, 20, uniforms=u)
    second = sampler.sample(1.0, 20, uniforms=u)
    np.testing.assert_array_equal(first, second)


def test_sampler_quantile_knots_form_a_monotone_cdf(problem):
    X, Y = problem
    grid = response_grid(Y, 50)
    sampler = PreparedConditionalSampler(X, Y, X[:5], [1.0, 0.5], grid)
    for i in range(5):
        knots = sampler._quantile_knots(i, 1.0, 1e-12)
        assert knots is not None
        probs, responses = knots
        assert probs[0] >= 0.0 and probs[-1] <= 1.0
        assert np.all(np.diff(probs) > 0)
        assert np.all(np.diff(responses) >= 0)


def test_sampler_is_monotone_in_the_uniform_draw(problem):
    X, Y = problem
    grid = response_grid(Y, 50)
    sampler = PreparedConditionalSampler(X, Y, X[:1], [1.0, 0.0], grid)
    u = np.linspace(0.0, 1.0, 40)[None, :]
    samples = sampler.sample(1.0, 40, uniforms=u)[0]
    assert np.all(np.diff(samples) >= 0)


def test_sampler_falls_back_to_the_mean_when_weights_vanish(problem):
    X, Y = problem
    grid = response_grid(Y, 50)
    far_query = np.array([[100.0, 100.0]])
    samples = sample_conditional(
        X, Y, far_query, [1.0, 0.0], 0.5, 5, grid, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(samples, Y.mean())


def test_sampler_rejects_invalid_inputs(problem):
    X, Y = problem
    grid = response_grid(Y, 10)
    with pytest.raises(ValueError):
        PreparedConditionalSampler(X, Y, X[:2], [0.0, 0.0], grid)
    with pytest.raises(ValueError):
        PreparedConditionalSampler(X, Y, X[:2], [1.0, 0.0], grid[::-1])
    sampler = PreparedConditionalSampler(X, Y, X[:2], [1.0, 0.0], grid)
    with pytest.raises(ValueError):
        sampler.sample(-1.0, 5)


def test_energy_score_of_a_uniform_quantile_matches_the_closed_form():
    probs = np.array([0.0, 1.0])
    responses = np.array([0.0, 1.0])
    # Uniform(0, 1): CRPS at y=0.5 is E|Z-y| - 0.5 E|Z-Z'| = 0.25 - 1/6.
    score = energy_score_from_quantile(probs, responses, 0.5)
    assert np.isclose(score, 0.25 - 1.0 / 6.0)


def test_sampler_energy_score_is_finite_and_nonnegative(problem):
    X, Y = problem
    grid = response_grid(Y, 50)
    sampler = PreparedConditionalSampler(X, Y, X[:5], [1.0, 0.0], grid)
    score = sampler.energy_score(1.0, Y[:5])
    assert np.isfinite(score) and score >= 0


def _skewed_linear_gaussian_data(seed, n=3000):
    """Y = 1 + 2 X0 + N(0, 0.2^2) with a skewed density of X0 on [-1, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    X[:, 0] = 2.0 * rng.uniform(size=n) ** 2 - 1.0
    Y = 1.0 + 2.0 * X[:, 0] + 0.2 * rng.standard_normal(n)
    return X, Y


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sampler_recovers_a_known_conditional_median(seed):
    X, Y = _skewed_linear_gaussian_data(seed)
    grid = response_grid(Y, 400)
    x0 = np.array([-0.6, -0.2, 0.2, 0.6])
    queries = np.stack([x0, np.zeros(4)], axis=1)
    sampler = PreparedConditionalSampler(X, Y, queries, [1.0, 0.0], grid)
    m = 999
    uniforms = np.tile((np.arange(m) + 0.5) / m, (4, 1))
    samples = sampler.sample(0.3, m, uniforms=uniforms)
    # The conditional median is 1 + 2 x0. The skewed design makes the
    # local-linear correction matter: with the right weights the error is
    # below 0.032, with a wrong sign on the correction it exceeds 0.09.
    np.testing.assert_allclose(
        np.median(samples, axis=1), 1.0 + 2.0 * x0, atol=0.06
    )
