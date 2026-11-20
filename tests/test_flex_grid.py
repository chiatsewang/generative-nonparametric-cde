"""Tests for the FlexCode saved-grid check in scripts/sample_flex.py."""

import numpy as np
import pytest

from scripts.sample_flex import predict_on_training_grid, sample_Y_given_X


class FakeFlexModel:
    """Stands in for a trained FlexCodeModel; needs no flexcode install."""

    def __init__(self, y_grid, returned_grid="same", rows=None):
        self.y_grid = y_grid
        self.returned_grid = returned_grid
        self.rows = rows
        self.requested_n_grid = None

    def predict(self, X, n_grid):
        self.requested_n_grid = n_grid
        grid = np.linspace(0.0, 1.0, n_grid)
        rows = len(X) if self.rows is None else self.rows
        cde = np.ones((rows, n_grid))
        if self.returned_grid == "same":
            return cde, grid
        if self.returned_grid == "shifted":
            return cde, grid + 0.5
        return cde


X_COND = np.zeros((3, 1))


def test_uses_the_saved_training_grid_size():
    grid = np.linspace(0.0, 1.0, 17)
    model = FakeFlexModel(grid)
    cde, y_grid = predict_on_training_grid(model, X_COND)
    assert model.requested_n_grid == 17
    assert cde.shape == (3, 17)
    np.testing.assert_allclose(y_grid, grid)


def test_accepts_predict_that_returns_only_the_density():
    model = FakeFlexModel(np.linspace(0.0, 1.0, 9), returned_grid="none")
    cde, y_grid = predict_on_training_grid(model, X_COND)
    assert cde.shape == (3, 9) and y_grid.shape == (9,)


def test_refuses_a_checkpoint_without_a_saved_grid():
    with pytest.raises(ValueError, match="no saved response grid"):
        predict_on_training_grid(FakeFlexModel(None), X_COND)

    class NoAttribute:
        pass

    with pytest.raises(ValueError):
        predict_on_training_grid(NoAttribute(), X_COND)


@pytest.mark.parametrize(
    "bad_grid",
    [
        np.array([0.0]),
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, 0.5, 0.5, 1.0]),
        np.array([1.0, 0.5, 0.0]),
    ],
)
def test_refuses_an_invalid_saved_grid(bad_grid):
    with pytest.raises(ValueError, match="finite and increasing"):
        predict_on_training_grid(FakeFlexModel(bad_grid), X_COND)


def test_refuses_a_density_with_the_wrong_shape():
    model = FakeFlexModel(np.linspace(0.0, 1.0, 9), rows=2)
    with pytest.raises(ValueError, match="does not match"):
        predict_on_training_grid(model, X_COND)


def test_refuses_a_predicted_grid_that_differs_from_the_saved_one():
    model = FakeFlexModel(np.linspace(0.0, 1.0, 9), returned_grid="shifted")
    with pytest.raises(ValueError, match="different"):
        predict_on_training_grid(model, X_COND)


def test_sampling_returns_values_on_the_grid_and_follows_the_density():
    np.random.seed(0)
    grid = np.array([0.0, 1.0, 2.0])
    cde = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    samples = sample_Y_given_X(cde, grid, num_samples=200)
    assert samples.shape == (3, 200, 1)
    assert np.all(samples[0] == 1.0)
    assert np.all(samples[1] == 0.0)
    # An all-zero density falls back to uniform over the grid.
    assert set(np.unique(samples[2])) <= {0.0, 1.0, 2.0}
