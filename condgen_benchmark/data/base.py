import abc
from typing import Optional, Tuple

import numpy as np


class SyntheticData(abc.ABC):
    def __init__(self, predictor_dim: int, response_dim: int, seed: int = 230):
        """
        Base class for synthetic conditional distributions.

        Args:
            predictor_dim (int): Dimensionality of the predictor variable X.
            response_dim (int): Dimensionality of the response variable Y.
            seed (int): Random seed for reproducibility.
        """
        self.predictor_dim = predictor_dim
        self.response_dim = response_dim
        self.seed = seed
        np.random.seed(self.seed)

    def generate(
        self, n_samples: int, X: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples Y ~ p(Y | X).

        Args:
            n_samples (int): Number of samples to generate per row of X.
            X (np.ndarray, optional): Conditioning variable of shape
                (1, predictor_dim) or (n_X, predictor_dim).
                If None, X is sampled from N(0, I) with shape (n_samples, predictor_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Y: Sampled outputs, shape (n_samples, response_dim) if X is None,
                     or (n_X * n_samples, response_dim) if X is given.
                - X: Original conditioning variable of shape
                     (n_samples, predictor_dim) if X is None,
                     or (n_X, predictor_dim) if X is provided.
        """
        if X is None:
            X = np.random.randn(n_samples, self.predictor_dim)
            X_rep = X
        else:
            X = np.atleast_2d(X)
            assert (
                X.shape[1] == self.predictor_dim
            ), f"X must have shape (*, {self.predictor_dim}), but got {X.shape}"
            X_rep = np.repeat(X, repeats=n_samples, axis=0)

        Y = self._forward(X_rep)
        return Y.reshape(-1, self.response_dim), X

    @abc.abstractmethod
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute a sample Y ~ p(Y | X). Must be implemented by subclasses.

        Args:
            X (np.ndarray): Conditioning variable of shape (n, predictor_dim).

        Returns:
            np.ndarray: Generated samples Y of shape (n, response_dim).
        """
        pass

    @abc.abstractmethod
    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the conditional mean and standard deviation of Y given X.

        Args:
            X (np.ndarray): Conditioning input of shape (n, predictor_dim)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mean: Conditional expectation E[Y|X], shape (n, response_dim)
                - std: Conditional standard deviation std[Y|X], shape (n, response_dim)
        """
        pass
