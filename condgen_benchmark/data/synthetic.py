from typing import Dict, Tuple, Type

import numpy as np
from scipy.special import iv

from condgen_benchmark.data.base import SyntheticData

# Global registry for synthetic data classes
DATA_CLASS_REGISTRY: Dict[str, Type[SyntheticData]] = {}


def register_synthetic_data(name: str):
    """
    Decorator to register a synthetic data class in the global registry.

    Args:
        name (str): The name to register the class under

    Example:
        @register_synthetic_data("M1")
        class M1Data(SyntheticData):
            pass
    """

    def decorator(cls: Type[SyntheticData]) -> Type[SyntheticData]:
        DATA_CLASS_REGISTRY[name] = cls
        return cls

    return decorator


@register_synthetic_data("M1")  # HY2 data
class M1Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """M1: Sinusoidal transformation with Gaussian noise"""
        super().__init__(predictor_dim=4, response_dim=1, seed=seed)
        self.theta_true = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.theta_true /= np.linalg.norm(self.theta_true)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        r"""
        .. math::

            Y = \sin(X) \cdot \theta + \varepsilon

        where:
            - :math:`\theta = \frac{[0.5, 0.5, 0.5, 0.5]}{\|\cdot\|}`
            - :math:`X \in \mathbb{R}^4`
            - :math:`\varepsilon \sim \mathcal{N}(0, 1)`
        """
        eps = np.random.randn(len(X))
        return np.sin(X) @ self.theta_true + eps

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.sin(X) @ self.theta_true
        std = np.ones_like(mean)
        return mean, std


@register_synthetic_data("M2")
class M2Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """
        M2: 10D input but only first 4 dimensions matter (intrinsic dim test)
        """
        super().__init__(predictor_dim=10, response_dim=1, seed=seed)
        self.theta_true = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.theta_true /= np.linalg.norm(self.theta_true)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        r"""
        .. math::

            Y = \sin(X_{1:4}) \cdot \theta + \varepsilon

        where:
            - :math:`\theta = \frac{[0.5, 0.5, 0.5, 0.5]}{\|\cdot\|}`
            - :math:`X \in \mathbb{R}^{10}` but only first 4 dimensions used
            - :math:`\varepsilon \sim \mathcal{N}(0, 1)`
        """
        eps = np.random.randn(len(X))
        return np.sin(X[:, :4]) @ self.theta_true + eps

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.sin(X[:, :4]) @ self.theta_true
        std = np.ones_like(mean)
        return mean, std


@register_synthetic_data("M3")
class M3Data(M1Data):
    """
    M3: Same as M1, except predictors are correlated.
    We use the transformation: X = Σ^(1/2) Z, where Z ~ N(0,I), so that X ~ N(0,Σ).
    In M5, Σ has power decay correlation structure:
    Σ_ij = rho^{|i-j|}, i,j=1,...,4, with 0 <= |rho| <= 1.
    """

    def __init__(self, seed: int = 230, rho: float = 0.5):
        super().__init__(seed)
        self.rho = rho
        idx = np.arange(self.predictor_dim)
        Sigma = self.rho ** np.abs(np.subtract.outer(idx, idx))
        self.Sigma_sqrt = np.linalg.cholesky(Sigma)

    def generate(self, n_samples: int, X: np.ndarray = None):
        if X is None:
            Z = np.random.randn(n_samples, self.predictor_dim)
            X = Z @ self.Sigma_sqrt.T
            X_rep = X
        else:
            X = np.atleast_2d(X)
            X_rep = np.repeat(X, repeats=n_samples, axis=0)

        Y = self._forward(X_rep)
        return Y.reshape(-1, self.response_dim), X

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.sin(X) @ self.theta_true
        std = np.ones_like(mean)
        return mean, std


@register_synthetic_data("M4")
class M4Data(SyntheticData):
    """
    M4: multimodal conditional distribution (extends M1 by introducing latent sign variables Z_j)
    Y = 0.5 * sum_{j=1}^4 sin(Z_j * X_j) + eps
    X_j ~ N(0,1) i.i.d.,  Z_j ∈{-1,+1}, Pr(Zj = 1) = Pr(Zj = −1) = 1/2,  eps ~ N(0,1).
    """

    def __init__(self, seed: int = 230):
        super().__init__(predictor_dim=4, response_dim=1, seed=seed)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape [n, 4]
        return Y: shape [n]
        """
        Z = np.random.choice([-1.0, 1.0], size=X.shape).astype(np.float32)
        base = 0.5 * np.sin(Z * X).sum(axis=1)
        eps = np.random.randn(len(X))
        return (base + eps).astype(np.float32)

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        E[Y|X]=0
        Var[Y|X] = 1 + 1/4 * Σ sin^2(X_j)
        """
        mean = np.zeros(X.shape[0], dtype=np.float32)
        var = 1.0 + 0.25 * (np.sin(X) ** 2).sum(axis=1)
        std = np.sqrt(var).astype(np.float32)
        return mean, std


@register_synthetic_data("M5")
class M5Data(SyntheticData):
    """
    M5: multimodal conditional distribution with latent amplitude scalers Z_j ~ Uniform(0,1)
    Y = 0.5 * sum_{j=1}^4 sin(Z_j * X_j) + eps,
    X_j ~ N(0,1) i.i.d., Z_j ~ U(0,1) i.i.d., eps ~ N(0,1).

    """

    def __init__(self, seed: int = 230):
        super().__init__(predictor_dim=4, response_dim=1, seed=seed)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape [n, 4]
        return Y: shape [n]
        """
        Z = np.random.rand(*X.shape).astype(np.float32)
        base = 0.5 * np.sin(Z * X).sum(axis=1)
        eps = np.random.randn(len(X))
        return (base + eps).astype(np.float32)

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        E[Y|X] = 0.5 * sum_j (1 - cos X_j) / X_j        (use limit 0 at X_j=0)
        Var[Y|X] = 1 + 0.25 * sum_j { E[sin^2(Z_j X_j)|X_j] - E[sin(Z_j X_j)|X_j]^2 }
                 = 1 + 0.25 * sum_j { [1/2 - sin(2X_j)/(4X_j)] - [(1 - cos X_j)/X_j]^2 }
        """
        x = X.astype(np.float32)
        ax = np.abs(x)

        m1 = np.where(ax > 1e-6, (1.0 - np.cos(x)) / x, 0.0).astype(np.float32)
        m2 = np.where(
            ax > 1e-6, 0.5 - np.sin(2.0 * x) / (4.0 * x), 0.0
        ).astype(np.float32)

        v1 = m2 - m1**2
        mean = 0.5 * m1.sum(axis=1).astype(np.float32)
        var = 1.0 + 0.25 * v1.sum(axis=1)
        std = np.sqrt(var).astype(np.float32)
        return mean, std


@register_synthetic_data("M6")  # ZhouM2Data
class M6Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """M6: A model with heteroscedastic Gaussian noise"""
        super().__init__(predictor_dim=5, response_dim=1, seed=seed)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        r"""
        .. math::

            Y = X_1^2 + \exp(X_2 + X_3 / 3) + X_4 - X_5 +
            \left(0.5 + \frac{X_2^2}{2} + \frac{X_5^2}{2}\right) \varepsilon

        where:
            - :math:`X \in \mathbb{R}^5`
            - :math:`\varepsilon \sim \mathcal{N}(0, 1)`
        """
        eps = np.random.randn(len(X))
        sigma = 0.5 + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 4] ** 2
        return (
            X[:, 0] ** 2
            + np.exp(X[:, 1] + X[:, 2] / 3)
            + X[:, 3]
            - X[:, 4]
            + sigma * eps
        )

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4]
        std = 0.5 + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 4] ** 2
        return mean, std


@register_synthetic_data("M7")
class M7Data(SyntheticData):
    def __init__(self, seed: int = 230, gamma: float = 0.5):
        """
        M7: multiplicative noise with covariate-dependent mixture

        Y = m(X) * exp(0.25 * eps)

        where
          m(X) = 5 + X1^2/3 + X2^2 + X3^2 + X4 + X5,
          eps | X1 ~ π(X1) N(-1, 1) + {1 - π(X1)} N(1, 1),
          π(x) = logit^{-1}(γ x) for a fixed γ > 0.
        """
        super().__init__(predictor_dim=30, response_dim=1, seed=seed)
        self.gamma = gamma

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Generate Y given X for the M7 model.
        """
        n = X.shape[0]

        # m(X)
        base = (
            5
            + X[:, 0] ** 2 / 3
            + X[:, 1] ** 2
            + X[:, 2] ** 2
            + X[:, 3]
            + X[:, 4]
        )
        print(f"gamma = {self.gamma}")
        logits = self.gamma * X[:, 0]
        pi = 1.0 / (1.0 + np.exp(-logits))

        U = np.random.rand(n)
        means = np.where(U < pi, -1.0, 1.0)
        eps = np.random.normal(loc=means, scale=1.0, size=n)
        return base * np.exp(0.25 * eps)

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical conditional mean and std of Y | X for M7.
        """
        base = (
            5
            + X[:, 0] ** 2 / 3
            + X[:, 1] ** 2
            + X[:, 2] ** 2
            + X[:, 3]
            + X[:, 4]
        )

        logits = self.gamma * X[:, 0]
        pi = 1.0 / (1.0 + np.exp(-logits))  # shape (n,)

        a1 = 0.25
        a2 = 0.5

        e_exp = pi * np.exp(a1 * (-1.0) + 0.5 * a1**2) + (1.0 - pi) * np.exp(
            a1 * (1.0) + 0.5 * a1**2
        )  # shape (n,)

        e_exp2 = pi * np.exp(a2 * (-1.0) + 0.5 * a2**2) + (
            1.0 - pi
        ) * np.exp(a2 * (1.0) + 0.5 * a2**2)

        mean = base * e_exp
        # var = (base**2) * (e_exp2 - e_exp**2)
        std = np.abs(base) * np.sqrt(e_exp2 - e_exp**2)
        return mean, std


@register_synthetic_data("M8")  # ZhouM3Data
class M8Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """M8: A model with multiplicative non-Gaussian noise"""
        super().__init__(predictor_dim=30, response_dim=1, seed=seed)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        r"""
        .. math::

            Y = \left(5 + \frac{X_1^2}{3} + X_2^2 + X_3^2 + X_4 + X_5\right)
            \cdot \exp(0.5 \varepsilon)

        where:
            - :math:`X \in \mathbb{R}^{30}`
            - :math:`\varepsilon \sim \mathcal{N}(-2, 1)` or
              :math:`\mathcal{N}(2, 1)` with equal probability
        """
        U = np.random.rand(len(X))
        eps = np.where(
            U < 0.5,
            np.random.normal(-2, 1, size=len(X)),
            np.random.normal(2, 1, size=len(X)),
        )
        base = (
            5
            + X[:, 0] ** 2 / 3
            + X[:, 1] ** 2
            + X[:, 2] ** 2
            + X[:, 3]
            + X[:, 4]
        )
        return base * np.exp(0.5 * eps)

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        base = (
            5
            + X[:, 0] ** 2 / 3
            + X[:, 1] ** 2
            + X[:, 2] ** 2
            + X[:, 3]
            + X[:, 4]
        )
        e_exp = 0.5 * (np.exp(-1 + 0.125) + np.exp(1 + 0.125))
        e_exp2 = 0.5 * (np.exp(-2 + 0.5) + np.exp(2 + 0.5))
        mean = base * e_exp
        std = np.abs(base) * np.sqrt(e_exp2 - e_exp**2)
        return mean, std


@register_synthetic_data("M9")
class M9Data(SyntheticData):
    """
    M9: Non-additive noise via sine transformation, with exponential link

        X ~ N(0,1), eps ~ N(0,1) independent
        T = c X + eps
        u = sin(T)
        Y = g(u) = exp(u) = exp(sin(T))

    Conditionally on X = x, we have
        T | X=x ~ N(mu, 1),  mu = c x
        Y | X=x = exp(sin(T))

    mean_std() uses the Bessel-series representation:
        E[exp(z sin T) | X=x]
          = I_0(z) + 2 sum_{n>=1} I_n(z) exp(-n^2/2) cos(n (mu - pi/2)),
        where I_n is the modified Bessel function of the first kind.
    """

    def __init__(self, seed: int = 230, c: float = 20.0, K: int = 20):
        """
        Parameters
        ----------
        seed : int
            Random seed.
        c : float
            Nonlinearity parameter in T = c X + eps.
        K : int
            Truncation level for the Bessel series (n = 1,...,K).
            Larger K -> more accurate, but slightly slower.
        """
        super().__init__(predictor_dim=1, response_dim=1, seed=seed)
        self.c = float(c)
        self.K = int(K)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Generate Y given X for the M9 model.

        X: shape [n, 1] or [n]
        return Y: shape [n]
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        x1 = X[:, 0]

        eps = np.random.normal(loc=0.0, scale=1.0, size=n)
        T = self.c * x1 + eps
        u = np.sin(T)
        Y = np.exp(u)
        return Y

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytic conditional mean and std of Y | X for M9 with g(u) = exp(u),
        using truncated Bessel-series expansions.

        T | X=x ~ N(mu, 1), mu = c x
        Y = exp(sin(T))

        E[Y | X=x]  = m1(x)
        E[Y^2 | X=x]= m2(x)
        Var(Y|X=x)  = m2(x) - m1(x)^2
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        x1 = X[:, 0]
        mu = self.c * x1

        K = self.K
        n_vals = np.arange(1, K + 1, dtype=float)

        coeff1 = iv(n_vals, 1.0) * np.exp(-0.5 * n_vals**2)
        angle = np.outer(n_vals, mu - np.pi / 2.0)
        cos_terms = np.cos(angle)
        m1 = iv(0, 1.0) + 2.0 * (coeff1[:, None] * cos_terms).sum(axis=0)

        coeff2 = iv(n_vals, 2.0) * np.exp(-0.5 * n_vals**2)
        cos_terms2 = cos_terms  # same angles
        m2 = iv(0, 2.0) + 2.0 * (coeff2[:, None] * cos_terms2).sum(axis=0)

        var = m2 - m1**2
        std = np.sqrt(var)
        return m1, std


@register_synthetic_data("M10")
class M10Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """M10: Multivariate output model with 5D input and 7D output"""
        super().__init__(predictor_dim=5, response_dim=7, seed=seed)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        r"""
        .. math::

            Y = \begin{pmatrix}
            X_1^2 \\
            X_2^2 \\
            X_3^2 \\
            X_4^2 \\
            X_5^2 \\
            \exp(X_2 + X_5/3) \\
            \sin(X_4 + X_5)
            \end{pmatrix} + \varepsilon

        where:
            - :math:`X \in \mathbb{R}^5`
            - :math:`\varepsilon \sim \mathcal{N}(0, \mathbf{I}_7)`
        """
        n = len(X)
        eps = np.random.randn(n, 7)

        Y = np.zeros((n, 7))
        Y[:, 0] = X[:, 0] ** 2
        Y[:, 1] = X[:, 1] ** 2
        Y[:, 2] = X[:, 2] ** 2
        Y[:, 3] = X[:, 3] ** 2
        Y[:, 4] = X[:, 4] ** 2
        Y[:, 5] = np.exp(X[:, 1] + X[:, 4] / 3)
        Y[:, 6] = np.sin(X[:, 3] + X[:, 4])
        return Y + eps

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)

        mean = np.zeros((n, 7))
        mean[:, 0] = X[:, 0] ** 2
        mean[:, 1] = X[:, 1] ** 2
        mean[:, 2] = X[:, 2] ** 2
        mean[:, 3] = X[:, 3] ** 2
        mean[:, 4] = X[:, 4] ** 2
        mean[:, 5] = np.exp(X[:, 1] + X[:, 4] / 3)
        mean[:, 6] = np.sin(X[:, 3] + X[:, 4])

        std = np.ones((n, 7))
        return mean, std


def get_synthetic_data_class(data_name: str) -> Type[SyntheticData]:
    """
    Return the SyntheticData class for the given data name.

    Args:
        data_name (str): Name of the registered synthetic data model

    Returns:
        Type[SyntheticData]: Corresponding synthetic data generator class

    Raises:
        ValueError: If data_name is not recognized
    """
    if data_name not in DATA_CLASS_REGISTRY:
        raise ValueError(
            f"Unknown data_name: {data_name}. "
            f"Available data models: {list(DATA_CLASS_REGISTRY.keys())}"
        )
    return DATA_CLASS_REGISTRY[data_name]
