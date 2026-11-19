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
        @register_synthetic_data("U1")
        class U1Data(SyntheticData):
            pass
    """

    def decorator(cls: Type[SyntheticData]) -> Type[SyntheticData]:
        DATA_CLASS_REGISTRY[name] = cls
        return cls

    return decorator


@register_synthetic_data("U1")  # HY2 data
class U1Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """U1: Sinusoidal transformation with Gaussian noise"""
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


@register_synthetic_data("U2")
class U2Data(SyntheticData):
    """
    U2: multimodal conditional distribution (extends U1 by introducing latent sign variables Z_j)
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


@register_synthetic_data("U3")
class U3Data(SyntheticData):
    """
    U3: multimodal conditional distribution with latent amplitude scalers Z_j ~ Uniform(0,1)
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


@register_synthetic_data("U4")  # ZhouM2Data
class U4Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """U4: A model with heteroscedastic Gaussian noise"""
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


@register_synthetic_data("U5")
class U5Data(SyntheticData):
    def __init__(self, seed: int = 230, gamma: float = 0.5):
        """
        U5: multiplicative noise with covariate-dependent mixture

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
        Generate Y given X for the U5 model.
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
        Analytical conditional mean and std of Y | X for U5.
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


@register_synthetic_data("U6")  # ZhouM3Data
class U6Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """U6: A model with multiplicative non-Gaussian noise"""
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


@register_synthetic_data("U7")
class U7Data(SyntheticData):
    r"""
    U7: Correctly specified single-index conditional Gaussian model.

        X ~ N_3(0, I_3), eps ~ N(0, 1), independently
        beta = (1, 2, -1) / sqrt(6)
        Z = beta^T X
        Y = sin(1.5 Z) + (0.7 + 0.2 tanh(Z)) eps

    Consequently,
        E[Y | X] = sin(1.5 Z)
        SD[Y | X] = 0.7 + 0.2 tanh(Z)
    """

    def __init__(self, seed: int = 230):
        super().__init__(predictor_dim=3, response_dim=1, seed=seed)
        self.beta = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Sample the response conditionally on X."""
        Z = X @ self.beta
        eps = np.random.randn(len(X))
        return np.sin(1.5 * Z) + (0.7 + 0.2 * np.tanh(Z)) * eps

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical conditional mean and standard deviation."""
        Z = np.atleast_2d(X) @ self.beta
        mean = np.sin(1.5 * Z)
        std = 0.7 + 0.2 * np.tanh(Z)
        return mean, std


@register_synthetic_data("O7")
class O7Data(SyntheticData):
    r"""Original U7 model (before the October 2026 replacement).

        X ~ N(0, 1), eps ~ N(0, 1) independently
        T = c * X + eps, c = 20 by default
        Y = exp(sin(T))

    The conditional mean and standard deviation are computed using
    the truncated modified-Bessel expansion from the original U7 code.
    """

    def __init__(self, seed: int = 230, c: float = 20.0, K: int = 20):
        super().__init__(predictor_dim=1, response_dim=1, seed=seed)
        self.c = float(c)
        self.K = int(K)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1, 1)
        eps = np.random.normal(loc=0.0, scale=1.0, size=X.shape[0])
        T = self.c * X[:, 0] + eps
        return np.exp(np.sin(T))

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X).reshape(-1, 1)
        mu = self.c * X[:, 0]
        n_vals = np.arange(1, self.K + 1, dtype=float)
        angle = np.outer(n_vals, mu - np.pi / 2.0)
        cosine = np.cos(angle)

        coeff1 = iv(n_vals, 1.0) * np.exp(-0.5 * n_vals**2)
        m1 = iv(0, 1.0) + 2.0 * (coeff1[:, None] * cosine).sum(axis=0)

        coeff2 = iv(n_vals, 2.0) * np.exp(-0.5 * n_vals**2)
        m2 = iv(0, 2.0) + 2.0 * (coeff2[:, None] * cosine).sum(axis=0)
        return m1, np.sqrt(m2 - m1**2)


@register_synthetic_data("M1")
class M1Data(SyntheticData):
    def __init__(self, seed: int = 230):
        """M1: Multivariate output model with 5D input and 7D output"""
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


@register_synthetic_data("M2")
class M2Data(SyntheticData):
    """
    M2: a five-dimensional Gaussian response with
    covariate-dependent equicorrelation.

    X ~ Uniform(0, 1)^30,
    Y | X ~ N_5(0, Sigma(X)),
    where Sigma(X) has unit diagonal and all off-diagonal entries equal to X1.
    """

    def __init__(self, seed: int = 230):
        super().__init__(predictor_dim=30, response_dim=5, seed=seed)

    def generate(self, n_samples: int, X: np.ndarray = None):
        if X is None:
            X = np.random.rand(n_samples, self.predictor_dim)
            X_rep = X
        else:
            X = np.atleast_2d(X)
            assert (
                X.shape[1] == self.predictor_dim
            ), f"X must have shape (*, {self.predictor_dim}), but got {X.shape}"
            X_rep = np.repeat(X, repeats=n_samples, axis=0)

        Y = self._forward(X_rep)
        return Y.reshape(-1, self.response_dim), X

    def _forward(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        rho = X[:, 0]

        common_noise = np.random.randn(n, 1)
        idiosyncratic_noise = np.random.randn(n, self.response_dim)
        return (
            np.sqrt(rho)[:, None] * common_noise
            + np.sqrt(1.0 - rho)[:, None] * idiosyncratic_noise
        )

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        mean = np.zeros((n, self.response_dim))
        std = np.ones((n, self.response_dim))
        return mean, std


@register_synthetic_data("M3")
class M3Data(SyntheticData):
    """
    M3: multivariate extension of U5 with a
    covariate-dependent Gaussian-mixture error distribution.

    Y = m(X) * exp(0.25 * eps),

    where multiplication and exp(.) are applied componentwise,
    m1(X) = 5 + X1^2 / 3 + X2^2 + X3^2 / 2 + 0.5 * tanh(X4) + 0.5 * sin(X5),
    m2(X) = 5 + X1^2 / 2 + X2^2 / 3 + X3^2 + 0.5 * sin(X4) - 0.5 * tanh(X5),
    eps | X1 ~ pi(X1) N2(-mu, Sigma_eps) + {1 - pi(X1)} N2(mu, Sigma_eps),
    mu = (1, -1)^T,
    Sigma_eps = [[1.0, 0.5], [0.5, 1.0]],
    and pi(x) = logit^{-1}(gamma x) for a fixed gamma > 0.
    """

    def __init__(self, seed: int = 230, gamma: float = 0.5):
        super().__init__(predictor_dim=30, response_dim=2, seed=seed)
        self.gamma = gamma
        self.mu = np.array([1.0, -1.0])
        self.sigma_eps = np.array(
            [
                [1.0, 0.5],
                [0.5, 1.0],
            ]
        )

    def _location(self, X: np.ndarray) -> np.ndarray:
        m1 = (
            5.0
            + X[:, 0] ** 2 / 3.0
            + X[:, 1] ** 2
            + X[:, 2] ** 2 / 2.0
            + 0.5 * np.tanh(X[:, 3])
            + 0.5 * np.sin(X[:, 4])
        )

        m2 = (
            5.0
            + X[:, 0] ** 2 / 2.0
            + X[:, 1] ** 2 / 3.0
            + X[:, 2] ** 2
            + 0.5 * np.sin(X[:, 3])
            - 0.5 * np.tanh(X[:, 4])
        )

        return np.column_stack((m1, m2))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        base = self._location(X)

        print(f"gamma = {self.gamma}")
        logits = self.gamma * X[:, 0]
        pi = 1.0 / (1.0 + np.exp(-logits))

        component_negative = np.random.rand(n) < pi
        component_means = np.where(
            component_negative[:, None],
            -self.mu[None, :],
            self.mu[None, :],
        )

        gaussian_noise = np.random.multivariate_normal(
            mean=np.zeros(2), cov=self.sigma_eps, size=n
        )

        eps = component_means + gaussian_noise
        return base * np.exp(0.25 * eps)

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        base = self._location(X)
        logits = self.gamma * X[:, 0]
        pi = 1.0 / (1.0 + np.exp(-logits))

        a = 0.25
        marginal_variances = np.diag(self.sigma_eps)

        moment1_negative = np.exp(
            -a * self.mu + 0.5 * a**2 * marginal_variances
        )
        moment1_positive = np.exp(
            a * self.mu + 0.5 * a**2 * marginal_variances
        )
        exponential_mean = (
            pi[:, None] * moment1_negative[None, :]
            + (1.0 - pi[:, None]) * moment1_positive[None, :]
        )

        moment2_negative = np.exp(
            -2.0 * a * self.mu + 0.5 * (2.0 * a) ** 2 * marginal_variances
        )
        moment2_positive = np.exp(
            2.0 * a * self.mu + 0.5 * (2.0 * a) ** 2 * marginal_variances
        )
        exponential_second_moment = (
            pi[:, None] * moment2_negative[None, :]
            + (1.0 - pi[:, None]) * moment2_positive[None, :]
        )

        mean = base * exponential_mean
        variance = base**2 * (exponential_second_moment - exponential_mean**2)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        return mean, std


@register_synthetic_data("M4")
class M4Data(SyntheticData):
    r"""
    M4: predictor-dependent eight-component Gaussian mixture in R^10.

    X_j iid ~ Uniform(-1, 1), j = 1, ..., 5. Given X=x, a latent sign
    vector S=(S1,S2,S3)^T in {-1,1}^3 selects one of eight Gaussian
    mixture components. The predictor controls the component probabilities,
    the overall location, and a common rotation of the component geometry.

    The conditional distribution is

        Y | X=x ~ sum_s pi_s(x) N_10(
            m(x) + 2.5 Q(x) v(s),
            Q(x) D Q(x)^T
        ).
    """

    def __init__(self, seed: int = 230):
        super().__init__(predictor_dim=5, response_dim=10, seed=seed)

        self.states = np.array(
            [
                [s1, s2, s3]
                for s1 in (-1.0, 1.0)
                for s2 in (-1.0, 1.0)
                for s3 in (-1.0, 1.0)
            ],
            dtype=float,
        )

        s1 = self.states[:, 0]
        s2 = self.states[:, 1]
        s3 = self.states[:, 2]
        self.mode_directions = np.column_stack(
            [
                s1,
                s2,
                s3,
                s1 * s2,
                s2 * s3,
                s1 * s3,
                s1 * s2 * s3,
                -s1,
                -s2,
                -s3,
            ]
        ) / np.sqrt(10.0)

        self.D = np.kron(
            np.eye(5),
            np.diag([0.15**2, 0.35**2]),
        )

    def generate(self, n_samples: int, X: np.ndarray = None):
        if X is None:
            X = np.random.uniform(
                -1.0, 1.0, size=(n_samples, self.predictor_dim)
            )
            X_rep = X
        else:
            X = np.atleast_2d(X)
            assert (
                X.shape[1] == self.predictor_dim
            ), f"X must have shape (*, {self.predictor_dim}), but got {X.shape}"
            X_rep = np.repeat(X, repeats=n_samples, axis=0)

        Y = self._forward(X_rep)
        return Y.reshape(-1, self.response_dim), X

    def _component_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return pi_s(X) for all eight latent sign states."""
        s1 = self.states[:, 0][None, :]
        s2 = self.states[:, 1][None, :]
        s3 = self.states[:, 2][None, :]

        alpha = 1.2 * (
            X[:, 0, None] * s1 + X[:, 1, None] * s2 + X[:, 2, None] * s3
        )
        alpha += 0.6 * X[:, 3, None] * (s1 * s2 + s2 * s3)
        alpha += 0.6 * X[:, 4, None] * (s1 * s3)

        # Stable softmax over the eight components.
        alpha = alpha - alpha.max(axis=1, keepdims=True)
        exp_alpha = np.exp(alpha)
        pi_tilde = exp_alpha / exp_alpha.sum(axis=1, keepdims=True)
        return 0.8 * pi_tilde + 0.2 / 8.0

    def _location(self, X: np.ndarray) -> np.ndarray:
        m = np.zeros((X.shape[0], self.response_dim), dtype=float)
        m[:, 0] = np.sin(np.pi * X[:, 0])
        m[:, 1] = np.cos(np.pi * X[:, 0])
        m[:, 2] = X[:, 1]
        m[:, 3] = X[:, 1] ** 2
        m[:, 4] = np.tanh(X[:, 2])
        return m

    def _rotation_matrices(self, X: np.ndarray) -> np.ndarray:
        """Construct Q(x) = diag(R(theta_1), ..., R(theta_5))."""
        n = X.shape[0]
        Q = np.zeros((n, self.response_dim, self.response_dim), dtype=float)

        for j in range(1, 6):
            theta = (np.pi / 6.0) * (X[:, 3] + (j / 5.0) * X[:, 4])
            c = np.cos(theta)
            s = np.sin(theta)
            k = 2 * (j - 1)
            Q[:, k, k] = c
            Q[:, k, k + 1] = -s
            Q[:, k + 1, k] = s
            Q[:, k + 1, k + 1] = c

        return Q

    def _forward(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        pi = self._component_probabilities(X)
        location = self._location(X)
        Q = self._rotation_matrices(X)

        u = np.random.rand(n)
        cumulative = np.cumsum(pi, axis=1)
        component = (u[:, None] > cumulative).sum(axis=1)
        directions = self.mode_directions[component]

        rotated_direction = np.einsum("nij,nj->ni", Q, directions)

        # D^(1/2) has diagonal entries 0.15, 0.35 repeated five times.
        d_sqrt = np.sqrt(np.diag(self.D))
        eps = np.random.randn(n, self.response_dim)
        within_component_noise = np.einsum(
            "nij,nj->ni", Q, eps * d_sqrt[None, :]
        )

        return location + 2.5 * rotated_direction + within_component_noise

    def mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical coordinatewise conditional mean and standard deviation."""
        X = np.atleast_2d(X)
        pi = self._component_probabilities(X)
        location = self._location(X)
        Q = self._rotation_matrices(X)

        weighted_direction = pi @ self.mode_directions
        rotated_mean_direction = np.einsum("nij,nj->ni", Q, weighted_direction)
        mean = location + 2.5 * rotated_mean_direction

        rotated_directions = np.einsum("nij,sj->nsi", Q, self.mode_directions)
        component_offsets = 2.5 * rotated_directions
        centered_offsets = (
            component_offsets - 2.5 * rotated_mean_direction[:, None, :]
        )
        between_variance = np.sum(
            pi[:, :, None] * centered_offsets**2,
            axis=1,
        )

        # Diagonal of Q D Q^T for each x.
        within_covariance = np.einsum("nij,jk,nlk->nil", Q, self.D, Q)
        within_variance = np.diagonal(within_covariance, axis1=1, axis2=2)

        variance = np.maximum(within_variance + between_variance, 0.0)
        std = np.sqrt(variance)
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
