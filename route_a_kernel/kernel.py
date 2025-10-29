import numpy as np

def _relu_cov_next(S12, S11, S22, eps=1e-12):
    """
    ReLU covariance recursion.
    S12: (n1 x n2), cross-cov
    S11: (n1 x n1), auto-cov for X1
    S22: (n2 x n2), auto-cov for X2
    Returns S12_next, and also the Jacobian dotSigma used by the NTK recursion.
    """
    # pairwise normalization for cos(theta)
    denom = np.sqrt(np.clip(np.outer(np.diag(S11), np.diag(S22)), 0.0, None)) + eps
    cos_theta = np.clip(S12 / denom, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # ReLU kernel (Cho & Saul / Neural Tangents normalization)
    # Σ^{l+1}_{ij} = (1/π) * denom * (sin θ + (π - θ) cos θ)
    S12_next = (1.0 / np.pi) * denom * (np.sin(theta) + (np.pi - theta) * cos_theta)

    # derivative wrt previous layer covariance (elementwise):
    # dotΣ = ∂Σ^{l+1}/∂Σ^{l} for ReLU = (1/π) * (π - θ)
    dotSigma = (1.0 / np.pi) * (np.pi - theta)
    return S12_next, dotSigma

def _dot(X1, X2):
    # width-normalized inner product for stable recursion
    return (X1 @ X2.T) / X1.shape[1]

def ntk_relu(X1: np.ndarray, X2: np.ndarray, depth: int = 1) -> np.ndarray:
    """
    Infinite-width fully-connected ReLU NTK of depth `depth`.
    depth=1 corresponds to one hidden layer (two affine layers total).
    """
    # layer 0 covariances
    S11 = _dot(X1, X1)
    S22 = _dot(X2, X2)
    S12 = _dot(X1, X2)

    # NTK recursion init (Neural Tangents): Θ^0 = Σ^0
    Theta12 = S12.copy()

    for _ in range(depth):
        # advance covariances
        S12, dotSigma = _relu_cov_next(S12, S11, S22)
        S11, _ = _relu_cov_next(S11, S11, S11)  # auto-cov update for set 1
        S22, _ = _relu_cov_next(S22, S22, S22)  # auto-cov update for set 2

        # NTK recursion: Θ^{l+1} = Θ^{l} ⊙ dotΣ^{l+1} + Σ^{l+1}
        Theta12 = Theta12 * dotSigma + S12

    return Theta12

