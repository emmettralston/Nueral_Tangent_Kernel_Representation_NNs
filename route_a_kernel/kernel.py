"""
NTK Gram matrix builders.
Step 2 will implement a ReLU NTK recursion (depth parameter, activation swap).
For now we keep the API and a temporary placeholder.
"""
import numpy as np

def ntk_relu(X1: np.ndarray, X2: np.ndarray, depth: int = 1):
    """
    Placeholder: will be replaced by closed-form ReLU NTK recursion in Step 2.
    Returns a (n1 x n2) Gram matrix.
    """
    # TEMP: RBF as a stand-in so train.py can be wired; replaced next step.
    gamma = 1.0 / X1.shape[1]
    X1_sq = (X1**2).sum(1)[:,None]
    X2_sq = (X2**2).sum(1)[None,:]
    d2 = X1_sq + X2_sq - 2*X1@X2.T
    return np.exp(-gamma*d2)
