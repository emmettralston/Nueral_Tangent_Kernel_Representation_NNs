"""
Data loading & standardization utilities.
Route A needs tiny, clean datasets + deterministic splits.
"""
from typing import Tuple
import numpy as np
from sklearn.datasets import make_moons, load_diabetes, load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import Split

def _split_standardize(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Split:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
    xs = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s, X_te_s = xs.transform(X_tr), xs.transform(X_val), xs.transform(X_te)
    return Split(X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)

def load_classification(dataset: str = "moons", seed: int = 42) -> Split:
    dataset = dataset.lower()
    if dataset == "moons":
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
        return _split_standardize(X, y, seed)

    if dataset == "cancer":
        D = load_breast_cancer()
        X, y = D.data, D.target
        return _split_standardize(X, y, seed)

    if dataset == "digits2d":
        D = load_digits()
        mask = np.isin(D.target, [0, 1])
        X, y = D.data[mask], D.target[mask]
        # PCA to 2D for boundary visualization; deterministic solver
        pca = PCA(n_components=2, svd_solver="full")
        X_2d = pca.fit_transform(X)
        return _split_standardize(X_2d, y, seed)

    raise ValueError(f"Unknown classification dataset '{dataset}'.")

def load_regression(seed: int = 42) -> Split:
    D = load_diabetes()  # tiny numeric tabular regression set
    X, y = D.data, D.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
    xs, ys = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr.reshape(-1,1))
    return Split(xs.transform(X_tr), ys.transform(y_tr.reshape(-1,1)).ravel(),
                 xs.transform(X_val), ys.transform(y_val.reshape(-1,1)).ravel(),
                 xs.transform(X_te), ys.transform(y_te.reshape(-1,1)).ravel())
