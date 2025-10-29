"""
Data loading & standardization utilities.
Route A needs tiny, clean datasets + deterministic splits.
"""
from typing import Tuple
import numpy as np
from sklearn.datasets import make_moons, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import Split

def load_classification(seed: int = 42) -> Split:
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
    xs = StandardScaler().fit(X_tr)
    return Split(xs.transform(X_tr), y_tr, xs.transform(X_val), y_val, xs.transform(X_te), y_te)

def load_regression(seed: int = 42) -> Split:
    D = load_diabetes()  # tiny numeric tabular regression set
    X, y = D.data, D.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
    xs, ys = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr.reshape(-1,1))
    return Split(xs.transform(X_tr), ys.transform(y_tr.reshape(-1,1)).ravel(),
                 xs.transform(X_val), ys.transform(y_val.reshape(-1,1)).ravel(),
                 xs.transform(X_te), ys.transform(y_te.reshape(-1,1)).ravel())
