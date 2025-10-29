from dataclasses import dataclass
import numpy as np
import random, os

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray