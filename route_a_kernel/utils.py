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

import csv, time, os
from typing import Dict, Optional

def append_result_row(outdir: str, row: Dict, filename: str = "results_log.csv") -> str:
    """
    Appends a dict as one CSV row. Creates the file with headers if missing.
    Returns the CSV path.
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, filename)
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return csv_path