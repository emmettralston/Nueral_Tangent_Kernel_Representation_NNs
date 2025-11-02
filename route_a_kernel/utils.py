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
    Appends a dict as one CSV row, expanding headers if new keys appear.
    Returns the CSV path.
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, filename)
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = [r for r in reader if any((v or "").strip() for v in r.values())]
        fieldnames = list(existing_fieldnames)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
        existing_rows.append(row)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in existing_rows:
                writer.writerow({k: rec.get(k, "") for k in fieldnames})
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
    return csv_path
