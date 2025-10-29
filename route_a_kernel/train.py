"""
Kernel ridge training & evaluation.
Implements: K_train, solve (K + λI)α = y, predict K_test α, metrics.
"""
import argparse, numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from .data import load_classification, load_regression
from .kernel import ntk_relu
from .utils import set_seed

def solve_ridge(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = K.shape[0]
    return np.linalg.solve(K + lam*np.eye(n), y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification","regression"], default="classification")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-6)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    if args.task == "classification":
        split = load_classification(args.seed)
        K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
        alpha = solve_ridge(K_tr, split.y_train*2-1, args.lam)  # labels to {-1,1}
        K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
        yv = (K_val @ alpha > 0).astype(int)
        K_te = ntk_relu(split.X_test, split.X_train, depth=args.depth)
        yt = (K_te @ alpha > 0).astype(int)
        print("val_acc:", accuracy_score(split.y_val, yv))
        print("test_acc:", accuracy_score(split.y_test, yt))

    else:  # regression
        split = load_regression(args.seed)
        K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
        alpha = solve_ridge(K_tr, split.y_train, args.lam)
        K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
        K_te  = ntk_relu(split.X_test, split.X_train, depth=args.depth)
        print("val_rmse:", mean_squared_error(split.y_val, K_val @ alpha, squared=False))
        print("test_rmse:", mean_squared_error(split.y_test, K_te  @ alpha, squared=False))

if __name__ == "__main__":
    main()
