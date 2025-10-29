import argparse, os, numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from .data import load_classification, load_regression
from .kernel import ntk_relu
from .utils import set_seed
from .plots import plot_kernel_spectrum, plot_lambda_curve, plot_decision_boundary, plot_confmat

def solve_ridge(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = K.shape[0]
    return np.linalg.solve(K + lam*np.eye(n), y)

def parse_lams(s: str):
    # accepts "1e-8,1e-6,1e-4,1e-2" or "logspace(start,stop,count)" e.g. "logspace(-8,-1,8)"
    s = s.strip()
    if s.startswith("logspace"):
        inside = s[s.find("(")+1:s.find(")")]
        a, b, c = inside.split(",")
        return np.logspace(float(a), float(b), int(c)).tolist()
    return [float(x) for x in s.split(",") if x]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification","regression"], default="classification")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-6)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--outdir", type=str, default="experiments")
    p.add_argument("--grid_lams", type=str, default="logspace(-8,-1,8)")
    p.add_argument("--db_grid", type=int, default=200, help="Decision-boundary grid resolution (per axis)")
    p.add_argument("--db_batch", type=int, default=8192, help="Batch size for decision-boundary predictions")

    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.task == "classification":
        split = load_classification(args.seed)

        # Train once with provided λ
        K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
        alpha = solve_ridge(K_tr, (split.y_train*2-1), args.lam)
        K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
        yv = (K_val @ alpha > 0).astype(int)
        K_te = ntk_relu(split.X_test, split.X_train, depth=args.depth)
        yt = (K_te @ alpha > 0).astype(int)
        print("val_acc:", accuracy_score(split.y_val, yv))
        print("test_acc:", accuracy_score(split.y_test, yt))

        if args.save_plots:
            # (a) spectrum on K_train
            sp_path = plot_kernel_spectrum(K_tr, args.outdir)

            # (b) λ-scan curve on validation accuracy
            lams = parse_lams(args.grid_lams)
            vals = []
            for lam in lams:
                a = solve_ridge(K_tr, (split.y_train*2-1), lam)
                yv_scan = (K_val @ a > 0).astype(int)
                vals.append(accuracy_score(split.y_val, yv_scan))
            lc_path = plot_lambda_curve(lams, vals, task="classification", outdir=args.outdir)

            # (c) decision boundary + confusion matrix (only if 2D)
            if split.X_train.shape[1] == 2:
                def predict_fn(grid):
                    Kg = ntk_relu(grid, split.X_train, depth=args.depth)
                    return (Kg @ alpha > 0).astype(int)
                db_path = plot_decision_boundary(
                    np.vstack([split.X_train, split.X_val, split.X_test]),
                    np.hstack([split.y_train, split.y_val, split.y_test]),
                    predict_fn,
                    args.outdir,
                    grid_res=args.db_grid,
                    batch=args.db_batch,
                )
            cm_path = plot_confmat(split.y_test, yt, args.outdir)

    else:  # regression
        split = load_regression(args.seed)

        K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
        alpha = solve_ridge(K_tr, split.y_train, args.lam)
        K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
        K_te  = ntk_relu(split.X_test, split.X_train, depth=args.depth)
        # sklearn version-agnostic RMSE:
        print("val_rmse:", np.sqrt(mean_squared_error(split.y_val, K_val @ alpha)))
        print("test_rmse:", np.sqrt(mean_squared_error(split.y_test, K_te  @ alpha)))

        if args.save_plots:
            sp_path = plot_kernel_spectrum(K_tr, args.outdir)
            lams = parse_lams(args.grid_lams)
            vals = []
            for lam in lams:
                a = solve_ridge(K_tr, split.y_train, lam)
                vals.append(np.sqrt(mean_squared_error(split.y_val, (K_val @ a))))
            lc_path = plot_lambda_curve(lams, vals, task="regression", outdir=args.outdir)

if __name__ == "__main__":
    main()
