import argparse, os, time, numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from .data import load_classification, load_regression
from .kernel import ntk_relu
from .utils import set_seed, append_result_row
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

def parse_seeds(s: str):
    """Parses comma-separated ints or range(start,stop[,step])."""
    s = s.strip()
    if not s:
        return []
    if s.startswith("range"):
        inside = s[s.find("(")+1:s.find(")")]
        parts = [int(x) for x in inside.split(",")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError("range expects range(start, stop[, step])")
        return list(range(start, stop, step))
    return [int(x) for x in s.split(",") if x.strip()]

def run_classification(args, lam_grid):
    seed_grid = parse_seeds(args.grid_seeds) if args.cv and args.grid_seeds else [args.seed]
    best_seed = args.seed
    best_lam = args.lam if not args.cv else None
    best_val_acc = -np.inf

    if args.cv:
        for seed in seed_grid:
            set_seed(seed)
            split = load_classification(dataset=args.dataset, seed=seed)
            y_train_pm1 = split.y_train * 2 - 1
            K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
            K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
            for lam in lam_grid:
                alpha = solve_ridge(K_tr, y_train_pm1, lam)
                y_val_pred = (K_val @ alpha > 0).astype(int)
                val_acc = accuracy_score(split.y_val, y_val_pred)
                if (val_acc > best_val_acc + 1e-12) or (
                    np.isclose(val_acc, best_val_acc) and (best_lam is None or lam < best_lam)
                ):
                    best_val_acc = val_acc
                    best_seed = seed
                    best_lam = lam
        if best_lam is None:
            raise RuntimeError("CV scan did not evaluate any (seed, λ) pairs.")
    else:
        set_seed(args.seed)

    set_seed(best_seed)
    split = load_classification(dataset=args.dataset, seed=best_seed)
    y_train_pm1 = split.y_train * 2 - 1

    K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
    K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
    alpha_cv = solve_ridge(K_tr, y_train_pm1, best_lam)
    y_val_pred = (K_val @ alpha_cv > 0).astype(int)
    val_acc = accuracy_score(split.y_val, y_val_pred)

    if args.cv:
        X_train_final = np.vstack([split.X_train, split.X_val])
        y_train_final = np.hstack([split.y_train, split.y_val])
        y_train_final_pm1 = y_train_final * 2 - 1
        K_final = ntk_relu(X_train_final, X_train_final, depth=args.depth)
        alpha_final = solve_ridge(K_final, y_train_final_pm1, best_lam)
        train_feats = X_train_final
    else:
        K_final = K_tr
        alpha_final = alpha_cv
        train_feats = split.X_train

    K_test = ntk_relu(split.X_test, train_feats, depth=args.depth)
    y_test_pred = (K_test @ alpha_final > 0).astype(int)
    test_acc = accuracy_score(split.y_test, y_test_pred)

    print("val_acc:", val_acc)
    print("test_acc:", test_acc)

    sp_path = lc_path = db_path = cm_path = None
    if args.save_plots:
        plot_savedir = os.path.join(args.outdir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(plot_savedir, exist_ok=True)

        sp_path = plot_kernel_spectrum(K_final, args.outdir, savedir=plot_savedir)
        lam_curve_vals = []
        for lam in lam_grid:
            alpha_loop = solve_ridge(K_tr, y_train_pm1, lam)
            y_val_loop = (K_val @ alpha_loop > 0).astype(int)
            lam_curve_vals.append(accuracy_score(split.y_val, y_val_loop))
        lc_path = plot_lambda_curve(
            lam_grid,
            lam_curve_vals,
            task="classification",
            outdir=args.outdir,
            savedir=plot_savedir,
        )

        if train_feats.shape[1] == 2:
            def predict_fn(grid):
                Kg = ntk_relu(grid, train_feats, depth=args.depth)
                return (Kg @ alpha_final > 0).astype(int)
            db_path = plot_decision_boundary(
                np.vstack([split.X_train, split.X_val, split.X_test]),
                np.hstack([split.y_train, split.y_val, split.y_test]),
                predict_fn,
                args.outdir,
                grid_res=args.db_grid,
                batch=args.db_batch,
                savedir=plot_savedir,
            )
        cm_path = plot_confmat(split.y_test, y_test_pred, args.outdir, savedir=plot_savedir)

    row = dict(
        task="classification",
        dataset=args.dataset,
        depth=args.depth,
        lam=best_lam,
        seed=best_seed,
        cv=int(args.cv),
        n_train=train_feats.shape[0],
        n_val=split.X_val.shape[0],
        n_test=split.X_test.shape[0],
        val_acc=float(val_acc),
        test_acc=float(test_acc),
        spectrum_path=sp_path,
        lambda_curve_path=lc_path,
        decision_boundary_path=db_path,
        confusion_matrix_path=cm_path,
    )
    csv_path = append_result_row(args.outdir, row)
    print("logged_to:", csv_path)

def run_regression(args, lam_grid):
    seed_grid = parse_seeds(args.grid_seeds) if args.cv and args.grid_seeds else [args.seed]
    best_seed = args.seed
    best_lam = args.lam if not args.cv else None
    best_val_rmse = np.inf

    if args.cv:
        for seed in seed_grid:
            set_seed(seed)
            split = load_regression(seed)
            K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
            K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
            for lam in lam_grid:
                alpha = solve_ridge(K_tr, split.y_train, lam)
                val_rmse = np.sqrt(mean_squared_error(split.y_val, K_val @ alpha))
                if (val_rmse < best_val_rmse - 1e-12) or (
                    np.isclose(val_rmse, best_val_rmse) and (best_lam is None or lam < best_lam)
                ):
                    best_val_rmse = val_rmse
                    best_seed = seed
                    best_lam = lam
        if best_lam is None:
            raise RuntimeError("CV scan did not evaluate any (seed, λ) pairs.")
    else:
        set_seed(args.seed)

    set_seed(best_seed)
    split = load_regression(best_seed)

    K_tr = ntk_relu(split.X_train, split.X_train, depth=args.depth)
    K_val = ntk_relu(split.X_val, split.X_train, depth=args.depth)
    alpha_cv = solve_ridge(K_tr, split.y_train, best_lam)
    val_rmse = np.sqrt(mean_squared_error(split.y_val, K_val @ alpha_cv))

    if args.cv:
        X_train_final = np.vstack([split.X_train, split.X_val])
        y_train_final = np.hstack([split.y_train, split.y_val])
        K_final = ntk_relu(X_train_final, X_train_final, depth=args.depth)
        alpha_final = solve_ridge(K_final, y_train_final, best_lam)
        train_feats = X_train_final
    else:
        K_final = K_tr
        alpha_final = alpha_cv
        train_feats = split.X_train

    K_test = ntk_relu(split.X_test, train_feats, depth=args.depth)
    test_rmse = np.sqrt(mean_squared_error(split.y_test, K_test @ alpha_final))

    print("val_rmse:", val_rmse)
    print("test_rmse:", test_rmse)

    sp_path = lc_path = None
    if args.save_plots:
        plot_savedir = os.path.join(args.outdir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(plot_savedir, exist_ok=True)

        sp_path = plot_kernel_spectrum(K_final, args.outdir, savedir=plot_savedir)
        lam_curve_vals = []
        for lam in lam_grid:
            alpha_loop = solve_ridge(K_tr, split.y_train, lam)
            val_loop = np.sqrt(mean_squared_error(split.y_val, K_val @ alpha_loop))
            lam_curve_vals.append(val_loop)
        lc_path = plot_lambda_curve(
            lam_grid,
            lam_curve_vals,
            task="regression",
            outdir=args.outdir,
            savedir=plot_savedir,
        )

    row = dict(
        task="regression",
        dataset="diabetes",
        depth=args.depth,
        lam=best_lam,
        seed=best_seed,
        cv=int(args.cv),
        n_train=train_feats.shape[0],
        n_val=split.X_val.shape[0],
        n_test=split.X_test.shape[0],
        val_rmse=float(val_rmse),
        test_rmse=float(test_rmse),
        spectrum_path=sp_path,
        lambda_curve_path=lc_path,
    )
    csv_path = append_result_row(args.outdir, row)
    print("logged_to:", csv_path)

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
    p.add_argument("--dataset", choices=["moons","cancer","digits2d"], default="moons",
                   help="Classification dataset to use (classification task only).")
    p.add_argument("--cv", action="store_true",
                   help="Scan λ (and optional seeds) on validation, retrain on train+val, and evaluate on test.")
    p.add_argument("--grid_seeds", type=str, default=None,
                   help="Comma list or range(start,stop[,step]) of seeds to scan when using --cv.")

    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lam_grid = parse_lams(args.grid_lams)

    if args.task == "classification":
        run_classification(args, lam_grid)
    else:
        run_regression(args, lam_grid)

if __name__ == "__main__":
    main()
