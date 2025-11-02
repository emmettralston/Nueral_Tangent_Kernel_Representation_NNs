import os, time
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Callable, Optional
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def _timestamp_dir(outdir: str) -> str:
    t = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(outdir, t)
    os.makedirs(path, exist_ok=True)
    return path

def _resolve_savedir(outdir: str, savedir: Optional[str]) -> str:
    if savedir is None:
        return _timestamp_dir(outdir)
    os.makedirs(savedir, exist_ok=True)
    return savedir

def plot_kernel_spectrum(
    K: np.ndarray,
    outdir: str,
    title: str = "Kernel eigenvalue spectrum",
    savedir: Optional[str] = None,
) -> str:
    """Saves a log-eigs plot and returns path."""
    eigs = np.linalg.eigvalsh(K)
    eigs = np.maximum(eigs, 1e-15)  # clamp for log
    fig = plt.figure()
    plt.plot(np.arange(1, len(eigs)+1), np.sort(eigs)[::-1])
    plt.yscale("log")
    plt.xlabel("eigenvalue index (desc)")
    plt.ylabel("eigenvalue (log scale)")
    plt.title(title)
    savedir = _resolve_savedir(outdir, savedir)
    path = os.path.join(savedir, "kernel_spectrum.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)
    return path

def plot_lambda_curve(
    lams: Sequence[float],
    vals: Sequence[float],
    task: str,
    outdir: str,
    savedir: Optional[str] = None,
) -> str:
    """Classification: accuracy; Regression: RMSE."""
    fig = plt.figure()
    plt.semilogx(lams, vals, marker="o")
    plt.xlabel("λ")
    if task == "classification":
        plt.ylabel("Validation accuracy")
        ttl = "λ-scan (validation accuracy)"
        fname = "lambda_curve_acc.png"
    else:
        plt.ylabel("Validation RMSE")
        ttl = "λ-scan (validation RMSE)"
        fname = "lambda_curve_rmse.png"
    plt.title(ttl)
    savedir = _resolve_savedir(outdir, savedir)
    path = os.path.join(savedir, fname)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)
    return path

def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    outdir: str,
    grid_res: int = 200,
    batch: int = 8192,
    savedir: Optional[str] = None,
) -> str:
    """Only for 2D features. Uses batched prediction to avoid OOM."""
    assert X.shape[1] == 2, "Decision boundary plot requires 2D inputs."
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # batched predictions
    zz = np.empty(grid.shape[0], dtype=np.int32)
    for i in range(0, grid.shape[0], batch):
        j = min(i + batch, grid.shape[0])
        zz[i:j] = predict_fn(grid[i:j])
    zz = zz.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, zz, levels=2, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor="k")
    plt.title("NTK decision boundary")
    plt.xlabel("x1"); plt.ylabel("y1")
    savedir = _resolve_savedir(outdir, savedir)
    path = os.path.join(savedir, "decision_boundary.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)
    return path

def plot_confmat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: str,
    savedir: Optional[str] = None,
) -> str:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title("Confusion matrix")
    savedir = _resolve_savedir(outdir, savedir)
    path = os.path.join(savedir, "confusion_matrix.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)
    return path
