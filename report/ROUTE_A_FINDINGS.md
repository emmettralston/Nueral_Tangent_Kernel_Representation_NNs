## Route A Findings Snapshot

- **Reproduce**: `MPLCONFIGDIR=.matplotlib python experiments/sweep.py`
- **Artifacts**: Latest runs logged in `experiments/results_log.csv`; plots in timestamped folders under `experiments/`.

### Classification
- **Moons**: best val acc 0.875 at depth 2 / λ=1e-1; test acc 0.87. Spectrum remains heavy-tailed with gentle decay, indicating many useful modes; decision boundary shows NTK smoothing around noisy crescent edges.
- **Breast Cancer**: depth 2 / λ=1e-1 reaches val 0.989, test 0.991; spectrum dominated by top eigens (clear elbow) aligning with low intrinsic dimensionality; confusion matrix shows perfect separation on test.
- **Digits (PCA-2D)**: depth 2 / λ=1e-8 achieves val/test 1.0; spectrum flattens more slowly than moons due to digit manifolds; boundary tightly tracks class clusters with minimal margin errors.

### Regression (Diabetes)
- Depth 1 / λ=1e-1 minimizes val RMSE at 0.69 (test 0.70). Spectra exhibit rapid eigen decay; deeper kernels provide no gain, suggesting limited benefit beyond first depth.

### Notes
- CV sweeps search λ over `logspace(-8,-1,8)` and seeds `{42,123}`, retrain on train+val, and log `cv=1`.
- Decision-boundary plots saved when feature dimension == 2 (moons & digits2d). Consider seeding more runs if broader uncertainty estimates are needed.
