"""
Deterministic sweep over Route A configurations.
Runs CV-tuned classification/regression jobs, saving plots and appending rows to experiments/results_log.csv.
"""
import subprocess, sys, os, shlex

PY = sys.executable  # current python
ROOT = os.path.dirname(os.path.dirname(__file__))

def run(cmd: str):
    print(">>", cmd)
    return subprocess.run(shlex.split(cmd), cwd=ROOT, check=True)

def main():
    depths = [1, 2]
    lam_grid = '"logspace(-8,-1,8)"'
    seed_grid = "42,123"

    # Classification datasets
    for depth in depths:
        for dataset in ["moons", "cancer", "digits2d"]:
            run(
                f"{PY} -m route_a_kernel.train "
                f"--task classification --depth {depth} "
                f"--dataset {dataset} --cv "
                f"--grid_lams {lam_grid} --grid_seeds {seed_grid} "
                f"--save_plots --outdir experiments"
            )

    # Regression baseline (diabetes)
    for depth in depths:
        run(
            f"{PY} -m route_a_kernel.train "
            f"--task regression --depth {depth} "
            f"--cv --grid_lams {lam_grid} --grid_seeds {seed_grid} "
            f"--save_plots --outdir experiments"
        )

if __name__ == "__main__":
    main()
