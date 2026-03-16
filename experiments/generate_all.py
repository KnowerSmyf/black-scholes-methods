import subprocess
import sys


SCRIPTS = [
    "experiments/gbm_paths.py",
    "experiments/convergence_tests.py",
    "experiments/error_analysis.py",
    "experiments/runtime_comparison.py",
    "experiments/strike_sweep.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}...")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()