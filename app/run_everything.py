import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PIPELINE_SCRIPTS = [
    "app/test_environment.py",
    "app/run_pipeline.py",
    "app/run_ml_prep.py",
    "app/run_rl_control.py",
    "app/run_rl_feedback.py",
]


def print_header():
    print("\n" + "=" * 60)
    print(" OILFIELD ANALYTICS PLATFORM - FULL SYSTEM RUN ")
    print("=" * 60 + "\n")


def print_progress(results):
    print("\n" + "-" * 60)
    print("PROGRESS")
    print("-" * 60)

    for script, status, elapsed in results:
        dots = "." * max(1, 28 - len(script))
        if elapsed is None:
            print(f"{script} {dots} {status}")
        else:
            print(f"{script} {dots} {status} ({elapsed:.1f}s)")

    print("-" * 60 + "\n")


def run_script(script_path):
    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=ROOT
    )

    elapsed = time.time() - start
    return result.returncode, elapsed


def main():
    print_header()

    results = []

    for script in PIPELINE_SCRIPTS:
        print_progress(results + [(script, "RUNNING", None)])

        returncode, elapsed = run_script(script)

        if returncode == 0:
            results.append((script, "DONE", elapsed))
        else:
            results.append((script, "FAILED", elapsed))
            print_progress(results)
            print(f"Run stopped because {script} failed.\n")
            sys.exit(returncode)

    print_progress(results)

    print("FULL PLATFORM RUN COMPLETE\n")
    print("Review outputs in:")
    print("  data/modeled/")
    print("  outputs/ml/")
    print("  data/reports/\n")


if __name__ == "__main__":
    main()