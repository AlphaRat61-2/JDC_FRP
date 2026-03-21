import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PIPELINE_SCRIPTS = [
    ("app/test_environment.py", "Environment Test", False),
    ("app/run_pipeline.py", "Pipeline", False),
    ("app/run_ml_prep.py", "ML Prep", False),
    ("app/run_rl_control.py", "RL Control", True),
    ("app/run_rl_feedback.py", "RL Feedback", True),
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


def run_script(script_path: str):
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

    for script, step_name, optional in PIPELINE_SCRIPTS:
        script_full_path = ROOT / script

        print_progress(results + [(script, "RUNNING", None)])

        if not script_full_path.exists():
            if optional:
                results.append((script, "SKIPPED", None))
                print(f"[SKIP] {step_name}: file not found -> {script}\n")
                continue
            else:
                results.append((script, "FAILED", None))
                print_progress(results)
                print(f"Run stopped because required script was not found: {script}\n")
                sys.exit(1)

        returncode, elapsed = run_script(script)

        if returncode == 0:
            results.append((script, "DONE", elapsed))
        else:
            if optional:
                results.append((script, "FAILED", elapsed))
                print(f"[WARN] Optional step failed: {script}\n")
                continue
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