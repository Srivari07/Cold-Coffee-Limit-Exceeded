"""
CSAO Rail Recommendation System — Project Setup & Pipeline Runner

Usage:
    python setup.py install      # Create venv + install dependencies
    python setup.py run          # Run full 7-module pipeline (~15 min)
    python setup.py test         # Run 139 end-to-end validation tests (~10s)
    python setup.py all          # install + run + test (full setup from scratch)
    python setup.py status       # Check which outputs exist / are missing
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
TESTS_DIR = ROOT / "tests"
REQUIREMENTS = ROOT / "requirements.txt"

IS_WINDOWS = platform.system() == "Windows"
VENV_PYTHON = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / "python"
VENV_PYTHON_EXE = VENV_PYTHON.with_suffix(".exe") if IS_WINDOWS else VENV_PYTHON

PIPELINE_SCRIPTS = [
    ("01_data_generator.py", "Generating synthetic data", "~2 min"),
    ("02_feature_engineering.py", "Computing 200+ features", "~3 min"),
    ("03_baseline_models.py", "Running 4 baselines", "~1 min"),
    ("04_model_training.py", "Training ML pipeline (Two-Tower + LightGBM + DCN-v2 + MMR)", "~5 min"),
    ("05_evaluation.py", "Running evaluation + segment analysis", "~2 min"),
    ("06_system_design.py", "Simulating latency + architecture docs", "~30s"),
    ("07_business_impact.py", "A/B test design + business projections", "~30s"),
]

# Expected outputs for status check
EXPECTED_DATA = [
    "users.csv", "restaurants.csv", "menu_items.csv",
    "orders.csv", "order_items.csv", "csao_interactions.csv",
]
EXPECTED_OUTPUTS = [
    "features_train.csv", "features_val.csv", "features_test.csv",
    "baseline_results.json", "model_results.json",
    "evaluation_report.json", "segment_analysis.json", "feature_importance.json",
    "latency_benchmark.json", "monitoring_report.json",
    "ab_test_design.json", "final_submission_report.json", "cold_start_analysis.json",
    "two_tower_model.pt", "sasrec_model.pt", "dcnv2_model.pt",
    "two_tower_faiss.index", "two_tower_item_embeddings.npy", "two_tower_item_ids.npy",
    "dcnv2_norm_mean.npy", "dcnv2_norm_std.npy",
]
EXPECTED_DOCS = [
    "feature_dictionary.md", "system_architecture.md", "problem_formulation.md",
    "streaming_pipeline.md", "retraining_strategy.md", "deployment_playbook.md",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], description: str, cwd: Path = ROOT) -> bool:
    """Run a command, stream output, return True on success."""
    log.info(f"{description}...")
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), check=True,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"FAILED (exit code {e.returncode}): {description}")
        return False
    except FileNotFoundError:
        log.error(f"Command not found: {cmd[0]}")
        return False


def _find_uv() -> str | None:
    """Find uv executable."""
    return shutil.which("uv")


def _find_pip() -> str | None:
    """Find pip in venv."""
    pip_path = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / "pip"
    if IS_WINDOWS:
        pip_path = pip_path.with_suffix(".exe")
    return str(pip_path) if pip_path.exists() else None


def _venv_exists() -> bool:
    return VENV_PYTHON_EXE.exists()


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_install() -> bool:
    """Create virtual environment and install all dependencies."""
    print()
    print("=" * 70)
    print("CSAO Setup — Installing Dependencies")
    print("=" * 70)
    print()

    uv = _find_uv()
    start = time.time()

    # Step 1: Create venv
    if _venv_exists():
        log.info(f"Virtual environment already exists at {VENV_DIR}")
    else:
        if uv:
            log.info(f"Creating virtual environment with uv (Python 3.11)...")
            if not _run([uv, "venv", str(VENV_DIR), "--python", "3.11"], "Creating venv"):
                # Fallback: try without specifying python version
                if not _run([uv, "venv", str(VENV_DIR)], "Creating venv (default python)"):
                    log.error("Failed to create virtual environment with uv")
                    return False
        else:
            log.info("uv not found, using python -m venv...")
            if not _run([sys.executable, "-m", "venv", str(VENV_DIR)], "Creating venv"):
                return False

    # Step 2: Install dependencies
    if not REQUIREMENTS.exists():
        log.error(f"requirements.txt not found at {REQUIREMENTS}")
        return False

    if uv:
        log.info("Installing dependencies with uv...")
        success = _run(
            [uv, "pip", "install", "-r", str(REQUIREMENTS),
             "--python", str(VENV_PYTHON_EXE)],
            "Installing packages",
        )
        if not success:
            return False
        # Install pytest separately (not in requirements.txt)
        _run(
            [uv, "pip", "install", "pytest",
             "--python", str(VENV_PYTHON_EXE)],
            "Installing pytest",
        )
    else:
        pip = _find_pip()
        if not pip:
            # Bootstrap pip
            _run([str(VENV_PYTHON_EXE), "-m", "ensurepip", "--upgrade"], "Bootstrapping pip")
            pip = _find_pip()
        if pip:
            success = _run(
                [pip, "install", "-r", str(REQUIREMENTS)],
                "Installing packages",
            )
            if not success:
                return False
            _run([pip, "install", "pytest"], "Installing pytest")
        else:
            log.error("Could not find pip in virtual environment")
            return False

    elapsed = time.time() - start
    print()
    log.info(f"Installation complete in {_format_elapsed(elapsed)}")
    log.info(f"Python: {VENV_PYTHON_EXE}")
    print()
    return True


def cmd_run() -> bool:
    """Run the full 7-module pipeline sequentially."""
    print()
    print("=" * 70)
    print("CSAO Pipeline — Running All 7 Modules")
    print("=" * 70)
    print()

    if not _venv_exists():
        log.error(f"Virtual environment not found. Run: python setup.py install")
        return False

    # Create output directories
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    total_start = time.time()
    results = []

    for i, (script, description, est_time) in enumerate(PIPELINE_SCRIPTS, 1):
        script_path = ROOT / script
        if not script_path.exists():
            log.error(f"Script not found: {script}")
            results.append((script, False, 0))
            continue

        print()
        print(f"[{i}/7] {description} ({est_time})")
        print("-" * 50)

        step_start = time.time()
        success = _run(
            [str(VENV_PYTHON_EXE), str(script_path)],
            description,
        )
        elapsed = time.time() - step_start
        results.append((script, success, elapsed))

        if success:
            log.info(f"Completed in {_format_elapsed(elapsed)}")
        else:
            log.error(f"Failed after {_format_elapsed(elapsed)}")
            log.error("Stopping pipeline — fix the error above before continuing.")
            break

    total_elapsed = time.time() - total_start

    # Summary
    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for script, success, elapsed in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {script:<30s} {_format_elapsed(elapsed):>8s}")
    print(f"  {'':30s} {'─' * 8}")
    print(f"  {'Total':30s} {_format_elapsed(total_elapsed):>8s}")
    print()

    all_passed = all(s for _, s, _ in results)
    if all_passed:
        log.info("All 7 modules completed successfully!")
    else:
        log.error("Some modules failed. Check output above.")
    return all_passed


def cmd_test() -> bool:
    """Run pytest end-to-end validation tests."""
    print()
    print("=" * 70)
    print("CSAO Tests — Running 139 Validation Tests")
    print("=" * 70)
    print()

    if not _venv_exists():
        log.error(f"Virtual environment not found. Run: python setup.py install")
        return False

    test_file = TESTS_DIR / "test_pipeline.py"
    if not test_file.exists():
        log.error(f"Test file not found: {test_file}")
        return False

    return _run(
        [str(VENV_PYTHON_EXE), "-m", "pytest", str(test_file), "-v", "--tb=short"],
        "Running pytest",
    )


def cmd_status() -> None:
    """Check which pipeline outputs exist and which are missing."""
    print()
    print("=" * 70)
    print("CSAO Status — Output Inventory")
    print("=" * 70)
    print()

    # Environment
    print("  ENVIRONMENT:")
    print(f"    Virtual env:  {'EXISTS' if _venv_exists() else 'MISSING'} ({VENV_DIR})")
    uv = _find_uv()
    print(f"    uv:           {'FOUND' if uv else 'NOT FOUND'}")
    print()

    def _check_files(directory: Path, files: list[str], label: str) -> int:
        print(f"  {label} ({directory}):")
        missing = 0
        for f in files:
            path = directory / f
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"    [OK]      {f:<40s} ({size_mb:>7.1f} MB)")
            else:
                print(f"    [MISSING] {f}")
                missing += 1
        print()
        return missing

    m1 = _check_files(DATA_DIR, EXPECTED_DATA, "DATA FILES")
    m2 = _check_files(OUTPUTS_DIR, EXPECTED_OUTPUTS, "OUTPUT FILES")
    m3 = _check_files(DOCS_DIR, EXPECTED_DOCS, "DOCUMENTATION")

    total_missing = m1 + m2 + m3
    total_expected = len(EXPECTED_DATA) + len(EXPECTED_OUTPUTS) + len(EXPECTED_DOCS)
    total_found = total_expected - total_missing

    print(f"  SUMMARY: {total_found}/{total_expected} files present", end="")
    if total_missing == 0:
        print(" — Pipeline outputs complete!")
    else:
        print(f" — {total_missing} files missing (run: python setup.py run)")
    print()


def cmd_all() -> bool:
    """Full setup: install + run + test."""
    print()
    print("*" * 70)
    print("CSAO Full Setup — Install + Run Pipeline + Test")
    print("*" * 70)

    total_start = time.time()

    if not cmd_install():
        return False
    if not cmd_run():
        return False
    if not cmd_test():
        return False

    total_elapsed = time.time() - total_start
    print()
    print("*" * 70)
    print(f"FULL SETUP COMPLETE in {_format_elapsed(total_elapsed)}")
    print("*" * 70)
    print()
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSAO Rail Recommendation System — Setup & Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  install   Create virtual environment and install dependencies
  run       Run the full 7-module pipeline (~15 min)
  test      Run 139 end-to-end validation tests (~10s)
  all       install + run + test (full setup from scratch)
  status    Check which outputs exist / are missing
        """,
    )
    parser.add_argument(
        "command",
        choices=["install", "run", "test", "all", "status"],
        help="Command to execute",
    )

    args = parser.parse_args()

    commands = {
        "install": cmd_install,
        "run": cmd_run,
        "test": cmd_test,
        "all": cmd_all,
        "status": cmd_status,
    }

    result = commands[args.command]()

    if result is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
