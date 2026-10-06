"""
baseline_experiment.py
======================
Standalone script that reproduces the original MSANA (W-PWPAE) baseline.
Superseded by:  python3.9 run.py --exp baseline

Usage
-----
    python3.9 experiments/baseline_experiment.py
"""
import sys, os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE      = os.path.join(REPO_ROOT, ".venv39", "lib", "python3.9", "site-packages")
for p in [REPO_ROOT, SITE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Delegate to run.py's baseline experiment
sys.argv = ["run.py", "--exp", "baseline"]
exec(open(os.path.join(REPO_ROOT, "run.py")).read())
