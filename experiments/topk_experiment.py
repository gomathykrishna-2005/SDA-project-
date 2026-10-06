"""
topk_experiment.py
==================
Standalone script for the Top-K comparison experiment.
Superseded by:  python3.9 run.py --exp topk

Usage
-----
    python3.9 experiments/topk_experiment.py
"""
import sys, os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE      = os.path.join(REPO_ROOT, ".venv39", "lib", "python3.9", "site-packages")
for p in [REPO_ROOT, SITE]:
    if p not in sys.path:
        sys.path.insert(0, p)

sys.argv = ["run.py", "--exp", "topk"]
exec(open(os.path.join(REPO_ROOT, "run.py")).read())
