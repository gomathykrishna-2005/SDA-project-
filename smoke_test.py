"""
smoke_test.py  --  Quick end-to-end sanity check (small slice of data).
Verifies all modules import and the pipeline runs without error.
"""
import sys, os

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SITE      = os.path.join(REPO_ROOT, ".venv39", "lib", "python3.9", "site-packages")
for p in [REPO_ROOT, SITE]:
    if p not in sys.path:
        sys.path.insert(0, p)

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# --- import all project modules ---
from topk.relevance  import compute_relevance_scores, compute_relevance_dict
from topk.ranking    import rank_streams, ranking_table_str
from topk.selection  import select_top_k, select_top_k_models
from topk.tk_msana   import run_tk_msana, make_cicids_pool, make_iotid_pool
from evaluation.metrics    import build_summary_row, extract_timing_metrics
from evaluation.comparison import ResultsCollector

print("All imports: OK")

# --- unit test relevance ---
errors = {
    "S1": [0]*18 + [1, 0],   # mostly correct
    "S2": [1, 0]*10,          # 50% error
    "S3": [1]*20,             # always wrong
}
scores = compute_relevance_dict(errors, 20, window_fraction=0.5, warmup=5)
ranked = rank_streams(scores)
sel, _ = select_top_k(scores, k=2)
assert sel[0] == "S1", f"Expected S1 to rank 1, got {sel[0]}"
assert "S3" not in sel, f"S3 should not be in top-2"
print(f"Relevance unit test: OK  scores={scores}")
print(ranking_table_str(ranked))
print(f"Top-2 selected: {sel}")

# --- mini pipeline test ---
CICIDS_PATH = os.path.join(REPO_ROOT, "Data", "cic_0.005km_78.csv")
df   = pd.read_csv(CICIDS_PATH)
X    = df.drop(["Label"], axis=1)
y    = df["Label"]

# Use only first 3000 rows for smoke test speed
df_small = df.head(3000)
X_s = df_small.drop(["Label"], axis=1)
y_s = df_small["Label"]

Xtr, Xte, ytr, yte = train_test_split(
    X_s, y_s, train_size=0.1, test_size=0.9, shuffle=False, random_state=42)
Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)

print(f"\nSmoke test data: train={len(Xtr)} test={len(Xte)}")

for k in [None, 2, 1]:
    print(f"\n--- K={k if k is not None else 'all'} ---")
    res = run_tk_msana(
        pool_factory=make_cicids_pool,
        pool_size=4, k=k,
        X_train=Xtr, y_train=ytr,
        X_test=Xte,  y_test=yte,
        seed=42, fs_k=20, warmup=100,
    )
    assert res["metrics_final"]["accuracy"] > 0.5, "Accuracy too low"
    print(f"  Acc={res['metrics_final']['accuracy']*100:.2f}%  t={res['total_time']:.2f}s  OK")

# --- ResultsCollector test ---
collector = ResultsCollector()
collector.add("Test", "all", 4, res)
collector.print_table()

print("\n=== SMOKE TEST PASSED ===")
