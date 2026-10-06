"""
run.py  --  Master entry point for TK-MSANA experiments.

Usage
-----
    python3.9 run.py                        # all experiments
    python3.9 run.py --exp baseline         # baseline only
    python3.9 run.py --exp topk             # Top-K comparison only
    python3.9 run.py --exp scalability      # scalability only
    python3.9 run.py --dataset CICIDS2017   # one dataset only
    python3.9 run.py --exp topk --dataset IoTID20

The script sets up the Python path (venv39 site-packages + project root)
before importing any project modules, so it can be run directly with
the system python3.9 binary.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup -- must happen before any project imports
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SITE      = os.path.join(REPO_ROOT, ".venv39", "lib", "python3.9", "site-packages")

for p in [REPO_ROOT, SITE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Now import project modules
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from river import ensemble, feature_selection, linear_model, metrics
from river import neighbors, stats, stream, tree
from river.drift import EDDM, DDM

from topk.tk_msana    import run_tk_msana, make_cicids_pool, make_iotid_pool
from topk.relevance   import compute_relevance_dict
from topk.ranking     import rank_streams, ranking_table_str
from topk.selection   import select_top_k
from evaluation.metrics    import build_summary_row, extract_timing_metrics
from evaluation.comparison import ResultsCollector

import time, csv, logging
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
CSV_DIR  = os.path.join(REPO_ROOT, "results", "csv")
PLOT_DIR = os.path.join(REPO_ROOT, "results", "plots")
os.makedirs(CSV_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "CICIDS2017": {
        "path":         os.path.join(REPO_ROOT, "Data", "cic_0.005km_78.csv"),
        "label_col":    "Label",
        "pool_factory": make_cicids_pool,
        "drift_points": [4950, 6300, 7140, 7900, 9400, 10800],
        "ylim":         (93, 100.5),
        "fs_k":         40,
    },
    "IoTID20": {
        "path":         os.path.join(REPO_ROOT, "Data", "IoT_2020_b_0.01.csv"),
        "label_col":    "Label",
        "pool_factory": make_iotid_pool,
        "drift_points": [270, 600],
        "ylim":         (78, 101),
        "fs_k":         40,
    },
}

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "seed":          42,
    "train_size":    0.1,
    "pool_size":     4,
    "k_values":      [1, 2, 3, 5],  # None = all is added automatically
    "warmup":        1000,
    "window_frac":   0.9,
    "pool_sizes_scalability": [4, 10, 20],
}


def load_dataset(ds_name):
    ds = DATASETS[ds_name]
    df = pd.read_csv(ds["path"])
    X = df.drop([ds["label_col"]], axis=1)
    y = df[ds["label_col"]]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=CONFIG["train_size"],
        test_size=1 - CONFIG["train_size"],
        shuffle=False, random_state=CONFIG["seed"])
    return (Xtr.reset_index(drop=True), Xte.reset_index(drop=True),
            ytr.reset_index(drop=True), yte.reset_index(drop=True))


# ===========================================================================
# EXPERIMENT A: Baseline MSANA (exact notebook replication)
# ===========================================================================

def run_baseline(ds_name, X_train, X_test, y_train, y_test):
    """Exact replication of the MSANA notebook logic."""
    ds = DATASETS[ds_name]
    print(f"\n{'='*60}\n  Baseline MSANA -- {ds_name}\n{'='*60}")

    if ds_name == "CICIDS2017":
        model1 = ensemble.AdaptiveRandomForestClassifier(n_models=3)
        model2 = neighbors.KNNADWINClassifier(window_size=100)
        model3 = ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM())
        model4 = linear_model.PAClassifier()
    else:
        model1 = ensemble.AdaptiveRandomForestClassifier(n_models=3)
        model2 = ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=DDM(), warning_detector=DDM())
        model3 = ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM())
        model4 = tree.ExtremelyFastDecisionTreeClassifier()

    metric = metrics.Accuracy()
    mw1, mw2, mw3, mw4 = [], [], [], []
    t_list, m_list, yt_list, yp_list = [], [], [], []
    hat1, hat2, hat3, hat4 = model1, model2, model3, model4
    drift = 0
    j = 1
    WF = CONFIG["window_frac"]
    FS_K = ds["fs_k"]
    FS_V = 0.1

    sel1 = feature_selection.VarianceThreshold(threshold=FS_V)
    sel2 = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=FS_K)
    eddm = EDDM()

    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        sel1.learn_one(xi1)
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        sel2.learn_one(sel1.transform_one(xi1), yi1)
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xa = sel2.transform_one(sel1.transform_one(xi1))
        hat1.learn_one(xa, yi1); hat2.learn_one(xa, yi1)
        hat3.learn_one(xa, yi1); hat4.learn_one(xa, yi1)

    drift_indices = []
    t0 = time.perf_counter()

    for i, (xi, yi) in enumerate(stream.iter_pandas(X_test, y_test)):
        xi = sel2.transform_one(sel1.transform_one(xi))
        p1 = hat1.predict_one(xi); pb1 = hat1.predict_proba_one(xi); hat1.learn_one(xi, yi)
        p2 = hat2.predict_one(xi); pb2 = hat2.predict_proba_one(xi); hat2.learn_one(xi, yi)
        p3 = hat3.predict_one(xi); pb3 = hat3.predict_proba_one(xi); hat3.learn_one(xi, yi)
        p4 = hat4.predict_one(xi); pb4 = hat4.predict_proba_one(xi); hat4.learn_one(xi, yi)

        mw1.append(0 if p1 == yi else 1); mw2.append(0 if p2 == yi else 1)
        mw3.append(0 if p3 == yi else 1); mw4.append(0 if p4 == yi else 1)

        ep = 0.001
        if i < CONFIG["warmup"]:
            e1 = e2 = e3 = e4 = 0.0
        else:
            ws = round(WF * i); wl = len(mw1[ws:i])
            e1 = sum(mw1[ws:i]) / wl; e2 = sum(mw2[ws:i]) / wl
            e3 = sum(mw3[ws:i]) / wl; e4 = sum(mw4[ws:i]) / wl

        ea = 1/(e1+ep)+1/(e2+ep)+1/(e3+ep)+1/(e4+ep)
        w1=1/(e1+ep)/ea; w2=1/(e2+ep)/ea; w3=1/(e3+ep)/ea; w4=1/(e4+ep)/ea

        def pprob(pred, proba):
            if pred == 1:
                return 1 - proba.get(1, 0.5), proba.get(1, 0.5)
            return proba.get(0, 0.5), 1 - proba.get(0, 0.5)

        a0,a1 = pprob(p1,pb1); b0,b1 = pprob(p2,pb2)
        c0,c1 = pprob(p3,pb3); d0,d1 = pprob(p4,pb4)
        yp0 = w1*a0+w2*b0+w3*c0+w4*d0
        yp1 = w1*a1+w2*b1+w3*c1+w4*d1
        y_pred = 0 if yp0 >= yp1 else 1
        metric.update(yi, y_pred)

        val = 0.0 if yi == y_pred else 1.0
        in_drift, _ = eddm.update(val)
        if in_drift and i > CONFIG["warmup"]:
            print(f"  Drift at index {i}")
            drift_indices.append(i); drift = 1

        if drift == 1:
            xw = X_test.iloc[round(WF*i):i]; yw = y_test.iloc[round(WF*i):i]
            if ds_name == "CICIDS2017":
                hat1 = ensemble.AdaptiveRandomForestClassifier(n_models=3)
                hat2 = neighbors.KNNADWINClassifier(window_size=100)
                hat3 = ensemble.AdaptiveRandomForestClassifier(
                    n_models=3, drift_detector=EDDM(), warning_detector=EDDM())
                hat4 = linear_model.PAClassifier()
            else:
                hat1 = ensemble.AdaptiveRandomForestClassifier(n_models=3)
                hat2 = ensemble.AdaptiveRandomForestClassifier(
                    n_models=3, drift_detector=DDM(), warning_detector=DDM())
                hat3 = ensemble.AdaptiveRandomForestClassifier(
                    n_models=3, drift_detector=EDDM(), warning_detector=EDDM())
                hat4 = tree.ExtremelyFastDecisionTreeClassifier()
            sel1 = feature_selection.VarianceThreshold(threshold=FS_V)
            sel2 = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=FS_K)
            for xj, yj in stream.iter_pandas(xw, yw):
                sel1.learn_one(xj)
            for xj, yj in stream.iter_pandas(xw, yw):
                sel2.learn_one(sel1.transform_one(xj), yj)
            for xj, yj in stream.iter_pandas(xw, yw):
                xa = sel2.transform_one(sel1.transform_one(xj))
                hat1.learn_one(xa,yj); hat2.learn_one(xa,yj)
                hat3.learn_one(xa,yj); hat4.learn_one(xa,yj)
                if j == 1:
                    print(f"  Features after adaptation: {len(xa)}"); j = 0
            drift = 0
        j = 1

        t_list.append(i); m_list.append(metric.get()*100)
        yt_list.append(int(yi)); yp_list.append(y_pred)

    elapsed = time.perf_counter() - t0
    acc  = accuracy_score(yt_list, yp_list)
    prec = precision_score(yt_list, yp_list, zero_division=0)
    rec  = recall_score(yt_list, yp_list, zero_division=0)
    f1   = f1_score(yt_list, yp_list, zero_division=0)
    print(f"  Accuracy:  {acc*100:.2f}%  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%  F1: {f1*100:.2f}%")
    print(f"  Time: {elapsed:.2f}s  Drifts: {drift_indices}")

    return {"t": t_list, "m": m_list, "yt": yt_list, "yp": yp_list,
            "metrics_final": {"accuracy": round(acc,4), "precision": round(prec,4),
                              "recall": round(rec,4), "f1": round(f1,4)},
            "total_time": round(elapsed,4), "drift_log": [{"index": d} for d in drift_indices],
            "selection_times": [], "adaptation_times": [], "k": None, "pool_size": 4}


# ===========================================================================
# EXPERIMENT B: Top-K comparison
# ===========================================================================

def run_topk_experiment(ds_name, X_train, X_test, y_train, y_test):
    ds = DATASETS[ds_name]
    print(f"\n{'='*60}\n  TK-MSANA Experiment -- {ds_name}\n{'='*60}")

    collector = ResultsCollector()
    all_results = {}

    k_list = [None] + CONFIG["k_values"]

    for k in k_list:
        k_label = "all" if k is None else str(k)
        method  = "MSANA-Baseline" if k is None else "TK-MSANA"
        print(f"\n--- K={k_label} ({method}) ---")

        res = run_tk_msana(
            pool_factory=ds["pool_factory"],
            pool_size=CONFIG["pool_size"],
            k=k,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            seed=CONFIG["seed"],
            fs_k=ds["fs_k"],
            fs_variance_threshold=0.1,
            warmup=CONFIG["warmup"],
            window_fraction=CONFIG["window_frac"],
        )
        all_results[k_label] = res
        collector.add(method, k if k is not None else "all",
                      CONFIG["pool_size"], res)

        for evt in res["drift_log"]:
            print(f"  [drift@{evt['index']}] selected={evt['selected_streams']}")
            ranked = evt.get("ranking", [])
            if ranked:
                print("  " + ranking_table_str(ranked))

    print("\n" + "="*60 + "\n  COMPARISON TABLE\n" + "="*60)
    collector.print_table()

    csv_path = os.path.join(CSV_DIR, f"topk_comparison_{ds_name}.csv")
    collector.save_csv(csv_path)

    _plot_accuracy_vs_k(all_results, k_list, ds_name, ds)
    _plot_time_vs_k(all_results, k_list, ds_name)
    _plot_baseline_vs_proposed(all_results, k_list, ds_name, ds)
    _plot_relevance_scores(all_results, ds_name)

    return all_results


# ===========================================================================
# EXPERIMENT C: Scalability
# ===========================================================================

def run_scalability_experiment(ds_name, X_train, X_test, y_train, y_test):
    ds = DATASETS[ds_name]
    print(f"\n{'='*60}\n  Scalability Experiment -- {ds_name}\n{'='*60}")

    collector = ResultsCollector()
    pool_sizes = CONFIG["pool_sizes_scalability"]
    k_values   = [None] + CONFIG["k_values"]

    for m in pool_sizes:
        print(f"\n  Pool size m={m}")
        for k in k_values:
            if k is not None and k > m:
                continue
            k_label = "all" if k is None else str(k)
            method  = "MSANA-Baseline" if k is None else "TK-MSANA"
            print(f"    K={k_label} ... ", end="", flush=True)

            res = run_tk_msana(
                pool_factory=ds["pool_factory"],
                pool_size=m, k=k,
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                seed=CONFIG["seed"],
                fs_k=ds["fs_k"],
                fs_variance_threshold=0.1,
                warmup=CONFIG["warmup"],
                window_fraction=CONFIG["window_frac"],
            )
            collector.add(method, k if k is not None else "all", m, res)
            print(f"Acc={res['metrics_final']['accuracy']*100:.2f}%  t={res['total_time']:.2f}s")

    csv_path = os.path.join(CSV_DIR, f"scalability_{ds_name}.csv")
    collector.save_csv(csv_path)
    collector.print_table()
    _plot_scalability(collector, ds_name)


def run_synthetic_scalability():
    """Scalability on synthetic data with explicit relevant/irrelevant streams."""
    print(f"\n{'='*60}\n  Scalability -- Synthetic Data\n{'='*60}")

    def make_synthetic_pool(pool_size, seed):
        """Pool with relevant (ARF-ADWIN), weakly relevant (HAT), irrelevant (HT) streams."""
        n_rel  = max(1, pool_size // 3)
        n_weak = max(1, pool_size // 3)
        n_irr  = pool_size - n_rel - n_weak
        pool = {}
        for i in range(n_rel):
            pool[f"ARF-r{i}"] = ensemble.AdaptiveRandomForestClassifier(
                n_models=3, seed=seed+i)
        for i in range(n_weak):
            pool[f"HAT-w{i}"] = tree.HoeffdingAdaptiveTreeClassifier(seed=seed+i)
        for i in range(n_irr):
            pool[f"HT-irr{i}"] = tree.HoeffdingTreeClassifier()
        return pool

    # Generate SEA synthetic data with concept drift
    from river.datasets.synth import SEA
    import random
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    n_total = 8000
    half = n_total // 2
    rows = []
    for xi, yi in SEA(variant=0, seed=CONFIG["seed"]).take(half):
        rows.append({**xi, "label": int(yi)})
    for xi, yi in SEA(variant=2, seed=CONFIG["seed"]+1).take(n_total - half):
        rows.append({**xi, "label": int(yi)})

    df_syn = pd.DataFrame(rows)
    y_syn  = df_syn.pop("label")
    X_syn  = df_syn.astype(float)
    split  = int(len(X_syn) * CONFIG["train_size"])
    Xtr, Xte = X_syn.iloc[:split].reset_index(drop=True), X_syn.iloc[split:].reset_index(drop=True)
    ytr, yte = y_syn.iloc[:split].reset_index(drop=True), y_syn.iloc[split:].reset_index(drop=True)

    collector = ResultsCollector()
    pool_sizes = CONFIG["pool_sizes_scalability"]
    k_values   = [None] + CONFIG["k_values"]

    for m in pool_sizes:
        print(f"\n  Pool size m={m}")
        for k in k_values:
            if k is not None and k > m:
                continue
            k_label = "all" if k is None else str(k)
            method  = "MSANA-Baseline" if k is None else "TK-MSANA"
            print(f"    K={k_label} ... ", end="", flush=True)
            res = run_tk_msana(
                pool_factory=make_synthetic_pool,
                pool_size=m, k=k,
                X_train=Xtr, y_train=ytr,
                X_test=Xte,  y_test=yte,
                seed=CONFIG["seed"],
                fs_k=3,       # SEA has 3 features
                warmup=200,
            )
            collector.add(method, k if k is not None else "all", m, res)
            print(f"Acc={res['metrics_final']['accuracy']*100:.2f}%  t={res['total_time']:.2f}s")

    csv_path = os.path.join(CSV_DIR, "scalability_synthetic.csv")
    collector.save_csv(csv_path)
    collector.print_table()
    _plot_scalability(collector, "Synthetic")


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _plot_accuracy_vs_k(all_results, k_list, ds_name, ds):
    k_labels = ["all"] + [str(k) for k in k_list if k is not None]
    accs = [all_results[kl]["metrics_final"]["accuracy"] * 100 for kl in k_labels]
    f1s  = [all_results[kl]["metrics_final"]["f1"] * 100 for kl in k_labels]
    xlbls = ["All" if kl == "all" else kl for kl in k_labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_style("darkgrid")
    ax.plot(range(len(xlbls)), accs, "o-b", label="Accuracy (%)", linewidth=2)
    ax.plot(range(len(xlbls)), f1s,  "s--r", label="F1 (%)", linewidth=2)
    ax.set_xticks(range(len(xlbls))); ax.set_xticklabels(xlbls)
    ax.set_xlabel("K (number of selected streams)"); ax.set_ylabel("Performance (%)")
    ax.set_title(f"Prediction Performance vs K -- {ds_name}")
    ax.legend(); ax.set_ylim(ds["ylim"])
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"topk_accuracy_vs_k_{ds_name}.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")


def _plot_time_vs_k(all_results, k_list, ds_name):
    k_labels = ["all"] + [str(k) for k in k_list if k is not None]
    total_t  = [all_results[kl]["total_time"] for kl in k_labels]
    sel_t    = [sum(all_results[kl]["selection_times"]) for kl in k_labels]
    adp_t    = [sum(all_results[kl]["adaptation_times"]) for kl in k_labels]
    xlbls    = ["All" if kl == "all" else kl for kl in k_labels]
    x = range(len(xlbls))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_style("darkgrid")
    ax.bar([xi       for xi in x], total_t, 0.25, label="Total time (s)", alpha=0.8, color="steelblue")
    ax.bar([xi+0.25  for xi in x], sel_t,   0.25, label="Selection time (s)", alpha=0.8, color="darkorange")
    ax.bar([xi+0.50  for xi in x], adp_t,   0.25, label="Adaptation time (s)", alpha=0.8, color="seagreen")
    ax.set_xticks([xi+0.25 for xi in x]); ax.set_xticklabels(xlbls)
    ax.set_xlabel("K"); ax.set_ylabel("Time (s)")
    ax.set_title(f"Execution Time vs K -- {ds_name}"); ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"topk_time_vs_k_{ds_name}.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")


def _plot_baseline_vs_proposed(all_results, k_list, ds_name, ds):
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.set_style("darkgrid")
    colors = ["r", "g", "orange", "purple", "brown"]

    bl = all_results.get("all")
    if bl:
        ax.plot(bl["t"], bl["m"], "-b", linewidth=1.5,
                label=f"Baseline (all), Acc={bl['metrics_final']['accuracy']*100:.2f}%")

    for idx, k in enumerate([k for k in k_list if k is not None]):
        r = all_results.get(str(k))
        if r:
            ax.plot(r["t"], r["m"], "--", color=colors[idx % len(colors)], linewidth=1.2,
                    label=f"TK-MSANA K={k}, Acc={r['metrics_final']['accuracy']*100:.2f}%")

    for dp in ds["drift_points"]:
        ax.axvline(dp, color="red", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Samples"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Baseline vs Top-K MSANA -- {ds_name}")
    ax.set_ylim(ds["ylim"]); ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"topk_baseline_vs_proposed_{ds_name}.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")


def _plot_relevance_scores(all_results, ds_name):
    last_event = None
    for kl in reversed(list(all_results.keys())):
        dl = all_results[kl].get("drift_log", [])
        if dl and "relevance_scores" in dl[-1]:
            last_event = dl[-1]; break
    if last_event is None:
        print("  [relevance plot] No drift events with scores -- skipping.")
        return

    scores = last_event["relevance_scores"]
    names  = list(scores.keys()); vals = [scores[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.2), 5))
    bars = ax.bar(range(len(names)), vals, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Relevance Score (WIER)")
    ax.set_title(f"Auxiliary Stream Relevance -- {ds_name} (drift@{last_event['index']})")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"topk_relevance_scores_{ds_name}.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")


def _plot_scalability(collector, label):
    rows = collector.rows
    pool_sizes_seen = sorted(set(r["Pool_Size"] for r in rows))
    k_vals_seen = sorted(set(str(r["K"]) for r in rows),
                         key=lambda x: (x != "all", int(x) if x != "all" else 0))
    colors = ["b", "r", "g", "orange", "purple", "brown"]

    for metric, ylabel, suffix in [
        ("Total_Time_s",  "Total execution time (s)",      "time"),
        ("Accuracy",      "Accuracy (%)",                  "accuracy"),
        ("Adp_Time_s",    "Total adaptation time (s)",     "adaptation"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("darkgrid")
        for idx, kv in enumerate(k_vals_seen):
            pts = [(r["Pool_Size"], r[metric]) for r in rows if str(r["K"]) == kv]
            if pts:
                pts.sort(); xs, ys = zip(*pts)
                lbl = "All (baseline)" if kv == "all" else f"Top-K (K={kv})"
                ls  = "-" if kv == "all" else "--"
                ax.plot(xs, ys, ls, marker="o", color=colors[idx%len(colors)],
                        label=lbl, linewidth=2)
        ax.set_xlabel("Number of available streams (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Number of Streams -- {label}")
        ax.legend(); plt.tight_layout()
        sfx = label.replace(" ", "_")
        path = os.path.join(PLOT_DIR, f"scalability_{suffix}_vs_streams_{sfx}.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"  Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="TK-MSANA experiments")
    parser.add_argument("--exp",
                        choices=["baseline", "topk", "scalability", "all"],
                        default="all")
    parser.add_argument("--dataset",
                        choices=list(DATASETS.keys()) + ["all"],
                        default="all")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    exps     = ["baseline", "topk", "scalability"] if args.exp == "all" else [args.exp]

    for ds_name in datasets:
        print(f"\n{'#'*65}")
        print(f"#  Dataset: {ds_name}")
        print(f"{'#'*65}")
        X_train, X_test, y_train, y_test = load_dataset(ds_name)

        if "baseline" in exps:
            bl_res = run_baseline(ds_name, X_train, X_test, y_train, y_test)
            # Save baseline CSV
            csv_path = os.path.join(CSV_DIR, f"baseline_{ds_name}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Method","K","Pool_Size","Accuracy","Precision",
                             "Recall","F1","Total_Time_s","N_Drifts"])
                m = bl_res["metrics_final"]
                w.writerow(["MSANA-Baseline","all",4,
                             round(m["accuracy"]*100,2), round(m["precision"]*100,2),
                             round(m["recall"]*100,2), round(m["f1"]*100,2),
                             bl_res["total_time"], len(bl_res["drift_log"])])
            print(f"  CSV saved: {csv_path}")
            # Baseline accuracy plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.set_style("darkgrid")
            ax.plot(bl_res["t"], bl_res["m"], "-b",
                    label=f"Baseline, Acc={bl_res['m'][-1]:.2f}%")
            ds = DATASETS[ds_name]
            for dp in ds["drift_points"]:
                ax.axvline(dp, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
            for evt in bl_res["drift_log"]:
                ax.axvline(evt["index"], color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Samples"); ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Baseline MSANA -- {ds_name}")
            ax.set_ylim(ds["ylim"]); ax.legend()
            plt.tight_layout()
            plot_path = os.path.join(PLOT_DIR, f"baseline_{ds_name}_accuracy.png")
            plt.savefig(plot_path, dpi=150); plt.close()
            print(f"  Plot saved: {plot_path}")

        if "topk" in exps:
            run_topk_experiment(ds_name, X_train, X_test, y_train, y_test)

    if "scalability" in exps:
        # IoTID20 is faster (~25s/run vs ~20s for CICIDS2017 at pool=4)
        # Run scalability on IoTID20 to keep total time reasonable
        scale_ds = [ds_name for ds_name in datasets if ds_name == "IoTID20"]
        if not scale_ds:
            scale_ds = datasets[:1]   # fall back to first dataset
        for ds_name in scale_ds:
            X_train, X_test, y_train, y_test = load_dataset(ds_name)
            run_scalability_experiment(ds_name, X_train, X_test, y_train, y_test)

    if "scalability" in exps and "synthetic" not in (args.dataset or ""):
        run_synthetic_scalability()

    print("\n[run.py] All done. Results in results/csv/ and results/plots/")


if __name__ == "__main__":
    main()
