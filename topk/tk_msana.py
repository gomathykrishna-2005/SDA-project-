"""
tk_msana.py
===========
TK-MSANA: Top-K stream selection layer added to the MSANA (W-PWPAE) baseline.

Architecture
------------
    Multistream Pool (m learners)
        -> EDDM Drift Detection         (same as baseline)
        -> Relevance Scoring            [NEW: WIER per stream]
        -> Stream Ranking               [NEW]
        -> Top-K Selection              [NEW: configurable K]
        -> W-PWPAE Ensemble Prediction  (same weighting logic as baseline)
        -> Drift Adaptation             (same window-based retraining, applied
                                        only to selected Top-K streams)

Baseline equivalence
--------------------
When K = None or K >= pool size, TK-MSANA is mathematically equivalent to
the baseline MSANA because the same streams are used in the same way.
This allows a direct fair comparison.

Key design decisions
--------------------
1. Relevance = windowed inverse-error-rate (WIER): the same measure already
   computed for ensemble weights, so no extra computation per step.
2. K is applied at BOTH prediction time (only Top-K streams vote) and
   adaptation time (only Top-K streams are retrained on drift).
3. Pool is configurable via the make_stream_pool() factory.
4. No future-data leakage: relevance is computed from the history
   [0 .. current_index), which contains only past errors.
5. The baseline MSANA notebooks are NOT modified.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from river import ensemble, feature_selection, linear_model, metrics
from river import neighbors, stats, stream, tree
from river.drift import DDM, EDDM

from topk.relevance import compute_relevance_dict
from topk.ranking import rank_streams, ranking_table_str
from topk.selection import select_top_k_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream pool factories
# ---------------------------------------------------------------------------

def make_cicids_pool(pool_size: int = 4, seed: int = 42) -> Dict[str, Any]:
    """Create a pool of base learners for CICIDS2017.

    The first 4 match the original MSANA notebook exactly.
    Extra learners fill the pool from other River classifiers.

    Parameters
    ----------
    pool_size : int  Total number of learners in the pool.
    seed : int       Random seed for reproducibility.
    """
    # Base 4 matching the notebook
    ordered_pool = [
        ("ARF-ADWIN", lambda: ensemble.AdaptiveRandomForestClassifier(n_models=3, seed=seed)),
        ("KNN-ADWIN", lambda: neighbors.KNNADWINClassifier(window_size=100)),
        ("ARF-EDDM",  lambda: ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM(), seed=seed)),
        ("OPA",       lambda: linear_model.PAClassifier()),
        # Extended pool
        ("ARF-DDM",   lambda: ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=DDM(), warning_detector=DDM(), seed=seed)),
        ("HAT",       lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=seed)),
        ("HT",        lambda: tree.HoeffdingTreeClassifier()),
        ("EFDT",      lambda: tree.ExtremelyFastDecisionTreeClassifier()),
        ("LR",        lambda: linear_model.LogisticRegression()),
        ("PA2",       lambda: linear_model.PAClassifier()),
    ]

    pool: Dict[str, Any] = {}
    for name, factory in ordered_pool:
        if len(pool) >= pool_size:
            break
        pool[name] = factory()

    # Pad with numbered ARFs if needed
    idx = 0
    while len(pool) < pool_size:
        pool[f"ARF-ADWIN-{idx}"] = ensemble.AdaptiveRandomForestClassifier(
            n_models=3, seed=seed + idx + 1)
        idx += 1

    return pool


def make_iotid_pool(pool_size: int = 4, seed: int = 42) -> Dict[str, Any]:
    """Create a pool of base learners for IoTID20.

    The first 4 match the original MSANA IoTID20 notebook exactly.
    """
    ordered_pool = [
        ("ARF-ADWIN", lambda: ensemble.AdaptiveRandomForestClassifier(n_models=3, seed=seed)),
        ("ARF-DDM",   lambda: ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=DDM(), warning_detector=DDM(), seed=seed)),
        ("ARF-EDDM",  lambda: ensemble.AdaptiveRandomForestClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM(), seed=seed)),
        ("EFDT",      lambda: tree.ExtremelyFastDecisionTreeClassifier()),
        # Extended pool
        ("KNN-ADWIN", lambda: neighbors.KNNADWINClassifier(window_size=100)),
        ("OPA",       lambda: linear_model.PAClassifier()),
        ("HAT",       lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=seed)),
        ("HT",        lambda: tree.HoeffdingTreeClassifier()),
        ("LR",        lambda: linear_model.LogisticRegression()),
        ("PA2",       lambda: linear_model.PAClassifier()),
    ]

    pool: Dict[str, Any] = {}
    for name, factory in ordered_pool:
        if len(pool) >= pool_size:
            break
        pool[name] = factory()

    idx = 0
    while len(pool) < pool_size:
        pool[f"ARF-ADWIN-{idx}"] = ensemble.AdaptiveRandomForestClassifier(
            n_models=3, seed=seed + idx + 1)
        idx += 1

    return pool


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def run_tk_msana(
    pool_factory: Callable[[int, int], Dict[str, Any]],
    pool_size: int,
    k: Optional[int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 42,
    fs_k: int = 40,
    fs_variance_threshold: float = 0.1,
    warmup: int = 1000,
    window_fraction: float = 0.9,
) -> Dict[str, Any]:
    """Run TK-MSANA and return a results dictionary.

    Parameters
    ----------
    pool_factory : callable
        pool_factory(pool_size, seed) -> dict  -- creates the stream pool.
    pool_size : int
        Total number of base learners in the pool (m).
    k : int or None
        Number of top streams to use.  None = use all (equivalent to baseline).
    X_train, y_train : training data (typically 10% split)
    X_test, y_test   : test data (prequential evaluation)
    seed : int
    fs_k : int                    SelectKBest k parameter (baseline uses 40)
    fs_variance_threshold : float VarianceThreshold parameter (baseline 0.1)
    warmup : int                  samples before windowed scoring activates
    window_fraction : float       history fraction excluded from window (0.9 = last 10%)

    Returns
    -------
    dict with keys:
        t               : list of sample indices
        m               : list of cumulative accuracy percentages
        yt, yp          : true labels, predicted labels
        metrics_final   : dict {accuracy, precision, recall, f1}
        drift_log       : list of drift event dicts (with relevance info)
        total_time      : total wall-clock seconds
        selection_times : list of seconds per selection step
        adaptation_times: list of seconds per adaptation step
        k               : K value used
        pool_size       : pool size used
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    np.random.seed(seed)

    # -----------------------------------------------------------------------
    # Initialise pool and state
    # -----------------------------------------------------------------------
    pool = pool_factory(pool_size, seed)
    stream_names = list(pool.keys())

    metric_ensemble = metrics.Accuracy()
    per_stream_metrics = {name: metrics.Accuracy() for name in stream_names}

    # error_histories[name] = list of 0/1 per step (0=correct, 1=error)
    error_histories: Dict[str, List[int]] = {name: [] for name in stream_names}

    t_list: List[int] = []
    m_list: List[float] = []
    yt_list: List[int] = []
    yp_list: List[int] = []

    drift_log: List[Dict] = []
    selection_times: List[float] = []
    adaptation_times: List[float] = []

    # Feature selectors (same as baseline)
    selector1 = feature_selection.VarianceThreshold(threshold=fs_variance_threshold)
    selector2 = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=fs_k)

    # Drift detector (same as baseline: EDDM)
    eddm = EDDM()
    drift_flag = 0

    # -----------------------------------------------------------------------
    # Phase 1 -- Initial feature selection on training set (same as baseline)
    # -----------------------------------------------------------------------
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        selector1.learn_one(xi1)
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xi1 = selector1.transform_one(xi1)
        selector2.learn_one(xi1, yi1)

    # -----------------------------------------------------------------------
    # Phase 2 -- Pre-train all learners on training set (same as baseline)
    # -----------------------------------------------------------------------
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xi1 = selector1.transform_one(xi1)
        xi1 = selector2.transform_one(xi1)
        for name in stream_names:
            pool[name].learn_one(xi1, yi1)

    # -----------------------------------------------------------------------
    # Phase 3 -- Prequential test loop
    # -----------------------------------------------------------------------
    pending_drift_event: Optional[Dict] = None
    total_start = time.perf_counter()

    for i, (xi, yi) in enumerate(stream.iter_pandas(X_test, y_test)):
        xi = selector1.transform_one(xi)
        xi = selector2.transform_one(xi)

        # -- Step A: predict & learn for ALL streams in pool --
        predictions: Dict[str, int] = {}
        probas: Dict[str, Dict] = {}
        for name in stream_names:
            try:
                predictions[name] = pool[name].predict_one(xi)
                probas[name] = pool[name].predict_proba_one(xi)
                pool[name].learn_one(xi, yi)
            except Exception:
                # Fallback for learners that fail after feature-space change
                predictions[name] = 0
                probas[name] = {0: 0.5, 1: 0.5}
            err = 0 if predictions[name] == yi else 1
            error_histories[name].append(err)
            per_stream_metrics[name].update(yi, predictions[name])

        # -- Step B: compute relevance scores (WIER) --
        t_sel_start = time.perf_counter()
        rel_scores = compute_relevance_dict(
            error_histories, i,
            window_fraction=window_fraction,
            warmup=warmup,
        )

        # -- Step C: select Top-K streams [THE PROPOSED CONTRIBUTION] --
        selected_models, full_ranking = select_top_k_models(pool, rel_scores, k)
        sel_names = list(selected_models.keys())
        selection_times.append(time.perf_counter() - t_sel_start)

        # -- Step D: ensemble prediction using only the selected Top-K streams --
        # Weights = normalised WIER of selected streams only
        sel_scores = {n: rel_scores[n] for n in sel_names}
        total_score = sum(sel_scores.values())
        weights = {n: sel_scores[n] / total_score for n in sel_names}

        y_prob_0 = 0.0
        y_prob_1 = 0.0
        for name in sel_names:
            w = weights[name]
            y_pred_s = predictions[name]
            y_prob_s = probas[name]
            # Same probability normalisation as the baseline notebook
            if y_pred_s == 1:
                p0 = 1.0 - y_prob_s.get(1, 0.5)
                p1 = y_prob_s.get(1, 0.5)
            else:
                p0 = y_prob_s.get(0, 0.5)
                p1 = 1.0 - y_prob_s.get(0, 0.5)
            y_prob_0 += w * p0
            y_prob_1 += w * p1

        y_pred_ensemble = 0 if y_prob_0 >= y_prob_1 else 1
        metric_ensemble.update(yi, y_pred_ensemble)

        # -- Step E: drift detection (same as baseline: EDDM + i > warmup guard) --
        val = 0.0 if yi == y_pred_ensemble else 1.0
        in_drift, _in_warning = eddm.update(val)
        if in_drift and (i > warmup):
            print(f"[TK-MSANA K={k}] Drift at index {i}  selected={sel_names}")
            drift_flag = 1
            pending_drift_event = {
                "index":            i,
                "k":                k if k is not None else "all",
                "pool_size":        pool_size,
                "selected_streams": sel_names,
                "all_streams":      stream_names,
                "relevance_scores": {n: round(rel_scores[n], 4) for n in stream_names},
                "ranking":          [(nm, round(sc, 4), rk)
                                     for nm, sc, rk in full_ranking],
            }

        # -- Step F: drift adaptation (only for selected Top-K streams) --
        # Baseline: retrain ALL 4 on last 10% window.
        # Proposed:  retrain only Top-K on last 10% window.
        if drift_flag == 1:
            t_adapt_start = time.perf_counter()

            x_window = X_test.iloc[round(window_fraction * i): i]
            y_window = y_test.iloc[round(window_fraction * i): i]

            # Reinitialise selected streams with fresh models
            fresh_pool = pool_factory(pool_size, seed)
            for name in sel_names:
                pool[name] = fresh_pool[name]

            # Re-fit feature selectors on window data (same as baseline)
            selector1 = feature_selection.VarianceThreshold(
                threshold=fs_variance_threshold)
            selector2 = feature_selection.SelectKBest(
                similarity=stats.PearsonCorr(), k=fs_k)
            for xj, yj in stream.iter_pandas(x_window, y_window):
                selector1.learn_one(xj)
            for xj, yj in stream.iter_pandas(x_window, y_window):
                xj = selector1.transform_one(xj)
                selector2.learn_one(xj, yj)
            for xj, yj in stream.iter_pandas(x_window, y_window):
                xj = selector1.transform_one(xj)
                xa = selector2.transform_one(xj)
                for name in sel_names:
                    pool[name].learn_one(xa, yj)

            adapt_time = time.perf_counter() - t_adapt_start
            adaptation_times.append(adapt_time)

            if pending_drift_event is not None:
                pending_drift_event["adaptation_time_s"] = round(adapt_time, 4)
                drift_log.append(pending_drift_event)
                pending_drift_event = None

            drift_flag = 0

        t_list.append(i)
        m_list.append(metric_ensemble.get() * 100)
        yt_list.append(int(yi))
        yp_list.append(y_pred_ensemble)

    total_end = time.perf_counter()

    # -----------------------------------------------------------------------
    # Final metrics
    # -----------------------------------------------------------------------
    acc  = accuracy_score(yt_list, yp_list)
    prec = precision_score(yt_list, yp_list, zero_division=0)
    rec  = recall_score(yt_list, yp_list, zero_division=0)
    f1   = f1_score(yt_list, yp_list, zero_division=0)

    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-score:  {f1*100:.2f}%")
    print(f"  Time:      {total_end - total_start:.2f}s  |  Drifts: {len(drift_log)}")

    return {
        "t":               t_list,
        "m":               m_list,
        "yt":              yt_list,
        "yp":              yp_list,
        "metrics_final": {
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        },
        "drift_log":        drift_log,
        "total_time":       round(total_end - total_start, 4),
        "selection_times":  selection_times,
        "adaptation_times": adaptation_times,
        "k":                k,
        "pool_size":        pool_size,
    }
