"""
metrics.py
==========
Evaluation utilities for TK-MSANA experiments.

Provides:
- Classification metrics (accuracy, precision, recall, F1)
- Drift detection metrics (TPR, FPR, precision, recall, F1 for drift events)
- Computational efficiency metrics (timing, stream counts)
- Summary row builder for comparison tables
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Classification metrics (from results dict produced by run_tk_msana)
# ---------------------------------------------------------------------------

def extract_classification_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract final classification metrics from a run_tk_msana results dict."""
    return results["metrics_final"].copy()


# ---------------------------------------------------------------------------
# Drift detection metrics
# ---------------------------------------------------------------------------

def drift_detection_metrics(
    detected_indices: List[int],
    true_drift_indices: List[int],
    total_samples: int,
    tolerance: int = 100,
) -> Dict[str, float]:
    """Compute drift detection quality metrics.

    A detection is counted as a True Positive if it falls within
    ±``tolerance`` samples of a known true drift point.

    Parameters
    ----------
    detected_indices : list of int
        Indices where drift was detected.
    true_drift_indices : list of int
        Ground-truth drift locations.
    total_samples : int
        Total number of test samples (used to compute FP rate).
    tolerance : int
        Window around each true drift in which a detection counts as TP.

    Returns
    -------
    dict  with keys: tpr, fpr, precision, recall, f1
    """
    if not true_drift_indices:
        return dict(tpr=float("nan"), fpr=float("nan"),
                    precision=float("nan"), recall=float("nan"), f1=float("nan"))

    tp, fp, fn = 0, 0, 0
    matched_truths = set()
    matched_detections = set()

    for det_idx in detected_indices:
        found = False
        for t_idx in true_drift_indices:
            if abs(det_idx - t_idx) <= tolerance and t_idx not in matched_truths:
                tp += 1
                matched_truths.add(t_idx)
                matched_detections.add(det_idx)
                found = True
                break
        if not found:
            fp += 1

    fn = len(true_drift_indices) - len(matched_truths)

    tpr = tp / len(true_drift_indices) if true_drift_indices else 0.0
    tn  = total_samples - tp - fp - fn
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision_d = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_d    = tpr
    denom = precision_d + recall_d
    f1_d = 2 * precision_d * recall_d / denom if denom > 0 else 0.0

    return dict(
        tpr=round(tpr, 4),
        fpr=round(fpr, 4),
        precision=round(precision_d, 4),
        recall=round(recall_d, 4),
        f1=round(f1_d, 4),
    )


# ---------------------------------------------------------------------------
# Computational efficiency metrics
# ---------------------------------------------------------------------------

def extract_timing_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Summarise timing information from a results dict.

    Returns
    -------
    dict with:
        total_time_s          : total wall-clock seconds
        mean_selection_time_s : mean per-step selection time
        total_selection_time_s: sum of all selection times
        mean_adaptation_time_s: mean per-drift adaptation time
        total_adaptation_time_s: sum of adaptation times
        n_drifts_detected     : number of drift events
        n_streams_processed   : pool size (streams processed per step)
    """
    sel_times = results.get("selection_times", [])
    adp_times = results.get("adaptation_times", [])

    return {
        "total_time_s":           results.get("total_time", 0.0),
        "mean_selection_time_s":  round(statistics.mean(sel_times), 6) if sel_times else 0.0,
        "total_selection_time_s": round(sum(sel_times), 4),
        "mean_adaptation_time_s": round(statistics.mean(adp_times), 4) if adp_times else 0.0,
        "total_adaptation_time_s": round(sum(adp_times), 4),
        "n_drifts_detected":      len(results.get("drift_log", [])),
        "n_streams_processed":    results.get("pool_size", 0),
    }


# ---------------------------------------------------------------------------
# Build a flat summary row for comparison tables
# ---------------------------------------------------------------------------

def build_summary_row(
    method: str,
    k: Any,
    pool_size: int,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single flat row for the comparison CSV/table.

    Parameters
    ----------
    method : str  e.g. "MSANA-Baseline" or "TK-MSANA"
    k      : int or "all"
    pool_size : int
    results : dict returned by run_tk_msana

    Returns
    -------
    dict suitable for a pandas DataFrame row
    """
    clf = extract_classification_metrics(results)
    tim = extract_timing_metrics(results)

    row = {
        "Method":       method,
        "K":            k if k is not None else "all",
        "Pool_Size":    pool_size,
        "Accuracy":     round(clf["accuracy"]  * 100, 2),
        "Precision":    round(clf["precision"] * 100, 2),
        "Recall":       round(clf["recall"]    * 100, 2),
        "F1":           round(clf["f1"]        * 100, 2),
        "Total_Time_s": tim["total_time_s"],
        "Sel_Time_s":   tim["total_selection_time_s"],
        "Adp_Time_s":   tim["total_adaptation_time_s"],
        "N_Drifts":     tim["n_drifts_detected"],
        "Streams_Used": pool_size if (k is None or k == "all") else min(int(k), pool_size),
    }
    return row
