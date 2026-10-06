"""
relevance.py
============
Auxiliary-stream relevance calculation for the Top-K MSANA extension.

Background
----------
In the MSANA framework, "auxiliary streams" are the pool of parallel base
learners (hat1..hatN).  Each learner produces a stream of predictions that
is compared to the true labels, giving a per-step error signal.

Relevance measure
-----------------
We use the **windowed inverse-error-rate** (WIER) as the relevance score.

    relevance_i = 1 / (windowed_error_rate_i + epsilon)

Where:
  - windowed_error_rate_i  = mean error of learner i over the most-recent
                             window (last W% of samples seen so far).
  - epsilon = 1e-3         prevents division by zero.

This measure is:
  1. The **same measure already used by the baseline** for ensemble weighting
     (no new computational overhead).
  2. Computed exclusively from **past prediction errors** — no future-data
     leakage.
  3. Higher when a learner has been **more accurate recently**, making it a
     natural proxy for how well an auxiliary stream tracks the current concept.
  4. Interpretable and theoretically grounded (mirrors the inverse-error
     weighting in PWPAE / W-PWPAE literature).

No. | Step                    | Description
----+-------------------------+------------------------------------------
 1  | Accumulate error lists  | metric_w[i] ∈ {0,1} per sample per stream
 2  | Slice window            | last ~10% of samples (same as adaptation window)
 3  | Compute error rate      | mean(window_errors)
 4  | Compute WIER            | 1 / (error_rate + epsilon)

Note: Before 1000 test samples have been processed the windowed error rate
is unreliable, so we fall back to epsilon (equal relevance for all streams),
matching the baseline behaviour.
"""

from __future__ import annotations

from typing import List, Dict


EPSILON = 1e-3          # prevents division by zero
WARMUP_SAMPLES = 1000   # match the baseline guard: i > 1000
WINDOW_FRACTION = 0.9   # last (1 - WINDOW_FRACTION) of samples = ~10%


def compute_relevance_scores(
    error_histories: List[List[int]],
    current_index: int,
    window_fraction: float = WINDOW_FRACTION,
    epsilon: float = EPSILON,
    warmup: int = WARMUP_SAMPLES,
) -> List[float]:
    """Compute the relevance (WIER) score for each auxiliary stream.

    Parameters
    ----------
    error_histories : list of lists
        ``error_histories[i]`` is the list of per-step errors (0 or 1) for
        stream *i*.  Length of each sub-list must equal ``current_index``.
    current_index : int
        Number of test samples processed so far (0-indexed, i.e. the loop
        variable ``i`` in MSANA).
    window_fraction : float
        Fraction of history to treat as "past" and exclude from the window.
        Default 0.9 means the window covers the last ~10% of samples.
    epsilon : float
        Small constant added to avoid division by zero.
    warmup : int
        Minimum number of samples before windowed scoring is used.

    Returns
    -------
    list of float
        Relevance scores for each stream, in the same order as
        ``error_histories``.  Higher score = more relevant.
    """
    n_streams = len(error_histories)

    if current_index < warmup:
        # Not enough data yet — return equal relevance (all streams equivalent)
        return [1.0 / epsilon for _ in range(n_streams)]

    window_start = round(window_fraction * current_index)
    window_len = current_index - window_start
    if window_len == 0:
        return [1.0 / epsilon for _ in range(n_streams)]

    scores = []
    for errors in error_histories:
        window_errors = errors[window_start:current_index]
        error_rate = sum(window_errors) / window_len
        scores.append(1.0 / (error_rate + epsilon))

    return scores


def compute_relevance_dict(
    error_histories: Dict[str, List[int]],
    current_index: int,
    window_fraction: float = WINDOW_FRACTION,
    epsilon: float = EPSILON,
    warmup: int = WARMUP_SAMPLES,
) -> Dict[str, float]:
    """Same as :func:`compute_relevance_scores` but with named streams.

    Parameters
    ----------
    error_histories : dict mapping stream_name -> list of per-step errors
    current_index : int
    window_fraction, epsilon, warmup : same as above

    Returns
    -------
    dict mapping stream_name -> relevance score
    """
    names = list(error_histories.keys())
    histories = [error_histories[n] for n in names]
    scores = compute_relevance_scores(
        histories, current_index, window_fraction, epsilon, warmup
    )
    return dict(zip(names, scores))
