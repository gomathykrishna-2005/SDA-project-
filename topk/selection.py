"""
selection.py
============
Top-K stream selection for the TK-MSANA extension.

This module selects the K most relevant auxiliary streams from a ranked list.
K is fully configurable.  Passing K = None or K >= pool_size is treated as
"All" (baseline behaviour).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from topk.ranking import rank_streams


def select_top_k(
    relevance_scores: Dict[str, float],
    k: Optional[int],
) -> Tuple[List[str], List[Tuple[str, float, int]]]:
    """Select the Top-K streams by relevance score.

    Parameters
    ----------
    relevance_scores : dict
        {stream_name: relevance_score}
    k : int or None
        Number of streams to select.
        ``None`` or a value >= number of streams means "select all".

    Returns
    -------
    selected_names : list of str
        Names of the selected Top-K streams (highest relevance first).
    full_ranking : list of (name, score, rank)
        Complete ranking of all streams (for logging / analysis).
    """
    ranked = rank_streams(relevance_scores)
    pool_size = len(ranked)

    if k is None or k >= pool_size:
        # "All" mode — no selection, same as baseline
        selected_names = [name for name, _score, _rank in ranked]
    else:
        k_clamped = max(1, min(k, pool_size))
        selected_names = [name for name, _score, _rank in ranked[:k_clamped]]

    return selected_names, ranked


def select_top_k_models(
    models: Dict[str, Any],
    relevance_scores: Dict[str, float],
    k: Optional[int],
) -> Tuple[Dict[str, Any], List[Tuple[str, float, int]]]:
    """Convenience wrapper that returns the actual model objects.

    Parameters
    ----------
    models : dict
        {stream_name: model_object}
    relevance_scores : dict
        {stream_name: relevance_score}  (keys must match ``models``)
    k : int or None

    Returns
    -------
    selected_models : dict
        {stream_name: model_object} for the Top-K streams.
    full_ranking : list of (name, score, rank)
    """
    selected_names, ranked = select_top_k(relevance_scores, k)
    selected_models = {name: models[name] for name in selected_names}
    return selected_models, ranked
