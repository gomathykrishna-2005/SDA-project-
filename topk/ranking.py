"""
ranking.py
==========
Stream ranking for the Top-K MSANA extension.

Given a dictionary of {stream_name: relevance_score}, produces a sorted
ranking table that can be used for logging, analysis, and stream selection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def rank_streams(
    relevance_scores: Dict[str, float],
) -> List[Tuple[str, float, int]]:
    """Sort streams by relevance score (descending) and assign ranks.

    Parameters
    ----------
    relevance_scores : dict
        {stream_name: relevance_score}

    Returns
    -------
    list of (stream_name, score, rank)
        Sorted highest-to-lowest, rank starts at 1.

    Example
    -------
    >>> scores = {"S1": 0.91, "S3": 0.72, "S2": 0.84, "S4": 0.41}
    >>> rank_streams(scores)
    [('S1', 0.91, 1), ('S2', 0.84, 2), ('S3', 0.72, 3), ('S4', 0.41, 4)]
    """
    sorted_items = sorted(relevance_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, score, rank + 1) for rank, (name, score) in enumerate(sorted_items)]


def ranking_table_str(ranked: List[Tuple[str, float, int]]) -> str:
    """Pretty-print a ranking table for logging.

    Parameters
    ----------
    ranked : output of :func:`rank_streams`

    Returns
    -------
    str  formatted table
    """
    lines = [
        f"{'Stream':<15} {'Relevance':>12} {'Rank':>6}",
        "-" * 36,
    ]
    for name, score, rank in ranked:
        lines.append(f"{name:<15} {score:>12.4f} {rank:>6}")
    return "\n".join(lines)
