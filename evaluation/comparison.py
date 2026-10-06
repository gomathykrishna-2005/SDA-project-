"""
comparison.py
=============
Comparison table and CSV export for TK-MSANA experiments.

Usage
-----
    from evaluation.comparison import ResultsCollector
    collector = ResultsCollector()
    collector.add("MSANA-Baseline", k=None, pool_size=4, results=baseline_results)
    collector.add("TK-MSANA", k=1, pool_size=4, results=k1_results)
    collector.add("TK-MSANA", k=2, pool_size=4, results=k2_results)
    collector.save_csv("results/csv/comparison.csv")
    collector.print_table()
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from evaluation.metrics import build_summary_row


class ResultsCollector:
    """Collects experiment results and exports them to CSV / console table."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        method: str,
        k: Any,
        pool_size: int,
        results: Dict[str, Any],
    ) -> None:
        """Add one experiment result.

        Parameters
        ----------
        method    : display name (e.g. "MSANA-Baseline", "TK-MSANA")
        k         : K value used (int or None/"all")
        pool_size : total pool size (m)
        results   : dict returned by run_tk_msana
        """
        row = build_summary_row(method, k, pool_size, results)
        self.rows.append(row)

    def save_csv(self, path: str) -> None:
        """Write all rows to a CSV file, creating parent directories as needed."""
        if not self.rows:
            print("[comparison] No rows to save.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fieldnames = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[comparison] Results saved to {path}")

    def print_table(self) -> None:
        """Print a formatted comparison table to stdout."""
        if not self.rows:
            print("[comparison] No results to display.")
            return

        cols = [
            ("Method",       18, "s"),
            ("K",             4, "s"),
            ("Pool_Size",     9, "d"),
            ("Accuracy",      9, ".2f"),
            ("F1",            8, ".2f"),
            ("Total_Time_s", 13, ".3f"),
            ("Sel_Time_s",   12, ".4f"),
            ("Adp_Time_s",   12, ".4f"),
            ("N_Drifts",      8, "d"),
            ("Streams_Used", 12, "d"),
        ]

        # header
        header = "  ".join(f"{name:<{w}}" for name, w, _ in cols)
        sep    = "  ".join("-" * w for name, w, _ in cols)
        print("\n" + header)
        print(sep)

        for row in self.rows:
            parts = []
            for name, w, fmt in cols:
                val = row.get(name, "")
                if isinstance(val, float):
                    parts.append(f"{val:<{w}.{fmt[1:]}}" if fmt != "d" else f"{int(val):<{w}}")
                elif isinstance(val, int):
                    parts.append(f"{val:<{w}d}")
                else:
                    parts.append(f"{str(val):<{w}}")
            print("  ".join(parts))
        print()
