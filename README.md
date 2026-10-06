# MSANA + TK-MSANA — Efficient Top-K Stream Selection for Multistream Concept Drift Adaptation

This repository contains two layers:

1. **Baseline** — replication of the MSANA (W-PWPAE) framework from the paper below.
2. **Proposed** — TK-MSANA, a Top-K stream selection extension that investigates whether selecting only the most relevant auxiliary streams can reduce computational cost while maintaining prediction performance.

---

## Base Paper

**A Multi-Stage Automated Online Network Data Stream Analytics Framework for IIoT Systems**  
Li Yang and Abdallah Shami — OC2 Lab, ECE Department, Western University  
*IEEE Transactions on Industrial Informatics*, vol. 19, no. 2, pp. 2107–2116, 2023  
doi: [10.1109/TII.2022.3212003](https://ieeexplore.ieee.org/document/9910406)

---

## Work by

**Adithyaa S Kumar(2023BCS0055)**
**Gomathy Krishna Binu(2023BCD0028)**
**Aswin Suresh Kumar(2023BCS0073)**
**H S Adithyan(2023BCS0034)**

---

## Research Question

> **"Can selecting the Top-K most relevant auxiliary streams reduce the computational cost of multistream concept-drift adaptation while maintaining prediction performance?"**

---

## Proposed Pipeline

```
Multistream Pool (m learners)
    → EDDM Drift Detection       [baseline]
    → Relevance Scoring (WIER)   [NEW]
    → Stream Ranking             [NEW]
    → Top-K Selection            [NEW]
    → W-PWPAE Ensemble           [baseline]
    → Drift Adaptation           [baseline — applied only to Top-K]
```

The relevance measure is the **Windowed Inverse-Error-Rate (WIER)**:

```
relevance_i = 1 / (windowed_error_rate_i + ε)
```

where the window covers the most recent ~10% of test samples seen so far — no future-data leakage.

---

## Repository Structure

```
.
├── TII-MSANA-CICIDS2017.ipynb    # Original baseline notebook (unchanged)
├── TII-MSANA-IoTID20.ipynb       # Original baseline notebook (unchanged)
│
├── topk/                         # TK-MSANA extension modules
│   ├── relevance.py              # WIER relevance scoring
│   ├── ranking.py                # Stream ranking
│   ├── selection.py              # Top-K selection
│   └── tk_msana.py               # Core TK-MSANA algorithm + pool factories
│
├── evaluation/                   # Evaluation utilities
│   ├── metrics.py                # Build comparison table rows
│   └── comparison.py             # ResultsCollector, CSV export, table printing
│
├── run.py                        # Master experiment entry point
├── smoke_test.py                 # Quick sanity check (~10s)
│
├── experiments/                  # Standalone experiment scripts
│   ├── baseline_experiment.py    # Baseline MSANA only
│   ├── topk_experiment.py        # Top-K comparison
│   └── scalability_experiment.py # Pool-size scalability
│
├── results/
│   ├── csv/                      # All experiment results as CSV
│   └── plots/                    # All generated plots (PNG)
│
├── Data/
│   ├── cic_0.005km_78.csv        # CICIDS2017 (sampled, 14 155 rows, 78 features)
│   └── IoT_2020_b_0.01.csv       # IoTID20 (sampled, 6 252 rows, 32 features)
│
└── Libraries/
    └── river_old/                # River 0.1.0 (reference copy, not used by scripts)
```

---

## Setup

The experiments require **Python 3.9** with the packages in `.venv39`.  
The venv is already populated — no installation needed.

```bash
# Run the quick sanity check first (~10 seconds)
python3.9 smoke_test.py
```

---

## Running Experiments

All experiments are controlled through `run.py`.

```bash
# All experiments on both datasets (takes ~30–60 min for scalability)
python3.9 run.py

# Baseline replication only
python3.9 run.py --exp baseline

# Top-K comparison (K = 1, 2, 3, 5, all) on both datasets
python3.9 run.py --exp topk

# Scalability experiment (pool sizes 4, 10, 20)
python3.9 run.py --exp scalability --dataset IoTID20

# Single dataset
python3.9 run.py --exp topk --dataset CICIDS2017
```

### What each experiment produces

| Experiment | CSVs | Plots |
|---|---|---|
| `baseline` | `baseline_{DS}.csv` | `baseline_{DS}_accuracy.png` |
| `topk` | `topk_comparison_{DS}.csv` | 4 plots per dataset |
| `scalability` | `scalability_{DS}.csv`, `scalability_synthetic.csv` | 3 plots per dataset |

---

## Key Results (CICIDS2017, pool size = 4)

| Method | K | Accuracy | F1 | Total Time (s) | Adaptation Time (s) |
|---|---:|---:|---:|---:|---:|
| MSANA-Baseline | all | 99.16% | 97.44% | 19.46 | 3.48 |
| TK-MSANA | 1 | 98.78% | 96.31% | 18.28 | 2.26 |
| TK-MSANA | 2 | 98.98% | 96.90% | 19.03 | 2.65 |
| TK-MSANA | 3 | 99.06% | 97.13% | 19.35 | 2.91 |
| TK-MSANA | 5 | 99.16% | 97.44% | 19.86 | 3.45 |

**K=3 recovers 99.1% accuracy with 16% faster adaptation than baseline.**  
**K=1 reduces adaptation time by 35% with less than 1.2pp accuracy loss.**

## Key Results (IoTID20, pool size = 4)

| Method | K | Accuracy | F1 | Total Time (s) | Adaptation Time (s) |
|---|---:|---:|---:|---:|---:|
| MSANA-Baseline | all | 98.44% | 99.18% | 23.42 | 0 |
| TK-MSANA | 1 | 97.94% | 98.91% | 24.15 | 0.48 |
| TK-MSANA | 2 | 98.24% | 99.07% | 23.32 | 0.12 |
| TK-MSANA | 3 | 98.17% | 99.03% | 23.57 | 0.17 |
| TK-MSANA | 5 | 98.44% | 99.18% | 24.79 | 0 |

---

## Generated Plots

| File | Description |
|---|---|
| `topk_accuracy_vs_k_{DS}.png` | Accuracy and F1 vs K |
| `topk_time_vs_k_{DS}.png` | Total / selection / adaptation time vs K |
| `topk_baseline_vs_proposed_{DS}.png` | Accuracy-over-time: baseline vs all K values |
| `topk_relevance_scores_{DS}.png` | WIER scores at last drift event |
| `scalability_time_vs_streams_{DS}.png` | Execution time vs pool size m |
| `scalability_accuracy_vs_streams_{DS}.png` | Accuracy vs pool size m |
| `scalability_adaptation_vs_streams_{DS}.png` | Adaptation time vs pool size m |

---

## Module Reference

### `topk/relevance.py` — `compute_relevance_dict(error_histories, i, ...)`
Computes WIER score for each stream from its past error history.  
No future data used. Window = last ~10% of test samples seen.

### `topk/ranking.py` — `rank_streams(relevance_scores)`
Returns a sorted list of `(stream_name, score, rank)` tuples.

### `topk/selection.py` — `select_top_k(relevance_scores, k)`
Selects the top K stream names. `k=None` selects all (baseline mode).

### `topk/tk_msana.py` — `run_tk_msana(...)`
Core algorithm. Accepts any `pool_factory(pool_size, seed) -> dict` callable.  
Returns a results dict with metrics, drift log, and timing.

### `evaluation/comparison.py` — `ResultsCollector`
Collects rows from multiple runs and exports to CSV or prints a table.

---

## Real-Time Pipeline (original)

The original Kafka + InfluxDB streaming pipeline is unchanged.

```bash
python producer.py          # stream CICIDS2017 rows to Kafka
python consumer_influx.py   # consume from Kafka → InfluxDB
```

Requires Kafka on `localhost:9092` and InfluxDB on `http://localhost:8086`.

---

## Citation

```bibtex
@ARTICLE{9910406,
  author={Yang, Li and Shami, Abdallah},
  journal={IEEE Transactions on Industrial Informatics},
  title={A Multi-Stage Automated Online Network Data Stream Analytics Framework for IIoT Systems},
  year={2023},
  volume={19},
  number={2},
  pages={2107-2116},
  doi={10.1109/TII.2022.3212003}
}
```
