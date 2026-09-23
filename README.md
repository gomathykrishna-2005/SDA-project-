# MSANA — Online Data Stream Analytics & Concept Drift Adaptation

Based on the paper: **[A Multi-Stage Automated Online Network Data Stream Analytics Framework for IIoT Systems](https://arxiv.org/pdf/2210.01985.pdf)**  
Published in **IEEE Transactions on Industrial Informatics** (Q1, IF: 11.648), doi: [10.1109/TII.2022.3212003](https://ieeexplore.ieee.org/document/9910406)  
Authors: Li Yang and Abdallah Shami — OC2 Lab, ECE Department, Western University

---

## Overview

This repo implements the MSANA framework for automated online network data stream analytics and concept drift adaptation in IIoT systems. It includes a real-time data pipeline using Apache Kafka and InfluxDB, alongside Jupyter notebooks for offline experimentation.

![System Overview](System_Overview.jpg)

---

## Repository Structure

```
.
├── Data/
│   ├── cic_0.005km_78.csv        # Sampled CICIDS2017 dataset
│   └── IoT_2020_b_0.01.csv       # Sampled IoTID20 dataset
├── Libraries/
│   └── river-0.1.0.zip           # Offline River library
├── producer.py                   # Kafka producer — streams CSV records to Kafka
├── consumer_influx.py            # Kafka consumer — stores records into InfluxDB
├── TII-MSANA-CICIDS2017.ipynb    # MSANA pipeline on CICIDS2017
├── TII-MSANA-IoTID20.ipynb       # MSANA pipeline on IoTID20
└── Paper_2210.01985.pdf          # Full paper (local copy)
```

---

## Real-Time Pipeline

The pipeline streams network traffic data through Kafka and persists it in InfluxDB for real-time monitoring or downstream analytics.

### producer.py

Reads the CICIDS2017 CSV and produces records to a Kafka topic as JSON messages.

- Topic: `network-events`
- Broker: `localhost:9092`
- Sends the first 10 rows by default (adjustable)

```bash
python producer.py
```

### consumer_influx.py

Consumes messages from the Kafka topic and writes each record as a data point into InfluxDB.

- Kafka group: `influx-storage`
- InfluxDB bucket: `network-events`
- Handles both numeric and string field types automatically

```bash
python consumer_influx.py
```

### Prerequisites

Make sure Kafka and InfluxDB are running locally before starting the pipeline.

| Service   | Default address       |
|-----------|-----------------------|
| Kafka     | `localhost:9092`      |
| InfluxDB  | `http://localhost:8086` |

InfluxDB config (set in `consumer_influx.py`):
- Org: `msana`
- Token: `msana-token-123456789`
- Bucket: `network-events`

Install dependencies:

```bash
pip install confluent-kafka influxdb-client pandas
```

---

## MSANA Framework (Notebooks)

### Pipeline Stages

1. Dynamic Data Pre-Processing — balancing, normalization
2. Dynamic Feature Engineering — Drift-based Dynamic Feature Selection (DD-FS)
3. Base Model Learning & Selection — online model training and dynamic selection
4. Online Ensemble — W-PWPAE with concept drift detection

### Algorithms

Online learners: ARF-ADWIN, ARF-EDDM, SRP, EFDT, KNN-ADWIN, SAM-KNN, OPA, LB, PWPAE, W-PWPAE  
Drift detectors: ADWIN, EDDM

### Notebooks

- `TII-MSANA-CICIDS2017.ipynb` — runs MSANA on the sampled CICIDS2017 dataset
- `TII-MSANA-IoTID20.ipynb` — runs MSANA on the sampled IoTID20 dataset

### Requirements

- Python 3.6+
- scikit-learn
- River (included in `Libraries/river-0.1.0.zip`)

---

## Datasets

| Dataset     | File                        | Source |
|-------------|-----------------------------|--------|
| CICIDS2017  | `Data/cic_0.005km_78.csv`   | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| IoTID20     | `Data/IoT_2020_b_0.01.csv`  | [Google Sites](https://sites.google.com/view/iot-network-intrusion-dataset/home) |

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

## Contact

- Email: [liyanghart@gmail.com](mailto:liyanghart@gmail.com)
- GitHub: [LiYangHart](https://github.com/LiYangHart) / [Western OC2 Lab](https://github.com/Western-OC2-Lab/)
