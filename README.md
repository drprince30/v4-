
# FCCT AI Infrastructure Cascade Defense Engine V4

V4 is the hard pivot from generic GPU monitoring to **multi-layer AI infrastructure cascade defense**.

## What V4 does

- separates cascade risk from ordinary fleet stress
- classifies failure shape
- builds multiple graphs:
  - thermal graph
  - cooling graph
  - power graph
  - fabric graph
  - scheduler graph
  - tenant graph
- computes per-graph FCCT cascade risk
- detects cooling/power/fabric/scheduler cascade risks
- treats isolated HBM/ECC/XID as point failures, not cascades
- generates operator recommendations and audit report
- includes stress-test suite

## Run

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

## Local

```bash
pip install -r requirements.txt
python mock_prometheus.py
streamlit run app.py
```

## Stress test

```bash
python stress_test.py
```

Outputs:

```text
stress_results/v4_stress_results.csv
stress_results/v4_stress_summary.csv
```

## Best company positioning

"AI Infrastructure Cascade Defense: topology-aware risk intelligence for cooling, power, fabric and scheduler cascades in GPU clusters."

## What it is not

- not a replacement for DCGM
- not generic AIOps
- not automatic control
- not a guarantee
- not meant to diagnose every isolated GPU defect
