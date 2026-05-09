
from __future__ import annotations
import pandas as pd
import numpy as np


def observability_report(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["temp_c","power_w","gpu_util","mem_util","fan_pct","throttle_flag","ecc_errors","pcie_replay","xid_errors","nvlink_crc","ib_crc","packet_drops","nccl_latency_ms","rack_power_kw","pdu_current_a","coolant_flow_lpm","coolant_pressure_psi","job_queue_depth","pod_restarts","oom_events"]
    rows = []
    n = len(df)
    for m in metrics:
        present = m in df.columns
        vals = pd.to_numeric(df[m], errors="coerce") if present else pd.Series(dtype=float)
        valid = float(vals.notna().mean()) if present and n else 0
        span = float(vals.max()-vals.min()) if present and vals.notna().any() else 0
        stale_parts = []
        if present:
            for _, g in df.groupby(["node_id","gpu_id"], sort=False):
                v = pd.to_numeric(g[m], errors="coerce").dropna()
                stale_parts.append(float((v.diff().fillna(999)==0).mean()) if len(v)>1 else 1.0)
        stale = float(np.mean(stale_parts)) if stale_parts else 1.0
        warning = []
        if not present: warning.append("missing")
        if valid < .90: warning.append("low_validity")
        if stale > .75 and m not in ["throttle_flag","ecc_errors","pcie_replay","xid_errors","nvlink_crc","ib_crc","packet_drops","pod_restarts","oom_events"]: warning.append("stale")
        rows.append({"metric":m,"present":present,"valid_ratio":round(valid,3),"stale_ratio":round(stale,3),"range":round(span,3),"warning":"; ".join(warning) if warning else "ok"})
    return pd.DataFrame(rows)


def observability_confidence(report: pd.DataFrame) -> float:
    if report.empty: return 0.0
    penalty = 0.0
    for _, r in report.iterrows():
        if not r["present"]: penalty += .025
        if r["valid_ratio"] < .9: penalty += .04
        if "stale" in str(r["warning"]): penalty += .05
    return round(float(max(0, min(1, 1-penalty))), 3)
