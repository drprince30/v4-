
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from core import ScaleCalibration


def percentile_calibration(df: pd.DataFrame, metric: str, high_is_risk=True, cap: Optional[float] = None) -> ScaleCalibration:
    vals = pd.to_numeric(df.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if len(vals) < 30:
        qs = (0, 25, 50, 75, 90)
    else:
        qs = tuple(float(x) for x in np.percentile(vals, [60, 75, 85, 92, 97]))
    if cap is not None:
        qs = list(qs)
        qs[-1] = min(qs[-1], cap)
        qs = tuple(qs)
    return ScaleCalibration(metric, tuple(sorted(qs)), "percentile", f"Percentile calibration for {metric}", high_is_risk)


def event_counter_calibration(metric: str) -> ScaleCalibration:
    return ScaleCalibration(metric, (1, 2, 5, 10, 20), "event_counter", f"Counter calibration for {metric}", True)


def binary_calibration(metric: str) -> ScaleCalibration:
    return ScaleCalibration(metric, (0.5, 1, 1.5, 2, 3), "binary", f"Binary/event calibration for {metric}", True)


def inverse_flow_calibration(df: pd.DataFrame, metric: str) -> ScaleCalibration:
    # Lower coolant flow/pressure is riskier.
    vals = pd.to_numeric(df.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if len(vals) < 30:
        qs = (10, 20, 30, 35, 40)
    else:
        # For inverse, thresholds are still sorted; assign_scale will invert.
        qs = tuple(float(x) for x in np.percentile(vals, [3, 8, 15, 25, 40]))
    return ScaleCalibration(metric, tuple(sorted(qs)), "inverse_percentile", f"Low {metric} is risky", False)


def build_calibrations(df: pd.DataFrame) -> Dict[str, ScaleCalibration]:
    cals: Dict[str, ScaleCalibration] = {}
    for metric in ["temp_c", "power_w", "gpu_util", "mem_util", "fan_pct", "rack_power_kw", "pdu_current_a", "coolant_return_c", "coolant_supply_c", "job_queue_depth", "pod_restarts", "oom_events", "storage_iops", "network_tx_gbps", "nccl_latency_ms"]:
        if metric in df.columns:
            cals[metric] = percentile_calibration(df, metric)
    for metric in ["coolant_flow_lpm", "coolant_pressure_psi"]:
        if metric in df.columns:
            cals[metric] = inverse_flow_calibration(df, metric)
    for metric in ["throttle_flag"]:
        if metric in df.columns:
            cals[metric] = binary_calibration(metric)
    for metric in ["ecc_errors", "pcie_replay", "xid_errors", "nvlink_crc", "ib_crc", "packet_drops"]:
        if metric in df.columns:
            cals[metric] = event_counter_calibration(metric)
    return cals


def calibration_confidence(df: pd.DataFrame, cals: Dict[str, ScaleCalibration]) -> pd.DataFrame:
    rows = []
    n = len(df)
    incident = int(pd.to_numeric(df.get("throttle_flag", pd.Series([0]*n)), errors="coerce").fillna(0).sum())
    for metric, cal in cals.items():
        vals = pd.to_numeric(df.get(metric, pd.Series(dtype=float)), errors="coerce")
        valid = float(vals.notna().mean()) if n else 0.0
        unique = int(vals.dropna().nunique())
        span = float(vals.max() - vals.min()) if vals.notna().any() else 0.0
        conf = 0.45 * min(1, valid) + 0.25 * min(1, unique / 20) + 0.15 * min(1, span / 30) + 0.15 * min(1, incident / 20)
        rows.append({"metric": metric, "method": cal.method, "valid_ratio": round(valid,3), "unique": unique, "span": round(span,3), "confidence": round(float(conf),3), "thresholds": tuple(round(x,3) for x in cal.thresholds)})
    return pd.DataFrame(rows)
