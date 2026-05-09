
from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from core import FCCTConfig, ScaleCalibration, assign_scale, combine_scales, ensure_component_id, compute_graph_risk
from graphs import build_multigraph_edges


LAYER_METRIC_WEIGHTS = {
    "thermal": {"temp_c_scale": .45, "fan_pct_scale": .10, "power_w_scale": .15, "gpu_util_scale": .10, "coolant_return_c_scale": .20},
    "cooling": {"coolant_flow_lpm_scale": .35, "coolant_pressure_psi_scale": .30, "coolant_return_c_scale": .20, "temp_c_scale": .15},
    "power": {"power_w_scale": .25, "rack_power_kw_scale": .30, "pdu_current_a_scale": .30, "gpu_util_scale": .15},
    "fabric": {"pcie_replay_scale": .20, "nvlink_crc_scale": .20, "ib_crc_scale": .20, "packet_drops_scale": .15, "nccl_latency_ms_scale": .20, "xid_errors_scale": .05},
    "scheduler": {"job_queue_depth_scale": .25, "pod_restarts_scale": .25, "oom_events_scale": .25, "gpu_util_scale": .15, "mem_util_scale": .10},
    "tenant": {"gpu_util_scale": .30, "mem_util_scale": .20, "network_tx_gbps_scale": .20, "storage_iops_scale": .20, "pod_restarts_scale": .10},
}


def add_metric_scales(df: pd.DataFrame, cals: Dict[str, ScaleCalibration], config: FCCTConfig) -> pd.DataFrame:
    out = df.copy()
    for metric, cal in cals.items():
        if metric in out.columns:
            out[f"{metric}_scale"] = assign_scale(out[metric], cal, config.max_scale)
    out = ensure_component_id(out)
    return out


def add_layer_scales(df: pd.DataFrame, config: FCCTConfig) -> pd.DataFrame:
    out = df.copy()
    for layer, weights in LAYER_METRIC_WEIGHTS.items():
        available = {k: v for k, v in weights.items() if k in out.columns}
        if not available:
            out[f"{layer}_scale"] = 0
        else:
            out[f"{layer}_scale"] = out.apply(lambda r: combine_scales(r, available, config.max_scale), axis=1)
    return out


def timeline_for_layer(df: pd.DataFrame, topology: pd.DataFrame, config: FCCTConfig, layer: str, edges) -> pd.DataFrame:
    rows = []
    prev = None
    for ts, frame in df.groupby("timestamp", sort=True):
        f = frame.copy()
        f["scale"] = f[f"{layer}_scale"].astype(int)
        risk = compute_graph_risk(f, edges, layer, config, prev_score=prev)
        prev = risk.cascade_risk
        rows.append({
            "timestamp": ts,
            "graph": layer,
            "fleet_stress": risk.fleet_stress,
            "cascade_risk": risk.cascade_risk,
            "headroom": risk.headroom,
            "forcing": risk.forcing,
            "decay": risk.decay,
            "net_pressure": risk.net_pressure,
            "max_scale": risk.max_scale,
            "coherent_pairs": risk.coherent_pairs,
            "risky_components": ", ".join(risk.risky_components[:30]),
            "classification_hint": risk.classification_hint,
        })
    return pd.DataFrame(rows)


def compute_multigraph_timelines(df: pd.DataFrame, topology: pd.DataFrame, config: FCCTConfig):
    components = df[["node_id", "gpu_id", "component_id"]].drop_duplicates()
    graph_edges = build_multigraph_edges(topology, components)
    timelines = []
    for layer in LAYER_METRIC_WEIGHTS:
        timelines.append(timeline_for_layer(df, topology, config, layer, graph_edges.get(layer, [])))
    layer_timeline = pd.concat(timelines, ignore_index=True) if timelines else pd.DataFrame()
    return layer_timeline, graph_edges


def aggregate_defense_score(layer_timeline: pd.DataFrame, config: FCCTConfig) -> pd.DataFrame:
    if layer_timeline.empty:
        return pd.DataFrame()
    piv = layer_timeline.pivot_table(index="timestamp", columns="graph", values="cascade_risk", aggfunc="max").fillna(0)
    fleet = layer_timeline.pivot_table(index="timestamp", columns="graph", values="fleet_stress", aggfunc="max").fillna(0)
    pressure = layer_timeline.pivot_table(index="timestamp", columns="graph", values="net_pressure", aggfunc="max").fillna(0)
    coherent = layer_timeline.pivot_table(index="timestamp", columns="graph", values="coherent_pairs", aggfunc="max").fillna(0)

    weights = {"thermal": .18, "cooling": .22, "power": .20, "fabric": .22, "scheduler": .12, "tenant": .06}
    out = pd.DataFrame(index=piv.index)
    out["cascade_score"] = sum(weights.get(c, 0) * piv.get(c, 0) for c in weights)
    out["fleet_stress_score"] = sum(weights.get(c, 0) * fleet.get(c, 0) for c in weights)
    out["net_pressure_score"] = sum(weights.get(c, 0) * pressure.get(c, 0) for c in weights)
    out["coherence_support"] = (coherent.sum(axis=1) > 0).astype(float)
    out["final_defense_score"] = (0.65*out["cascade_score"] + 0.20*out["net_pressure_score"] + 0.15*out["coherence_support"]).clip(0, 1)
    out["status"] = np.where(out["final_defense_score"] >= config.cascade_trigger, "ALERT", "WATCH")
    out = out.reset_index()
    return out


def latest_layer_snapshot(df: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    return df[df["timestamp"] == timestamp].copy()
