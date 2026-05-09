
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

from telemetry import SCENARIOS, generate_telemetry
from graphs import merge_topology, topology_health
from calibration import build_calibrations, calibration_confidence
from quality import observability_report, observability_confidence
from core import FCCTConfig
from layers import add_metric_scales, add_layer_scales, compute_multigraph_timelines, aggregate_defense_score
from classifier import classify_failure
from validation import validate_against_events


def run_one(scenario, seed=11, steps=120):
    t0=time.time()
    tel, topo = generate_telemetry(steps=steps, rows=2, cols=3, gpus_per_node=2, seed=seed, scenario=scenario)
    merged = merge_topology(tel, topo)
    cfg=FCCTConfig()
    obs=observability_report(merged); obs_conf=observability_confidence(obs)
    cals=build_calibrations(merged); cal_conf=calibration_confidence(merged, cals)
    scaled=add_layer_scales(add_metric_scales(merged, cals, cfg), cfg)
    layer_timeline, graphs = compute_multigraph_timelines(scaled, topo, cfg)
    defense=aggregate_defense_score(layer_timeline, cfg)
    topo_health=topology_health(topo, scaled[["node_id","gpu_id","component_id"]].drop_duplicates(), graphs)
    peak_ts = defense.sort_values("final_defense_score", ascending=False).iloc[0]["timestamp"]
    snapshot = scaled[scaled["timestamp"]==peak_ts]
    shape=classify_failure(layer_timeline, defense, snapshot)
    val, val_sum=validate_against_events(defense, merged, cfg.cascade_trigger)
    return {
        "scenario":scenario,"seed":seed,"runtime_sec":round(time.time()-t0,3),
        "peak_score":round(float(defense["final_defense_score"].max()),3),
        "failure_shape":shape["failure_shape"],"shape_confidence":shape["confidence"],
        "event_frames":int(val["event"].sum()),"alert_frames":int(val["fcct_alert"].sum()),
        "precision":float(val_sum[val_sum["alert"]=="fcct_alert"]["precision"].iloc[0]),
        "recall":float(val_sum[val_sum["alert"]=="fcct_alert"]["recall"].iloc[0]),
        "obs_conf":obs_conf,
        "avg_cal_conf":round(float(cal_conf["confidence"].mean()),3) if not cal_conf.empty else 0,
        "avg_topo_conf":round(float(topo_health["confidence"].mean()),3) if not topo_health.empty else 0,
    }

def run_suite(out_dir="stress_results", steps=120):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows=[]
    for s in SCENARIOS:
        for seed in [7,11,23]:
            rows.append(run_one(s,seed,steps))
    df=pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/"v4_stress_results.csv",index=False)
    summary=df.groupby("scenario").agg(cases=("seed","count"),avg_peak_score=("peak_score","mean"),avg_precision=("precision","mean"),avg_recall=("recall","mean"),common_shape=("failure_shape",lambda x:x.mode().iloc[0] if not x.mode().empty else "")).reset_index()
    summary.to_csv(Path(out_dir)/"v4_stress_summary.csv",index=False)
    return summary

if __name__=="__main__":
    print(run_suite().to_string(index=False))
