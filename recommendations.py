
from __future__ import annotations
import pandas as pd


def dcgm_diagnostic_command(shape: str) -> str:
    if "fabric" in shape:
        return "dcgmi diag -r 3  # includes PCIe/NVLink/NCCL-style deeper checks where supported"
    if "hardware" in shape:
        return "dcgmi diag -r 2  # hardware health diagnostic"
    return "dcgmi diag -r 1  # quick deployment/health diagnostic"


def build_recommendations(shape: dict, defense_row: pd.Series, top_layers: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    failure = shape.get("failure_shape","watch")
    priority = "critical" if defense_row.get("final_defense_score",0) >= .78 else "high" if defense_row.get("final_defense_score",0) >= .62 else "warning"

    rows.append({
        "priority": priority,
        "scope": "classification",
        "target": failure,
        "reason": shape.get("reason",""),
        "recommendation": "Treat this as a risk-classification signal; verify before disruptive action.",
    })

    if "cooling" in failure or "thermal" in failure:
        rows.append({"priority": priority, "scope":"cooling", "target":"cooling_loop/rack", "reason":"Cooling or thermal capacity graph is dominant.", "recommendation":"Stop scheduling new high-power jobs into affected loop/rack; inspect fan response, inlet/exhaust temp, coolant flow and pressure."})
    if "power" in failure:
        rows.append({"priority": priority, "scope":"power", "target":"PDU/power_domain", "reason":"Power domain cascade risk.", "recommendation":"Avoid synchronized load steps; stagger checkpointing/training starts; verify PDU phase balance and redundancy margin."})
    if "fabric" in failure or "network" in failure:
        rows.append({"priority": priority, "scope":"fabric", "target":"NVLink/IB/PCIe", "reason":"Fabric instability signs.", "recommendation":"Check link flap/CRC/PCIe replay/NCCL latency; isolate degraded path before collective jobs stall."})
    if "scheduler" in failure or "tenant" in failure:
        rows.append({"priority": priority, "scope":"scheduler", "target":"Kubernetes/Slurm", "reason":"Workload concentration risk.", "recommendation":"Spread new jobs away from risky topology group; avoid topology-blind allocation; move low-priority jobs first."})
    if "hardware_point" in failure:
        rows.append({"priority": "high", "scope":"hardware", "target":"specific GPU/node", "reason":"Likely isolated failure, not cascade.", "recommendation":"Do standard GPU health/RMA workflow; suppress cascade alert unless topology support appears."})

    rows.append({"priority":"info","scope":"diagnostic","target":"DCGM","reason":"Recommended diagnostic command.", "recommendation":dcgm_diagnostic_command(failure)})

    if not snapshot.empty:
        risky = snapshot.sort_values("scale", ascending=False).head(10)
        for _, r in risky.iterrows():
            rows.append({"priority":priority,"scope":"component","target":str(r.get("component_id","")),"reason":f"S{int(r.get('scale',0))}, temp={r.get('temp_c','n/a')}, util={r.get('gpu_util','n/a')}","recommendation":"Inspect job, neighbor components, and telemetry quality for this component."})

    return pd.DataFrame(rows)
