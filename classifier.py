
from __future__ import annotations
import pandas as pd
import numpy as np


def classify_failure(layer_timeline: pd.DataFrame, defense: pd.DataFrame, snapshot: pd.DataFrame) -> dict:
    if defense.empty:
        return {"failure_shape":"no_data","confidence":0.0,"reason":"No timeline."}
    row = defense.sort_values("final_defense_score", ascending=False).iloc[0]
    ts = row["timestamp"]
    layers = layer_timeline[layer_timeline["timestamp"] == ts]
    top_layer = layers.sort_values("cascade_risk", ascending=False).iloc[0] if not layers.empty else None

    # Point failure signals
    point = False
    point_reasons = []
    if not snapshot.empty:
        if pd.to_numeric(snapshot.get("xid_errors", pd.Series([0])), errors="coerce").max() > 0:
            point = True; point_reasons.append("XID error present")
        if pd.to_numeric(snapshot.get("ecc_errors", pd.Series([0])), errors="coerce").max() >= 10:
            point = True; point_reasons.append("high ECC error count")
    if point and float(row["final_defense_score"]) < 0.55:
        return {"failure_shape":"isolated_hardware_point_failure","confidence":0.78,"reason":"; ".join(point_reasons)}

    if top_layer is not None and top_layer["classification_hint"] == "fleet_stress_not_cascade":
        return {"failure_shape":"fleet_stress_not_cascade","confidence":0.70,"reason":"High stress without coherent topology support."}

    if top_layer is not None:
        graph = str(top_layer["graph"])
        risk = float(top_layer["cascade_risk"])
        coherent = int(top_layer["coherent_pairs"])
        if risk >= 0.58 and coherent > 0:
            if graph in ["cooling","thermal"]:
                shape = "thermal_or_cooling_capacity_cascade"
            elif graph == "power":
                shape = "power_domain_cascade"
            elif graph == "fabric":
                shape = "fabric_or_network_metastability"
            elif graph in ["scheduler","tenant"]:
                shape = "scheduler_or_tenant_concentration_cascade"
            else:
                shape = "topology_supported_cascade"
            return {"failure_shape":shape,"confidence":0.82,"reason":f"{graph} graph has risk {risk:.3f} with {coherent} coherent pairs."}

    if float(row["net_pressure_score"]) > 0 and float(row["final_defense_score"]) < 0.55:
        return {"failure_shape":"transient_or_early_forcing","confidence":0.55,"reason":"Risk is rising but below cascade trigger."}

    return {"failure_shape":"watch","confidence":0.50,"reason":"No dominant failure shape found."}
