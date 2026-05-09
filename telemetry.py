
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "node_id", "gpu_id", "temp_c"]

NUMERIC = [
    "temp_c", "power_w", "gpu_util", "mem_util", "fan_pct", "throttle_flag",
    "ecc_errors", "pcie_replay", "xid_errors", "nvlink_crc", "ib_crc",
    "packet_drops", "nccl_latency_ms", "rack_power_kw", "pdu_current_a",
    "coolant_flow_lpm", "coolant_pressure_psi", "coolant_supply_c", "coolant_return_c",
    "job_queue_depth", "pod_restarts", "oom_events", "storage_iops", "network_tx_gbps"
]

SCENARIOS = [
    "normal_day",
    "isolated_hbm_ecc",
    "xid_fallen_off_bus",
    "slow_hotspot_growth",
    "rack_airflow_blockage",
    "cdu_pressure_drop",
    "coolant_flow_degradation",
    "power_load_step",
    "pdu_n1_loss",
    "fabric_link_flap",
    "nvlink_degradation",
    "pcie_replay_storm",
    "nccl_straggler",
    "scheduler_packing",
    "noisy_neighbor",
    "checkpoint_storm",
    "oom_eviction_loop",
    "sensor_dropout",
    "stale_prometheus",
    "wrong_topology_label",
]


def validate_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Telemetry missing required columns: {missing}")
    out["timestamp"] = out["timestamp"].astype(str)
    out["node_id"] = out["node_id"].astype(str)
    out["gpu_id"] = out["gpu_id"].astype(str)
    for c in NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in NUMERIC:
        if c not in out.columns:
            out[c] = 0.0
    out["throttle_flag"] = out["throttle_flag"].fillna(0).astype(int)
    return out.sort_values(["timestamp", "node_id", "gpu_id"]).reset_index(drop=True)


def load_telemetry_csv(file_or_path) -> pd.DataFrame:
    return validate_telemetry(pd.read_csv(file_or_path))


def validate_topology(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "node_id" not in out.columns:
        raise ValueError("Topology must include node_id")
    out["node_id"] = out["node_id"].astype(str)
    for col in ["rack_id", "zone", "cooling_loop", "power_domain", "fabric_group", "scheduler_pool", "tenant_id"]:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)
    for col in ["row", "col"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_topology_csv(file_or_path) -> pd.DataFrame:
    return validate_topology(pd.read_csv(file_or_path))


def generate_topology(rows: int = 3, cols: int = 4) -> pd.DataFrame:
    records = []
    i = 0
    for r in range(rows):
        for c in range(cols):
            records.append({
                "node_id": f"gpu-node-{i:02d}",
                "rack_id": f"rack-{r+1}",
                "row": r,
                "col": c,
                "zone": "zone-A" if c < cols // 2 else "zone-B",
                "cooling_loop": f"loop-{1 if c < cols // 2 else 2}",
                "power_domain": f"pdu-{r+1}",
                "fabric_group": f"fabric-{c//2}",
                "scheduler_pool": "pool-train" if r < rows-1 else "pool-infer",
                "tenant_id": f"tenant-{c % 3}",
            })
            i += 1
    return validate_topology(pd.DataFrame(records))


def generate_telemetry(
    steps: int = 180,
    rows: int = 3,
    cols: int = 4,
    gpus_per_node: int = 4,
    seed: int = 13,
    scenario: str = "slow_hotspot_growth",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    topo = generate_topology(rows, cols)
    center = (rows // 2, cols // 2)
    records = []

    for t in range(steps):
        progress = t / max(1, steps - 1)
        for _, n in topo.iterrows():
            node = n["node_id"]
            r, c = int(n["row"]), int(n["col"])
            dist = abs(r - center[0]) + abs(c - center[1])

            util_boost = 0
            thermal_boost = 0
            fan_penalty = 0
            power_boost = 0
            fabric_noise = 0
            cooling_bad = 0
            queue_depth = 2
            pod_restarts = 0
            oom_events = 0
            storage_iops = 100
            network_tx = 20
            ecc = xid = pcie = nvcrc = ibcrc = drops = 0
            nccl_lat = 8 + rng.normal(0, 1)
            sensor_dropout = False
            stale = False

            # Scenario shaping.
            if scenario == "normal_day":
                util_boost = 8 * np.sin(t / 20)
            elif scenario == "isolated_hbm_ecc":
                hot = node == "gpu-node-05" and progress > 0.5
                ecc = 10 if hot else 0
                util_boost = 10
            elif scenario == "xid_fallen_off_bus":
                hot = node == "gpu-node-06" and progress > 0.6
                xid = 1 if hot else 0
                util_boost = 10
            elif scenario == "slow_hotspot_growth":
                hot = dist <= 1 and progress > 0.25
                util_boost = 45 if hot else 5
                thermal_boost = 30 * progress if hot else 0
            elif scenario == "rack_airflow_blockage":
                hot = r == 1 and progress > 0.3
                util_boost = 30 if hot else 5
                thermal_boost = 28 if hot else 0
                fan_penalty = 8 if hot else 0
            elif scenario == "cdu_pressure_drop":
                hot = n["cooling_loop"] == "loop-2" and progress > 0.35
                util_boost = 25 if hot else 5
                thermal_boost = 22 if hot else 0
                cooling_bad = 1 if hot else 0
            elif scenario == "coolant_flow_degradation":
                hot = n["cooling_loop"] == "loop-1" and progress > 0.25
                util_boost = 20 if hot else 5
                thermal_boost = 35 * progress if hot else 0
                cooling_bad = 1 if hot else 0
            elif scenario == "power_load_step":
                hot = n["power_domain"] == "pdu-2" and 0.35 < progress < 0.70
                util_boost = 60 if hot else 5
                power_boost = 120 if hot else 0
            elif scenario == "pdu_n1_loss":
                hot = n["power_domain"] == "pdu-2" and progress > 0.35
                util_boost = 35 if hot else 5
                power_boost = 80 if hot else 0
            elif scenario == "fabric_link_flap":
                hot = n["fabric_group"] == "fabric-1" and progress > 0.35
                fabric_noise = 1 if hot else 0
                util_boost = 20
                nccl_lat += 45 if hot else 0
            elif scenario == "nvlink_degradation":
                hot = n["fabric_group"] == "fabric-0" and progress > 0.45
                nvcrc = int(20 * progress) if hot else 0
                nccl_lat += 30 if hot else 0
            elif scenario == "pcie_replay_storm":
                hot = c == 1 and progress > 0.45
                pcie = int(40 * progress) if hot else 0
                nccl_lat += 20 if hot else 0
            elif scenario == "nccl_straggler":
                hot = dist <= 1 and progress > 0.50
                nccl_lat += 60 if hot else 0
                drops = 10 if hot else 0
            elif scenario == "scheduler_packing":
                hot = n["scheduler_pool"] == "pool-train" and c <= 1 and progress > 0.35
                util_boost = 60 if hot else 5
                queue_depth = 30 if hot else 4
            elif scenario == "noisy_neighbor":
                hot = n["tenant_id"] == "tenant-1" and progress > 0.40
                util_boost = 70 if hot else 5
                network_tx = 80 if hot else 20
            elif scenario == "checkpoint_storm":
                hot = 0.45 < progress < 0.58
                util_boost = 35 if hot else 5
                storage_iops = 5000 if hot else 100
                network_tx = 100 if hot else 20
            elif scenario == "oom_eviction_loop":
                hot = c == cols - 1 and progress > 0.4
                util_boost = 50 if hot else 5
                oom_events = 1 if hot and t % 7 == 0 else 0
                pod_restarts = 1 if hot and t % 9 == 0 else 0
            elif scenario == "sensor_dropout":
                sensor_dropout = node == "gpu-node-05" and progress > 0.45
                thermal_boost = 25 if sensor_dropout else 0
            elif scenario == "stale_prometheus":
                stale = node == "gpu-node-06" and progress > 0.40
                util_boost = 40 if stale else 5
                thermal_boost = 20 if stale else 0
            elif scenario == "wrong_topology_label":
                hot = r == 1 and progress > 0.35
                util_boost = 45 if hot else 5
                thermal_boost = 25 if hot else 0

            for g in range(gpus_per_node):
                util = float(np.clip(35 + util_boost + rng.normal(0, 8), 0, 100))
                power = float(np.clip(135 + 1.55 * util + power_boost + rng.normal(0, 14), 70, 520))
                temp = float(42 + 0.10 * power + 0.10 * util + thermal_boost + rng.normal(0, 2))
                fan = float(np.clip(35 + 0.75 * max(0, temp - 55) - fan_penalty + rng.normal(0, 4), 20, 100))
                flow = float(max(15, 42 - cooling_bad * (8 + 15 * progress) + rng.normal(0, 1.2)))
                pressure = float(max(8, 34 - cooling_bad * (4 + 12 * progress) + rng.normal(0, 0.8)))
                supply = float(20 + cooling_bad * 3 + rng.normal(0, 0.4))
                ret = float(supply + 6 + cooling_bad * 5 + rng.normal(0, 0.5))
                rack_power = float(45 + power_boost / 5 + util * 0.12 + rng.normal(0, 2))
                pdu = float(80 + rack_power * 1.1 + rng.normal(0, 3))
                throttle = int(temp > 91 or util > 96 or xid > 0 or ecc >= 10)
                temp_out = np.nan if sensor_dropout else temp
                if stale:
                    temp_out = 62.0
                records.append({
                    "timestamp": f"t{t:04d}",
                    "node_id": node,
                    "gpu_id": g,
                    "temp_c": round(float(temp_out), 2) if pd.notna(temp_out) else np.nan,
                    "power_w": round(power, 2),
                    "gpu_util": round(util, 2),
                    "mem_util": round(float(np.clip(util + rng.normal(0, 12), 0, 100)), 2),
                    "fan_pct": round(fan, 2),
                    "throttle_flag": throttle,
                    "ecc_errors": ecc,
                    "pcie_replay": pcie,
                    "xid_errors": xid,
                    "nvlink_crc": nvcrc,
                    "ib_crc": int(20 * fabric_noise * progress),
                    "packet_drops": drops + int(15 * fabric_noise * progress),
                    "nccl_latency_ms": round(float(nccl_lat), 2),
                    "rack_power_kw": round(rack_power, 2),
                    "pdu_current_a": round(pdu, 2),
                    "coolant_flow_lpm": round(flow, 2),
                    "coolant_pressure_psi": round(pressure, 2),
                    "coolant_supply_c": round(supply, 2),
                    "coolant_return_c": round(ret, 2),
                    "job_queue_depth": queue_depth,
                    "pod_restarts": pod_restarts,
                    "oom_events": oom_events,
                    "storage_iops": storage_iops,
                    "network_tx_gbps": network_tx,
                    "tenant_id": n["tenant_id"],
                    "job_id": f"job-{(t//20)%6}",
                })
    if scenario == "wrong_topology_label":
        topo.loc[topo["node_id"].isin(["gpu-node-04", "gpu-node-05"]), "cooling_loop"] = "wrong-loop"
        topo.loc[topo["node_id"].isin(["gpu-node-04", "gpu-node-05"]), "row"] = 0
    return validate_telemetry(pd.DataFrame(records)), validate_topology(topo)


def save_sample_files(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t, topo = generate_telemetry()
    p1, p2 = out_dir / "sample_ai_infra_telemetry.csv", out_dir / "sample_ai_infra_topology.csv"
    t.to_csv(p1, index=False)
    topo.to_csv(p2, index=False)
    return p1, p2
